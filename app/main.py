from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import init_db, get_db, APP_DATA_DIR
from .models import PromptRecord
from .schemas import (
    QuestionsRequest, QuestionsResponse,
    RefineRequest, RefineResponse,
    AcceptRequest, AcceptResponse,
    SaveDBRequest, SaveDBResponse,
    SaveFileRequest, SaveFileResponse,
)
from .llm import LLMClient, llm_json_list


load_dotenv()  # loads .env if present

app = FastAPI(title="LLM Interactive Prompt Builder", version="0.1.2")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/images", StaticFiles(directory=str(BASE_DIR.parent / "images")), name="images")


# ---- LLM call safety: concurrency + throttling ----
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "1"))
MIN_SECONDS_BETWEEN_LLM_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_LLM_CALLS", "1.5"))

_llm_sema = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
_last_call_by_client: Dict[str, float] = {}
_last_call_lock = asyncio.Lock()


def _client_key(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    return host


async def _throttle(request: Request) -> None:
    """Prevent flooding by enforcing a minimum spacing between LLM calls per client."""
    if MIN_SECONDS_BETWEEN_LLM_CALLS <= 0:
        return
    key = _client_key(request)
    now = time.time()
    async with _last_call_lock:
        last = _last_call_by_client.get(key, 0.0)
        delta = now - last
        if delta < MIN_SECONDS_BETWEEN_LLM_CALLS:
            wait = MIN_SECONDS_BETWEEN_LLM_CALLS - delta
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests. Please wait {wait:.1f}s and try again.",
            )
        _last_call_by_client[key] = now


class _LLMGuard:
    async def __aenter__(self):
        await _llm_sema.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        _llm_sema.release()


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # Remove leading ```lang? and trailing ```
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from arbitrary text."""
    s = _strip_code_fences(text)

    # Fast path: exact JSON
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    start = s.find("{")
    if start == -1:
        return None

    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1].strip()
    return None


def _parse_json_lenient(raw: str) -> Dict:
    """Parse JSON from LLM output even if it includes extra text."""
    candidate = _extract_first_json_object(raw)
    if not candidate:
        raise ValueError("No JSON object found in model output.")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON extracted from model output: {e}") from e


@app.on_event("startup")
def _startup():
    init_db()
    (APP_DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "provider": os.getenv("LLM_PROVIDER", "local"),
        },
    )


SYSTEM_QUESTIONS_QUESTIONNAIRE = """You are a prompt-design assistant.
Generate targeted clarifying questions that reduce ambiguity and surface constraints.
Return STRICT JSON ONLY in this shape:
{
  "questions": ["Q1", "Q2", ...]
}
Rules:
- Default to 7 questions unless the adaptive rules below suggest otherwise.
- Adaptive count:
  - If the initial prompt is less than one sentence (very short/fragmentary), ask 8 to 10 questions.
  - If the initial prompt is already well-structured, ask 4 to 6 questions.
    Treat it as well-structured ONLY if it meets at least two of these three signals:
      1) Section labels: includes at least two distinct labeled fields such as "Role:", "Goal:", "Context:", "Constraints:",
         "Output format:", "Examples:", "Success criteria:", "Tone:", "Failure handling:", "Edge cases:".
      2) Formatting: contains a heading marker (#/##/###) OR a divider line (---/===) OR 3+ bullets/numbered items.
      3) Output specificity: explicitly states an output format or deliverable (e.g., "Output format:", "Return:", "Deliverable:").
  - Otherwise, ask 7 questions.
- Prefer concrete, actionable questions
- Avoid duplicates
- Prioritize clarifying: goal, audience/role, output format, constraints, edge cases, success criteria, examples, tone/style, failure handling.
- Use judgment: include only what is necessary; you do not need to cover every topic every time.
"""


SYSTEM_QUESTIONS_SDLC = """You are a prompt-design assistant for an SDLC prompt-tree workflow.
Your job is to generate targeted clarifying questions for the given SDLC stage.

Return STRICT JSON ONLY in this shape:
{
  "questions": ["Q1", "Q2", ...]
}

Stage: {stage}

Rules:
- Default to 7 questions.
- Ask questions that help complete the stage template with concrete, testable details.
- Avoid duplicates.
- Prefer questions that map to the SDLC Structured Ingredients for the selected stage.

Stage focus:
- inception (what is wanted):
  - stakeholder map (user/player, sponsor, developer/maintainer, external systems)
  - problem statement (what, why now, consequence of not solving)
  - vision / future state (one-paragraph outcome)
  - requirements (functional, non-functional, constraints)
  - user/player experience goals (emotional goals, interaction style, success signals)
  - acceptance criteria (measurable pass/fail, edge cases, traceable IDs)
  - scope boundaries (IN / OUT)
  - priority matrix (must/should/could/won't)
  - early risk register (technical, resource, market, unknowns)
  - phase definition of done (requirements validated, acceptance criteria defined, stakeholder alignment)
- elaboration (what is possible — who/what/when/where/why/how):
  - system context (external systems, data flows, integration points)
  - capability map (hardware/software/SDK or engine limitations)
  - technical constraints (memory, performance, network, platform limits)
  - architecture blueprint (layers, module boundaries, dependency rules)
  - data model (entities, relationships, persistence)
  - state model (system/player/failure states, transitions)
  - interaction model (inputs, feedback, control loops)
  - security/compliance model (authn/authz, data protection)
  - performance model (latency, load, scaling)
  - traceability links (design mapped to requirements, coverage)
  - prototype validation (risky components tested early)
- construction (input | process | output):
  - logical architecture (package layering, module contracts: inputs/outputs/owned state/deps/failure behavior)
  - code-level structure (structs, enums, classes, functions; explicit types and lifecycles)
  - logic flow (validation, business rules, state updates, events, output formatting)
  - interface elements (2D/3D components, interactions, feedback, accessibility with triggers/state/feedback)
  - testing (unit/integration, coverage targets, edge cases, performance, UI validation)
  - governance (coding standards, reviews, static analysis, CI/CD, reproducibility)
- transition (deliverable product):
  - deliverables (build, install, config, compatibility)
  - coverage report (requirements/design/module/test)
  - documentation (architecture, data model, APIs, prompt hierarchy, limitations)
  - completion status (requirements/features, open issues, risk updates)
  - user validation (beta feedback, metrics)
  - production readiness (performance, security, backups, monitoring)
  - cross-phase essentials (traceability, version control, change/decision logs, risk tracking, ownership, coverage gates)
"""

SYSTEM_REFINE_QUESTIONNAIRE = """You are a prompt-design assistant.
Given an initial prompt and the user's answers, produce a refined prompt.
Return plain text only (no code fences). Keep it concise but explicit.
"""


SYSTEM_REFINE_GENERAL = """You are a prompt-design assistant.
Given an initial prompt, expand it into a high-quality prompt even if details are missing.
Infer sensible defaults and add structure.

Return plain text only (no code fences).

Target structure (adapt as needed):
- Role
- Goal
- Context
- Constraints
- Inputs
- Output format
- Examples (optional)
- Success criteria
- Edge cases / failure handling
- Tone / style
"""


SYSTEM_REFINE_SDLC = """You are a prompt-design assistant for an SDLC prompt-tree workflow.
Given an initial prompt and the user's answers, produce a refined prompt that matches the chosen SDLC stage template.
Return plain text only (no code fences).

Stage: {stage}

Templates:

inception (REQUIREMENT):
- Player-facing intent
- Problem / need
- Scope (in / out)
- Success metrics
- Acceptance criteria
- Verification hints
- Constraints (platform, budget, performance, safety)

elaboration (DESIGN):
- Responsibilities
- State
- Inputs / outputs
- Constraints
- Failure handling / edge cases
- Integration points

construction (MODULE):
- Public contract (APIs)
- Owned state
- Integration points
- Artifacts
- Testability notes
- Traceability hints (what requirement/design it satisfies)

transition (RELEASE/BUILD):
- Build slice selection
- Assembly plan
- Export / packaging configuration
- Verification mapping
- Traceability to modules
"""

SYSTEM_ACCEPT = """You are a prompt-design assistant.
Given an initial prompt, answers, and a refined prompt, produce TWO final prompt variants.

Return STRICT JSON ONLY (no markdown, no commentary, no code fences, no extra text) in this exact shape:
{
  "human_friendly_prompt": "...",
  "llm_optimized_prompt": "..."
}

Rules:
- Keep the two variants semantically aligned.
- Human-friendly: clear, natural language, easy to read.
- LLM-optimized: structured, explicit instructions with roles, constraints, and output format.
- The LLM-optimized variant should include:
  - Role
  - Goal
  - Constraints
  - Output format
  - Edge cases / failure handling (if relevant)
"""


def _norm_mode(mode: Optional[str]) -> str:
    m = (mode or "").strip().lower()
    if m in {"general", "prompt_questionnaire", "sdlc"}:
        return m
    # tolerate UI label variants
    if m in {"questionnaire", "prompt questionnaire", "prompt-questionnaire"}:
        return "prompt_questionnaire"
    return "prompt_questionnaire"


def _norm_stage(stage: Optional[str]) -> Optional[str]:
    if not stage:
        return None
    s = stage.strip().lower()
    aliases = {
        "inception": "inception",
        "elaboration": "elaboration",
        "construction": "construction",
        "transition": "transition",
    }
    return aliases.get(s)


def _system_questions(mode: str, stage: Optional[str]) -> str:
    if mode == "sdlc":
        return SYSTEM_QUESTIONS_SDLC.format(stage=stage or "inception")
    return SYSTEM_QUESTIONS_QUESTIONNAIRE


def _system_refine(mode: str, stage: Optional[str]) -> str:
    if mode == "general":
        return SYSTEM_REFINE_GENERAL
    if mode == "sdlc":
        return SYSTEM_REFINE_SDLC.format(stage=stage or "inception")
    return SYSTEM_REFINE_QUESTIONNAIRE


@app.post("/api/questions", response_model=QuestionsResponse)
async def api_questions(request: Request, payload: QuestionsRequest):
    await _throttle(request)
    mode = _norm_mode(payload.mode)
    stage = _norm_stage(payload.sdlc_stage)
    async with _LLMGuard():
        client = LLMClient()
        try:
            questions = await llm_json_list(
                client,
                _system_questions(mode, stage),
                (
                    f"Mode: {mode}\n"
                    f"Stage: {stage or ''}\n\n"
                    f"Initial prompt:\n{payload.initial_prompt.strip()}\n\nGenerate questions now."
                ),
            )
            return QuestionsResponse(questions=questions)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/refine", response_model=RefineResponse)
async def api_refine(request: Request, payload: RefineRequest):
    await _throttle(request)
    mode = _norm_mode(payload.mode)
    stage = _norm_stage(payload.sdlc_stage)
    answers_block = "\n".join([f"- {k}: {v}" for k, v in payload.answers.items() if v.strip()])
    user = (
        f"Mode: {mode}\nStage: {stage or ''}\n\n"
        f"Initial prompt:\n{payload.initial_prompt.strip()}\n\n"
        f"User answers:\n{answers_block}\n\n"
        "Produce a refined prompt."
    )
    async with _LLMGuard():
        client = LLMClient()
        try:
            refined = await client.complete(_system_refine(mode, stage), user)
            return RefineResponse(refined_prompt=refined.strip())
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/accept", response_model=AcceptResponse)
async def api_accept(request: Request, payload: AcceptRequest):
    await _throttle(request)
    mode = _norm_mode(payload.mode)
    stage = _norm_stage(payload.sdlc_stage)
    answers_block = "\n".join([f"- {k}: {v}" for k, v in payload.answers.items() if v.strip()])
    user = (
        f"Mode: {mode}\nStage: {stage or ''}\n\n"
        f"Initial prompt:\n{payload.initial_prompt.strip()}\n\n"
        f"User answers:\n{answers_block}\n\n"
        f"Refined prompt:\n{payload.refined_prompt.strip()}\n\n"
        "Generate final variants now."
    )
    async with _LLMGuard():
        client = LLMClient()
        raw = await client.complete(SYSTEM_ACCEPT, user)

    try:
        obj = _parse_json_lenient(raw)
    except Exception as e:
        snippet = raw.strip().replace("\r", "")[:600]
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse JSON from model output: {e}. Output starts with: {snippet}",
        )

    human = str(obj.get("human_friendly_prompt", "")).strip()
    opt = str(obj.get("llm_optimized_prompt", "")).strip()
    if not human or not opt:
        raise HTTPException(status_code=400, detail="JSON missing required fields or they were empty.")

    return AcceptResponse(
        human_friendly_prompt=human,
        llm_optimized_prompt=opt,
    )


@app.post("/api/save_db", response_model=SaveDBResponse)
def api_save_db(payload: SaveDBRequest, db: Session = Depends(get_db)):
    rec = PromptRecord(
        initial_prompt=payload.initial_prompt,
        questions_json=json.dumps(payload.questions, ensure_ascii=False),
        answers_json=json.dumps(payload.answers, ensure_ascii=False),
        refined_prompt=payload.refined_prompt,
        human_friendly_prompt=payload.human_friendly_prompt,
        llm_optimized_prompt=payload.llm_optimized_prompt,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return SaveDBResponse(record_id=rec.id)


@app.post("/api/save_file", response_model=SaveFileResponse)
def api_save_file(payload: SaveFileRequest):
    exports_dir = APP_DATA_DIR / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = (payload.filename or f"final_prompts_{ts}.txt").strip()

    filename = "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".", " ")).strip().replace(" ", "_")
    if not filename.lower().endswith(".txt"):
        filename += ".txt"

    path = exports_dir / filename
    content = (
        "=== Human-friendly Prompt ===\n\n"
        f"{payload.human_friendly_prompt.strip()}\n\n"
        "=== LLM-optimized Prompt ===\n\n"
        f"{payload.llm_optimized_prompt.strip()}\n"
    )
    path.write_text(content, encoding="utf-8")
    return SaveFileResponse(saved_path=str(path.resolve()))
