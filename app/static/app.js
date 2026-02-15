const el = (id) => document.getElementById(id);

const statusEl = el("status");
const errorEl = el("error");
const saveStatusEl = el("saveStatus");
const step3StatusEl = el("step3Status");

const initialPromptEl = el("initialPrompt");
const questionsContainer = el("questionsContainer");
const refinedPromptEl = el("refinedPrompt");
const humanPromptEl = el("humanPrompt");
const llmPromptEl = el("llmPrompt");
const fileNameEl = el("fileName");

const btnQuestions = el("btnQuestions");
const btnRefine = el("btnRefine");
const btnAccept = el("btnAccept");
const btnSaveDb = el("btnSaveDb");
const btnSaveFile = el("btnSaveFile");
const btnCopy = el("btnCopy");

const sectionQuestions = el("sectionQuestions");
const btnToggleQuestions = el("btnToggleQuestions");

const modeSelect = el("modeSelect");
const stageWrap = el("stageWrap");
const stageSelect = el("stageSelect");

let currentQuestions = [];
let currentAnswers = {};

function getMode() {
  return (modeSelect?.value || "general").trim();
}

function getStage() {
  return (stageSelect?.value || "inception").trim();
}

function resetWorkflow() {
  currentQuestions = [];
  currentAnswers = {};
  questionsContainer.innerHTML = "";
  refinedPromptEl.value = "";
  humanPromptEl.value = "";
  llmPromptEl.value = "";
  setStatus("");
  setStep3Status("");
  setSaveStatus("");
  setError("");
  btnRefine.disabled = true;
  btnAccept.disabled = true;
  btnSaveDb.disabled = true;
  btnSaveFile.disabled = true;
  btnCopy.disabled = true;
}

function setPanelVisible(panelEl, visible) {
  if (!panelEl) return;
  panelEl.classList.add("motion-panel");
  if (visible) {
    panelEl.classList.remove("mode-hidden");
    panelEl.removeAttribute("aria-hidden");
  } else {
    panelEl.classList.add("mode-hidden");
    panelEl.setAttribute("aria-hidden", "true");
  }
}

function updateModeUI() {
  const mode = getMode();
  const isGeneral = mode === "general";
  const isSdlc = mode === "sdlc";

  setPanelVisible(stageWrap, isSdlc);
  setPanelVisible(sectionQuestions, !isGeneral);
  if (btnQuestions) {
    btnQuestions.textContent = isGeneral ? "Refine prompr" : "Generate clarifying questions";
  }

  // In general mode, the questions workflow is not used.
  if (isGeneral) {
    btnRefine.disabled = true;
  }
}

modeSelect?.addEventListener("change", () => {
  updateModeUI();
  resetWorkflow();
});

stageSelect?.addEventListener("change", () => {
  if (getMode() === "sdlc") {
    resetWorkflow();
  }
});

function sanitizeRefinedPrompt(text) {
  if (!text) return "";
  return text.replace(/(<\|endoftext\|>\s*)+/g, "").trim();
}

function setStatus(msg) { statusEl.textContent = msg || ""; }
function setStep3Status(msg) { step3StatusEl.textContent = msg || ""; }
function setError(msg) {
  if (!msg) { errorEl.hidden = true; errorEl.textContent = ""; return; }
  errorEl.hidden = false;
  errorEl.textContent = msg;
}
function setSaveStatus(msg) { saveStatusEl.textContent = msg || ""; }

function setQuestionsCollapsed(collapsed) {
  if (!sectionQuestions || !btnToggleQuestions) return;
  sectionQuestions.classList.toggle("is-collapsed", !!collapsed);
  btnToggleQuestions.setAttribute("aria-expanded", collapsed ? "false" : "true");
  btnToggleQuestions.textContent = collapsed ? "Expand" : "Collapse";
}

btnToggleQuestions?.addEventListener("click", () => {
  const collapsed = sectionQuestions.classList.contains("is-collapsed");
  setQuestionsCollapsed(!collapsed);
});

refinedPromptEl.addEventListener("input", () => {
  btnAccept.disabled = !refinedPromptEl.value.trim();
});

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data?.detail || "Request failed.");
  }
  return data;
}

function renderQuestions(questions) {
  questionsContainer.innerHTML = "";
  const fieldset = document.createElement("fieldset");
  const legend = document.createElement("legend");
  legend.textContent = "Questions";
  legend.className = "sr-only";
  fieldset.appendChild(legend);

  questions.forEach((q, idx) => {
    const qId = `q_${idx}`;

    const wrapper = document.createElement("div");
    wrapper.style.marginBottom = "0.9rem";

    const label = document.createElement("label");
    label.className = "label";
    label.setAttribute("for", qId);
    label.textContent = `${idx + 1}. ${q}`;

    const input = document.createElement("textarea");
    input.className = "textarea";
    input.id = qId;
    input.rows = 3;
    input.setAttribute("aria-label", `Answer for question ${idx + 1}`);

    input.addEventListener("input", () => {
      currentAnswers[q] = input.value;
    });

    wrapper.appendChild(label);
    wrapper.appendChild(input);
    fieldset.appendChild(wrapper);
  });

  questionsContainer.appendChild(fieldset);
  btnRefine.disabled = questions.length === 0;
}

async function runGeneralGenerate(initial) {
  // General mode: refine with inferred structure, then accept to produce final variants.
  currentQuestions = [];
  currentAnswers = {};
  btnQuestions.disabled = true;
  btnAccept.disabled = true;
  btnSaveDb.disabled = true;
  btnSaveFile.disabled = true;
  btnCopy.disabled = true;
  setStatus("Generating prompts...");
  setStep3Status("Drafting refined prompt...");
  refinedPromptEl.value = "";
  humanPromptEl.value = "";
  llmPromptEl.value = "";

  const mode = getMode();

  try {
    const refineData = await postJSON("/api/refine", {
      initial_prompt: initial,
      answers: {},
      mode,
      sdlc_stage: null,
    });
    refinedPromptEl.value = sanitizeRefinedPrompt(refineData.refined_prompt || "");
    btnAccept.disabled = !refinedPromptEl.value.trim();
    setStep3Status("Generating final prompt variants...");

    const acceptData = await postJSON("/api/accept", {
      initial_prompt: initial,
      answers: {},
      refined_prompt: refinedPromptEl.value.trim(),
      mode,
      sdlc_stage: null,
    });

    humanPromptEl.value = sanitizeRefinedPrompt(acceptData.human_friendly_prompt || "");
    llmPromptEl.value = sanitizeRefinedPrompt(acceptData.llm_optimized_prompt || "");

    const ready = !!(humanPromptEl.value.trim() && llmPromptEl.value.trim());
    btnSaveDb.disabled = !ready;
    btnSaveFile.disabled = !ready;
    btnCopy.disabled = !ready;

    setStatus("Generation complete.");
    setStep3Status("Final variants ready.");
    refinedPromptEl.focus();
  } catch (e) {
    setError(e.message || String(e));
    setStatus("");
    setStep3Status("");
  } finally {
    btnQuestions.disabled = false;
  }
}

btnQuestions.addEventListener("click", async () => {
  setError("");
  setSaveStatus("");
  setStep3Status("");

  const initial = initialPromptEl.value.trim();
  if (!initial) {
    setError("Please enter an initial prompt first.");
    initialPromptEl.focus();
    return;
  }

  const mode = getMode();
  const stage = getStage();

  // General mode bypasses questions.
  if (mode === "general") {
    await runGeneralGenerate(initial);
    return;
  }

  btnQuestions.disabled = true;
  btnRefine.disabled = true;
  btnAccept.disabled = true;
  btnSaveDb.disabled = true;
  btnSaveFile.disabled = true;
  btnCopy.disabled = true;

  setStatus("Generating clarifying questions...");
  try {
    const data = await postJSON("/api/questions", {
      initial_prompt: initial,
      mode,
      sdlc_stage: mode === "sdlc" ? stage : null,
    });
    currentQuestions = data.questions || [];
    currentAnswers = {};
    renderQuestions(currentQuestions);
    setStatus(`Generated ${currentQuestions.length} questions.`);

    // Expand questions when freshly generated
    setQuestionsCollapsed(false);

    const first = document.getElementById("q_0");
    if (first) first.focus();
  } catch (e) {
    setError(e.message || String(e));
    setStatus("");
  } finally {
    btnQuestions.disabled = false;
  }
});

btnRefine.addEventListener("click", async () => {
  setError("");
  setSaveStatus("");

  const initial = initialPromptEl.value.trim();
  if (!initial) { setError("Initial prompt is missing."); return; }
  if (!currentQuestions.length) { setError("No questions generated yet."); return; }

  const mode = getMode();
  const stage = getStage();

  // Collapse questions so Step 3 status is visible
  setQuestionsCollapsed(true);

  btnRefine.disabled = true;
  btnAccept.disabled = true;

  setStep3Status("Refining prompt...");
  refinedPromptEl.value = "";

  try {
    const answers = {};
    currentQuestions.forEach((q) => { answers[q] = currentAnswers[q] || ""; });

    const data = await postJSON("/api/refine", {
      initial_prompt: initial,
      answers,
      mode,
      sdlc_stage: mode === "sdlc" ? stage : null,
    });
    refinedPromptEl.value = sanitizeRefinedPrompt(data.refined_prompt || "");
    btnAccept.disabled = !refinedPromptEl.value.trim();

    setStep3Status("Refinement complete.");
    refinedPromptEl.focus();
  } catch (e) {
    setError(e.message || String(e));
    setStep3Status("");
  } finally {
    btnRefine.disabled = false;
  }
});

btnAccept.addEventListener("click", async () => {
  const mode = getMode();
  const stage = getStage();
  setError("");
  setSaveStatus("");

  const initial = initialPromptEl.value.trim();
  const refined = refinedPromptEl.value.trim();
  if (!initial || !refined) { setError("Initial and refined prompts are required."); return; }

  btnAccept.disabled = true;
  setStep3Status("Generating final prompt variants...");
  try {
    const answers = {};
    currentQuestions.forEach((q) => (answers[q] = currentAnswers[q] || ""));

    const data = await postJSON("/api/accept", {
      initial_prompt: initial,
      answers,
      refined_prompt: refined,
      mode,
      sdlc_stage: mode === "sdlc" ? stage : null,
    });

    humanPromptEl.value = sanitizeRefinedPrompt(data.human_friendly_prompt || "");
    llmPromptEl.value = sanitizeRefinedPrompt(data.llm_optimized_prompt || "");

    const ready = !!(humanPromptEl.value.trim() && llmPromptEl.value.trim());
    btnSaveDb.disabled = !ready;
    btnSaveFile.disabled = !ready;
    btnCopy.disabled = !ready;

    setStep3Status("Final variants ready.");
    humanPromptEl.focus();
  } catch (e) {
    setError(e.message || String(e));
    setStep3Status("");
  } finally {
    btnAccept.disabled = false;
  }
});

btnSaveDb.addEventListener("click", async () => {
  setError("");
  const initial = initialPromptEl.value.trim();
  const refined = refinedPromptEl.value.trim();
  const human = humanPromptEl.value.trim();
  const llm = llmPromptEl.value.trim();

  if (!human || !llm) { setError("Nothing to save yet."); return; }

  setSaveStatus("Saving to DB...");
  btnSaveDb.disabled = true;
  try {
    const answers = {};
    currentQuestions.forEach((q) => (answers[q] = currentAnswers[q] || ""));

    const data = await postJSON("/api/save_db", {
      initial_prompt: initial,
      questions: currentQuestions,
      answers,
      refined_prompt: refined,
      human_friendly_prompt: human,
      llm_optimized_prompt: llm,
    });
    setSaveStatus(`Saved to DB. Record ID: ${data.record_id}`);
  } catch (e) {
    setError(e.message || String(e));
    setSaveStatus("");
  } finally {
    btnSaveDb.disabled = false;
  }
});

btnSaveFile.addEventListener("click", async () => {
  setError("");
  const human = humanPromptEl.value.trim();
  const llm = llmPromptEl.value.trim();

  if (!human || !llm) { setError("Nothing to save yet."); return; }

  setSaveStatus("Saving file...");
  btnSaveFile.disabled = true;
  try {
    const data = await postJSON("/api/save_file", {
      human_friendly_prompt: human,
      llm_optimized_prompt: llm,
      filename: fileNameEl.value.trim() || null,
    });
    setSaveStatus(`Saved file: ${data.saved_path}`);
  } catch (e) {
    setError(e.message || String(e));
    setSaveStatus("");
  } finally {
    btnSaveFile.disabled = false;
  }
});

btnCopy.addEventListener("click", async () => {
  setError("");
  const llm = llmPromptEl.value.trim();
  if (!llm) { setError("Nothing to copy yet."); return; }
  try {
    await navigator.clipboard.writeText(llm);
    setSaveStatus("Copied LLM-optimized prompt to clipboard.");
  } catch (e) {
    setError("Clipboard copy failed. Your browser may block clipboard access on insecure contexts.");
  }
});

// Initial UI state
updateModeUI();
