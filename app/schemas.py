from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class QuestionsRequest(BaseModel):
    initial_prompt: str = Field(min_length=1, max_length=20000)
    mode: Optional[str] = Field(default="general", description="general | prompt_questionnaire | sdlc")
    sdlc_stage: Optional[str] = Field(default=None, description="inception | elaboration | construction | transition")


class QuestionsResponse(BaseModel):
    questions: List[str]


class RefineRequest(BaseModel):
    initial_prompt: str = Field(min_length=1, max_length=20000)
    answers: Dict[str, str]
    mode: Optional[str] = Field(default="general", description="general | prompt_questionnaire | sdlc")
    sdlc_stage: Optional[str] = Field(default=None, description="inception | elaboration | construction | transition")


class RefineResponse(BaseModel):
    refined_prompt: str


class AcceptRequest(BaseModel):
    initial_prompt: str = Field(min_length=1, max_length=20000)
    answers: Dict[str, str]
    refined_prompt: str = Field(min_length=1, max_length=40000)
    mode: Optional[str] = Field(default="general", description="general | prompt_questionnaire | sdlc")
    sdlc_stage: Optional[str] = Field(default=None, description="inception | elaboration | construction | transition")


class AcceptResponse(BaseModel):
    human_friendly_prompt: str
    llm_optimized_prompt: str


class SaveDBRequest(BaseModel):
    initial_prompt: str
    questions: List[str] = []
    answers: Dict[str, str] = {}
    refined_prompt: Optional[str] = None
    human_friendly_prompt: Optional[str] = None
    llm_optimized_prompt: Optional[str] = None


class SaveDBResponse(BaseModel):
    record_id: int


class SaveFileRequest(BaseModel):
    human_friendly_prompt: str
    llm_optimized_prompt: str
    filename: Optional[str] = None


class SaveFileResponse(BaseModel):
    saved_path: str
