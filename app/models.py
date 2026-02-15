from __future__ import annotations

from datetime import datetime
from sqlalchemy import Integer, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class PromptRecord(Base):
    __tablename__ = "prompt_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    initial_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    questions_json: Mapped[str] = mapped_column(Text, nullable=True)
    answers_json: Mapped[str] = mapped_column(Text, nullable=True)

    refined_prompt: Mapped[str] = mapped_column(Text, nullable=True)
    human_friendly_prompt: Mapped[str] = mapped_column(Text, nullable=True)
    llm_optimized_prompt: Mapped[str] = mapped_column(Text, nullable=True)
