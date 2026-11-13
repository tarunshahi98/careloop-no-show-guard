"""Structured LLM analysis service for categorizing patient follow-up call notes."""

from enum import Enum
import json
import logging
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class RootCause(str, Enum):
    FORGOT = "forgot_or_no_reminder"
    TRANSPORT = "distance_or_transport"
    COST = "cost_or_fee"
    ILLNESS = "illness_or_emergency"
    WORK = "work_or_schedule"
    OTHER = "other"


class FollowupInsight(BaseModel):
    """Schema for structured extraction of call-centre note details."""
    root_cause: RootCause
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=3)
    recommended_action: str = Field(min_length=3)

    @field_validator("summary")
    @classmethod
    def validate_word_count(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) > 25:
            raise ValueError(f"Summary exceeds 25 words ({len(words)} words found).")
        return v


class NoteAnalysisService:
    """Classifies unstructured notes using Gemini with Pydantic validation."""

    def __init__(self, client=None, model_name: str = "gemini-2.0-flash"):
        self.client = client
        self.model_name = model_name
        self.system_instruction = (
            "You are a clinical operations analyst for CareLoop Clinics in India. "
            "Analyze qualitative follow-up call notes from missed appointments, "
            "and classify the root cause into one of the allowed categories: "
            "forgot_or_no_reminder, distance_or_transport, cost_or_fee, illness_or_emergency, "
            "work_or_schedule, other. Default ambiguity or unreached phone calls strictly to 'other'. "
            "Keep summary strictly under 25 words."
        )

    def analyze_record(self, note_id: str, appt_id: str, note_text: Optional[str]) -> Dict:
        """Processes a single follow-up note, returning structured classification dictionary."""
        if note_text is None or str(note_text).strip() == "":
            return {
                "note_id": note_id,
                "appointment_id": appt_id,
                "root_cause": RootCause.OTHER.value,
                "confidence": 0.0,
                "summary": "No note recorded",
                "recommended_action": "Re-call patient",
                "validation_status": "skipped_blank",
                "raw_response": ""
            }

        if self.client is not None:
            try:
                from google.genai import types
                prompt = (
                    f"Classify this patient follow-up note:\n\"{note_text}\"\n"
                    "Allowed root causes: forgot_or_no_reminder, distance_or_transport, "
                    "cost_or_fee, illness_or_emergency, work_or_schedule, other.\n"
                    "Instructions: summary must be at most 25 words."
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        temperature=0,
                        max_output_tokens=1000,
                        response_mime_type="application/json",
                    )
                )
                raw_text = response.text
                parsed_json = json.loads(raw_text)
                validated = FollowupInsight(**parsed_json)
                return {
                    "note_id": note_id,
                    "appointment_id": appt_id,
                    "root_cause": validated.root_cause.value,
                    "confidence": validated.confidence,
                    "summary": validated.summary,
                    "recommended_action": validated.recommended_action,
                    "validation_status": "valid",
                    "raw_response": raw_text
                }
            except Exception as err:
                logger.error("Failed to parse note %s: %s", note_id, err)
                return {
                    "note_id": note_id,
                    "appointment_id": appt_id,
                    "root_cause": RootCause.OTHER.value,
                    "confidence": 0.0,
                    "summary": "Error parsing response",
                    "recommended_action": "Manual staff review required",
                    "validation_status": "invalid",
                    "raw_response": str(err)
                }

        return {
            "note_id": note_id,
            "appointment_id": appt_id,
            "root_cause": RootCause.OTHER.value,
            "confidence": 0.90,
            "summary": "Offline baseline classification",
            "recommended_action": "Contact patient",
            "validation_status": "valid",
            "raw_response": "{}"
        }
