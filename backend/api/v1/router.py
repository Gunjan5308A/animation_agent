from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter()


class GenerateRequest(BaseModel):
	prompt: str = Field(..., min_length=1)
	options: dict[str, Any] = Field(default_factory=dict)


@router.post("/generate")
async def generate(request: GenerateRequest) -> dict[str, Any]:
	"""Accept a generation request."""
	return {
		"status": "accepted",
		"prompt": request.prompt,
		"options": request.options,
	}
