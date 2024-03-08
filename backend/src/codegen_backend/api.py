from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

from .llm_backend import GenerationPrompt, AdjustPrompt, BackendManager, LLMBackend


# ------------------------------- #
# --------- Data models --------- #
# ------------------------------- #


class GenerationArgs(BaseModel):
    model_name: str
    language: str
    original_prompt: str


class AdjustArgs(BaseModel):
    model_name: str
    language: str
    original_prompt: str
    reference_code: str
    feedback_prompt: str


class CodeOut(BaseModel):
    code_out: str


# ------------------------------- #
# ------------ Setup ------------ #
# ------------------------------- #


# Instantiate backend manager
backend_manager_object = BackendManager("config/backend_manager.yaml")


# Create dependency to make manager available to endpoints
async def backend_manager() -> BackendManager:
    return backend_manager_object


# To shorten dependency arguments
manager_dependency = Annotated[BackendManager, Depends(backend_manager)]

# Instantiate app
app = FastAPI()


# ------------------------------- #
# ---------- Functions ---------- #
# ------------------------------- #


def load_backend(manager: BackendManager, model_name: str) -> LLMBackend:
    """Loads the backend using the backend manager. Task-agnostic."""
    try:
        return manager.load_backend(model_name)
    except:
        raise HTTPException(status_code=404, detail="Model name not found")


# ------------------------------- #
# ---------- Endpoints ---------- #
# ------------------------------- #


@app.get("/")
def root():
    return {"health_check": "OK"}


@app.post(
    "/generate",
    response_model=CodeOut,
    description="Generates new code based on the prompt",
)
def generate_endpoint(payload: GenerationArgs, manager: manager_dependency):
    backend = load_backend(manager, payload.model_name)
    try:
        # Create prompt object without model_name
        prompt_dict = payload.model_dump(mode="python", exclude={"model_name"})
        prompt = GenerationPrompt(**prompt_dict)
        code_out = backend.generate_code(prompt)
        return {"code_out": code_out}
    except:
        raise HTTPException(status_code=424, detail=f"Generation failed")


@app.post(
    "/adjust",
    response_model=CodeOut,
    description="Modifies code based on the prompt, previous code and feedback",
)
def adjust_endpoint(payload: AdjustArgs, manager: manager_dependency):
    backend = load_backend(manager, payload.model_name)
    try:
        # Create prompt object without model_name
        prompt_dict = payload.model_dump(mode="python", exclude={"model_name"})
        prompt = AdjustPrompt(**prompt_dict)
        code_out = backend.adjust_code(prompt)
        return {"code_out": code_out}
    except:
        raise HTTPException(status_code=424, detail=f"Generation failed")
