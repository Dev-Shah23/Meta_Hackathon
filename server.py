"""
FastAPI server exposing the Email Triage environment via HTTP.
Endpoints mirror the OpenEnv spec.
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment import EmailTriageEnv, Action

app = FastAPI(title="Email Triage Environment", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env per task (task is set at reset time)
_envs: dict[int, EmailTriageEnv] = {}


class ResetRequest(BaseModel):
    task: int = 1


class StepRequest(BaseModel):
    task: int = 1
    action: Action


def _get_env(task: int) -> EmailTriageEnv:
    if task not in _envs:
        raise HTTPException(status_code=400, detail=f"Task {task} not initialised. Call /reset first.")
    return _envs[task]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    task = req.task if req else 1
    env = EmailTriageEnv(task=task)
    obs = env.reset()
    _envs[task] = env
    return {"observation": obs.model_dump(), "state": env.state()}


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.task)
    result = env.step(req.action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
        "score": env.score(),
    }


@app.get("/state")
def state(task: int = 1):
    env = _get_env(task)
    return {"state": env.state(), "score": env.score()}


@app.get("/score")
def score(task: int = 1):
    env = _get_env(task)
    return {"score": env.score(), "task": task}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
