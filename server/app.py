"""
FastAPI server exposing the Email Triage environment via HTTP.
Endpoints mirror the OpenEnv spec.
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Union
import uvicorn
import os
import sys

# Ensure the root directory is in sys.path so environment.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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


def _parse_task(task: Union[int, str]) -> int:
    if isinstance(task, str):
        if task.startswith("task"):
            return int(task[4:])
        return int(task)
    return task

class ResetRequest(BaseModel):
    task: Union[int, str] = 1


class StepRequest(BaseModel):
    task: Union[int, str] = 1
    action: Action


def _get_env(task: Union[int, str]) -> EmailTriageEnv:
    task_int = _parse_task(task)
    if task_int not in _envs:
        raise HTTPException(status_code=400, detail=f"Task {task_int} not initialised. Call /reset first.")
    return _envs[task_int]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    task = _parse_task(req.task if req else 1)
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
def state(task: Union[int, str] = 1):
    env = _get_env(task)
    return {"state": env.state(), "score": env.score()}


@app.get("/score")
def score(task: Union[int, str] = 1):
    env = _get_env(task)
    return {"score": env.score(), "task": _parse_task(task)}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
