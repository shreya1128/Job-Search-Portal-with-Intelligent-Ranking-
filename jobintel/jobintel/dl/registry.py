"""Stores training results in model_experiments."""
from __future__ import annotations

from sqlalchemy import delete, insert

from jobintel.db.models import ModelExperiment
from jobintel.db.session import get_engine


def store_experiment(kind: str, res: dict, info: dict, task: str = "job function classification (EMSCAD)", engine=None, dataset: str | None = None) -> None:
    e = engine or get_engine()
    row = dict(model_type=kind, task=task, dataset=dataset or f"emscad, {info.get('jobs')} jobs, {len(info.get('classes', []))} classes", hyperparameters=res.get("config"), accuracy=res.get("accuracy"),
               precision_macro=res.get("precision_macro"), recall_macro=res.get("recall_macro"), f1_macro=res.get("f1_macro"), val_loss=res.get("val_loss"),
               reconstruction_loss=res.get("reconstruction_loss"), train_seconds=res.get("train_seconds"), n_parameters=res.get("n_parameters"), artifact_path=res.get("artifact_path"),
               notes=res.get("note"))
    with e.begin() as c:
        c.execute(delete(ModelExperiment).where(ModelExperiment.model_type == kind, ModelExperiment.task == task))
        c.execute(insert(ModelExperiment), [row])
