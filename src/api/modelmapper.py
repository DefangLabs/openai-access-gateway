import os
import json
from pathlib import Path

_model_map = None

def load_model_map():
    global _model_map
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    modelmap_path = os.path.join(BASE_DIR, "../data/modelmap.json")
    with open(modelmap_path, "r") as f:
        _model_map = json.load(f)

def get_model(provider, model, fallback_model):
    provider = provider.lower()
    if model is None or model == "":
        model = fallback_model
    model = model.lower().removesuffix(":latest")

    available_models = _model_map.get(provider, {})
    return available_models.get(model, model)

