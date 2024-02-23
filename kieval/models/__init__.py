from kieval.models.local_hf_model import run_local_hf_inference
from kieval.models.remote_hf_model import run_remote_hf_inference
from kieval.models.openai_model import run_openai_inference
import logging

TYPE_TO_MODEL = {
    "local_hf": run_local_hf_inference,
    "remote_hf": run_remote_hf_inference,
    "openai": run_openai_inference,
}


def load_inference_function(type):
    assert type in TYPE_TO_MODEL
    return TYPE_TO_MODEL[type]


def load_eval_model(type, **kwargs):
    assert type in TYPE_TO_MODEL
    model_class = TYPE_TO_MODEL[type]

    logging.info(f"Loading model {type}, kwargs={kwargs}")

    return model_class(**kwargs)
