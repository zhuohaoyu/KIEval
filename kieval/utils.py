from typing import Dict, Optional
from hashlib import md5
from base64 import urlsafe_b64encode

import json
import json5


def parse_json(content: str):
    if "{" not in content or "}" not in content:
        return {}
    content = content[content.find("{") : content.rfind("}") + 1]
    try:
        data = json5.loads(content)
        return data
    except Exception as e:
        return {}


def calculate_inference_endpoint_hash(
    inference_config: Dict, context_hash: Optional[str] = None
):
    """Validate the inference config and calculate the hash."""
    hashstr = ""

    assert (
        "type" in inference_config
    ), "inference_config must contain a 'type' key, endpoint_type must be one of ['openai', 'local_hf', 'remote_hf']"
    endpoint_type = inference_config["type"]
    hashstr += f"$type: {endpoint_type}"

    assert endpoint_type in [
        "openai",
        "local_hf",
        "remote_hf",
    ], f"endpoint_type must be one of ['openai', 'local_hf', 'remote_hf'], got {endpoint_type}"

    assert (
        "inference_kwargs" in inference_config
    ), "inference_config must contain a 'inference_kwargs' key"
    inference_kwargs = inference_config["inference_kwargs"]

    if endpoint_type == "openai":
        assert (
            "openai_model" in inference_kwargs
        ), "inference_kwargs must contain a 'openai_model' key, e.g., 'gpt-3.5-turbo'"
        openai_model = inference_kwargs["openai_model"]
        generation_config = inference_kwargs.get("generation_config", {})
        hashstr += f"$openai_model: {openai_model}"
        hashstr += f"$generation_config: {json.dumps(generation_config)}"

    elif endpoint_type == "remote_hf":
        assert (
            "base_url" in inference_kwargs
        ), "inference_kwargs must contain a 'base_url' key"
        base_url = inference_kwargs["base_url"]
        model_name = inference_kwargs.get("model_name", None)
        generation_config = inference_kwargs.get("generation_config", {})
        prompt_postprocessor_config = inference_kwargs.get(
            "prompt_postprocessor_config", {}
        )
        if model_name is not None:
            hashstr += f"$model_name: {model_name}"
        else:
            hashstr += f"$base_url: {base_url}"
        hashstr += f"$generation_config: {json.dumps(generation_config)}"
        hashstr += (
            f"$prompt_postprocessor_config: {json.dumps(prompt_postprocessor_config)}"
        )

    elif endpoint_type == "local_hf":
        hashstr += f"$kwargs: {json.dumps(inference_kwargs)}"

    if context_hash is not None:
        hashstr += f"$context_hash: {context_hash}"

    hashstr = hashstr.encode("utf-8")
    hash_digest = md5(hashstr).digest()

    url_safe_hash = urlsafe_b64encode(hash_digest).rstrip(b"=").decode("utf-8")
    return url_safe_hash
