import argparse
import json
import os
import random
import time
import logging
import codecs
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    StoppingCriteria,
)
from typing import Optional, List, Dict
import shortuuid
import torch
import psutil
import math
from tqdm import tqdm
import jsonlines
import ray
import gc

from torch.nn import CrossEntropyLoss


class SequenceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, target_sequences, prompt):
        self.tokenizer = tokenizer
        self.target_sequences = target_sequences or []
        self.prompt = prompt

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = self.tokenizer.decode(
            input_ids[0], skip_special_tokens=True, padding=False
        )
        generated_text = generated_text.replace(self.prompt, "")

        # Check if the target sequence appears in the generated text
        for target_sequence in self.target_sequences:
            if target_sequence in generated_text:
                return True

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self


def load_pretrained_model(model_path, from_pretrained_kwargs: dict):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
    except NameError:
        model = AutoModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs,
        )
    return model, tokenizer


def get_gpu_memory(num_gpus: Optional[int] = None):
    """Get available memory for each GPU."""
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if num_gpus is None
        else min(num_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_model(
    model_path: str,
    generation_config: dict,
    device: str = "cuda",
    num_gpus_per_model: int = 1,
    max_gpu_memory: Optional[str] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    **kwargs,
):
    """Load a model from Hugging Face."""

    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus_per_model != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs["device_map"] = (
                    "sequential"  # This is important for not the same VRAM sizes
                )
                available_gpu_memory = get_gpu_memory(num_gpus_per_model)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus_per_model)
                }
            else:
                kwargs["max_memory"] = {
                    i: max_gpu_memory for i in range(num_gpus_per_model)
                }

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit

    # Load model
    model, tokenizer = load_pretrained_model(model_path, kwargs)

    if device == "cuda" and num_gpus_per_model == 1 and not cpu_offloading:
        model.to(device)

    logging.debug(model)

    return model, tokenizer


@torch.inference_mode()
def run_one_problem(
    request,
    model,
    tokenizer,
    generation_config: dict,
    output_path: str,
    device: str = "cuda",
    trial_run: bool = False,
    dump_individual_rsp: bool = True,
    inference_mode: str = "default",
    **kwargs,
) -> dict:
    if isinstance(request, dict):
        request_dict = request
    else:
        request_dict = request.__dict__

    if dump_individual_rsp:
        save_path = os.path.join(
            output_path, "responses", f'{request_dict["uuid"]}.json'
        )

        if os.path.exists(save_path) and not trial_run:
            with codecs.open(save_path) as f:
                rsp_content = json.load(f)
            logging.debug(f"Skipping {save_path}")
            return rsp_content

    if "generation_config" in request_dict and isinstance(
        request_dict["generation_config"], dict
    ):
        generation_config = generation_config.copy()
        generation_config.update(request_dict["generation_config"])
    else:
        generation_config = generation_config.copy()

    if inference_mode == "default":
        input_ids = tokenizer.encode(
            request_dict["prompt"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).to(device)
        stopping_criteria = SequenceStoppingCriteria(
            tokenizer,
            generation_config.pop("stop_sequences", None),
            request_dict["prompt"],
        )

        output_ids = model.generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            **generation_config,
        )
        output_ids = output_ids[0, len(input_ids[0]) :]
        output = tokenizer.decode(output_ids)

        response = {
            "generated_text": output,
            "output_ids": output_ids.tolist(),
        }

    elif inference_mode == "loglikelihood":
        input_tokens = tokenizer.encode(
            request_dict["prompt"], padding=False, add_special_tokens=False
        )
        target_tokens = tokenizer.encode(
            request_dict["choices"][0], padding=False, add_special_tokens=False
        )

        all_tokens = torch.tensor([input_tokens + target_tokens[:-1]], device=device)
        labels = torch.tensor([input_tokens[1:] + target_tokens], device=device)

        output = model(all_tokens, labels=labels)
        log_softmax = torch.nn.functional.log_softmax(output.logits, dim=-1)[0][
            -len(target_tokens) :
        ]
        target_logits = (
            torch.gather(
                log_softmax,
                1,
                torch.Tensor(target_tokens).unsqueeze(-1).type(torch.long).to(device),
            )
            .sum()
            .cpu()
            .item()
        )

        response = {
            "loglikelihood": target_logits,
            "target_length": len(target_tokens),
        }

    elif inference_mode == "lm_loss":
        input_tokens = tokenizer.encode(
            request_dict["prompt"], padding=False, add_special_tokens=False
        )
        target_tokens = tokenizer.encode(
            request_dict["choices"][0], padding=False, add_special_tokens=False
        )

        all_tokens = torch.tensor([input_tokens + target_tokens[:-1]], device=device)
        labels = torch.tensor([input_tokens[1:] + target_tokens], device=device)

        output = model(all_tokens, labels=labels)
        logits = output.logits[0]
        labels = labels.view(-1)
        loss_func = CrossEntropyLoss()

        lm_loss = loss_func(logits, labels).cpu().item()

        response = {
            "lm_loss": lm_loss,
            "length": all_tokens.shape[1],
        }

    elif inference_mode == "all_loglikelihood":
        input_tokens = tokenizer.encode(
            request_dict["prompt"], padding=False, add_special_tokens=False
        )
        target_tokens = tokenizer.encode(
            request_dict["choices"][0], padding=False, add_special_tokens=False
        )

        # all_tokens = torch.tensor([input_tokens + target_tokens[:-1]], device=device)
        # labels = torch.tensor([input_tokens[1:] + target_tokens], device=device)

        input_ids = torch.tensor([input_tokens + target_tokens], device=device)
        output = model(input_ids, labels=input_ids)
        log_softmax = torch.nn.functional.log_softmax(output.logits, dim=-1)[0]
        # all_probs = torch.gather(
        #     log_softmax, 1, torch.Tensor(input_tokens[1:] + target_tokens).unsqueeze(-1).type(torch.long).to(device)
        # ).cpu().tolist()
        all_probs = []
        for i, token_id in enumerate(input_ids[0][1:]):
            all_probs.append(log_softmax[i][token_id].cpu().item())

        response = {
            "loglikelihood": all_probs,
        }

    if dump_individual_rsp:
        with codecs.open(save_path, "w") as f:
            json.dump(
                {"request": request_dict, "response": response},
                f,
                ensure_ascii=False,
                indent=2,
            )

    return {"request": request_dict, "response": response}


@ray.remote
class ProgressActor:
    def __init__(self, total):
        self.progress = 0
        self.total = total

    def update(self, value):
        self.progress += value
        return self.progress

    def get_progress(self):
        return self.progress


@torch.inference_mode()
def model_batch_inference(
    dataset,
    model_path,
    progress_actor=None,
    **kwargs,
) -> List[dict]:

    model, tokenizer = load_model(model_path, **kwargs)

    preds = []

    if progress_actor is not None:
        for problem in dataset:
            preds.append(run_one_problem(problem, model, tokenizer, **kwargs))
            progress_actor.update.remote(1)
    else:
        for problem in tqdm(dataset):
            preds.append(run_one_problem(problem, model, tokenizer, **kwargs))

    del model, tokenizer

    return preds


def run_dataset_inference(
    dataset,
    model_path,
    output_path,
    generation_config: Optional[dict] = None,
    device: Optional[str] = "cuda",
    num_gpus_per_model: Optional[int] = 1,
    num_gpus_total: Optional[int] = 1,
    max_gpu_memory: Optional[str] = None,
    trial_run: Optional[bool] = False,
    dump_individual_rsp: Optional[bool] = True,
    inference_mode: Optional[str] = "default",
) -> List[dict]:
    """Run inference on a dataset."""

    generation_config = generation_config or {}

    # set max_new_tokens to 20 if not specified
    if inference_mode == "default" and "max_new_tokens" not in generation_config:
        logging.getLogger(__name__).warning(
            "max_new_tokens not specified, setting default to 20"
        )
        generation_config["max_new_tokens"] = 20

    num_models = num_gpus_total // num_gpus_per_model

    # Split the question file into `num_gpus` files
    use_ray = num_models > 1

    if use_ray:
        import ray

        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            model_batch_inference
        ).remote
        progress_bar = tqdm(total=len(dataset))
        progress_actor = ProgressActor.remote(len(dataset))
    else:
        get_answers_func = model_batch_inference
        progress_actor = None

    chunk_size = (len(dataset) + num_models - 1) // num_models
    preds = []

    for i in range(0, len(dataset), chunk_size):
        preds.append(
            get_answers_func(
                dataset[i : i + chunk_size],
                model_path=model_path,
                progress_actor=progress_actor,
                generation_config=generation_config,
                num_gpus_per_model=num_gpus_per_model,
                max_gpu_memory=max_gpu_memory,
                output_path=output_path,
                device=device,
                trial_run=trial_run,
                dump_individual_rsp=dump_individual_rsp,
                inference_mode=inference_mode,
            )
        )

    if use_ray:
        while progress_bar.n < progress_bar.total:
            progress_bar.update(
                ray.get(progress_actor.get_progress.remote()) - progress_bar.n
            )
            time.sleep(0.1)
        progress_bar.close()
        preds = ray.get(preds)
        ray.shutdown()

    # merge the results
    return [p for pred in preds for p in pred]


def write_results(responses: List[dict], output_path: str):
    save_path = os.path.join(output_path, "all_responses.jsonl")
    with jsonlines.open(save_path, "w") as f:
        f.write_all(responses)


def run_local_hf_inference(
    dataset,
    model_path,
    output_path,
    generation_config,
    device: str = "cuda",
    num_gpus_per_model: int = 1,
    num_gpus_total: int = 1,
    max_gpu_memory: Optional[str] = None,
    gpu_ids: Optional[List[int]] = None,
    trial_run=False,
    dump_individual_rsp=True,
    inference_mode: str = "default",
):
    logging.getLogger(__name__).info(
        f"Running local hf inference, num_requests: {len(dataset)}"
    )

    if gpu_ids is not None:
        assert (
            num_gpus_per_model == 1
        ), "Cannot specify gpu_ids with num_gpus_per_model > 1"
        assert num_gpus_total == len(gpu_ids), "num_gpus_total must match len(gpu_ids)"
        logging.getLogger(__name__).info(f"Using GPUs: {gpu_ids}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in gpu_ids])

    os.makedirs(output_path, exist_ok=True)

    if dump_individual_rsp:
        os.makedirs(os.path.join(output_path, "responses"), exist_ok=True)

    if os.path.exists(os.path.join(output_path, "all_responses.jsonl")):
        os.remove(os.path.join(output_path, "all_responses.jsonl"))

    try:
        preds = run_dataset_inference(
            dataset,
            model_path=model_path,
            output_path=output_path,
            generation_config=generation_config,
            device=device,
            num_gpus_per_model=num_gpus_per_model,
            num_gpus_total=num_gpus_total,
            max_gpu_memory=max_gpu_memory,
            trial_run=trial_run,
            dump_individual_rsp=dump_individual_rsp,
            inference_mode=inference_mode,
        )
        write_results(preds, output_path)

    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupt received! Closing...")

    gc.collect()
    torch.cuda.empty_cache()
