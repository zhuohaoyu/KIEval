import argparse
import json
import os

# Define the argument parser
parser = argparse.ArgumentParser(
    description="Generate config JSON for evaluation toolkit"
)
parser.add_argument(
    "--template", default="./config/template-basic.json", help="Config template"
)
parser.add_argument("--dataset", required=True, help="Dataset type")
parser.add_argument("--base_url", required=True, help="Candidate base URL")
parser.add_argument("--model_name", required=True, help="Candidate model name")
parser.add_argument("--model_path", required=True, help="Candidate model path")
parser.add_argument("--openai_api_base", required=True, help="OpenAI API base URL")
parser.add_argument("--openai_key", required=True, help="OpenAI API key")
parser.add_argument("--openai_model", required=True, help="OpenAI model")
parser.add_argument("--output_path", required=True, help="Output path")
parser.add_argument("--generate_path", required=True, help="Generated config path")

# Parse arguments
args = parser.parse_args()

dataset_configs = {
    "arc_challenge": {
        "type": "arc_challenge",
        "dataset_kwargs": {
            "seed": 2,
            "split": "test",
            "name_or_path": "allenai/ai2_arc",
            "config_name": "ARC-Challenge",
            "fewshot_split": "train",
            "fewshot_num": 5,
        }
    },
    "ceval": {
        "type": "ceval",
        "dataset_kwargs": {
            "seed": 2,
            "fewshot_split": "dev",
            "fewshot_num": 5,
            "name_or_path": "liyucheng/ceval_all",
        }
    },
    "hellaswag": {
        "type": "hellaswag",
        "dataset_kwargs": {
            "seed": 2,
            "split": "validation",
            "name_or_path": "liyucheng/ceval_all",
            "fewshot_split": "train",
            "fewshot_num": 5,
        },
    },
    "mmlu": {
        "type": "mmlu",
        "dataset_kwargs": {
            "seed": 2,
            "split": "test",
            "name_or_path": "cais/mmlu",
            "config_name": "all",
            "fewshot_split": "dev",
            "fewshot_num": 5,
        },
    },
    "arc_easy": {
        "type": "arc_challenge",
        "dataset_kwargs": {
            "seed": 2,
            "split": "test",
            "name_or_path": "allenai/ai2_arc",
            "config_name": "ARC-Easy",
            "fewshot_split": "train",
            "fewshot_num": 5,
        }
    }
}

with open(args.template) as f:
    config_template = json.load(f)

config_template["results_output_path"] = os.path.join(
    args.output_path, f"results-{args.model_name}-{args.dataset}.json"
)
# Update the template with command line arguments
for step in config_template["steps"]:
    step["step_name"] = f"{step['step_type']}-{args.model_name}-{args.dataset}"
    if "dataset_config" in step:
        step["dataset_config"] = dataset_configs[args.dataset]
    if "inference_config" in step:
        step["inference_config"]["inference_kwargs"]["model_name"] = args.model_name
        step["inference_config"]["inference_kwargs"]["base_url"] = [args.base_url]
    if "roles_config" in step:
        step["roles_config"]["candidate"]["inference_kwargs"][
            "model_name"
        ] = args.model_name
        step["roles_config"]["candidate"]["inference_kwargs"]["base_url"] = [
            args.base_url
        ]
        step["roles_config"]["candidate"]["prompt_postprocessor_config"][
            "tokenizer_name_or_path"
        ] = args.model_path
        step["roles_config"]["interactor"]["inference_kwargs"][
            "openai_api_base"
        ] = args.openai_api_base
        step["roles_config"]["interactor"]["inference_kwargs"][
            "openai_key"
        ] = args.openai_key
        step["roles_config"]["interactor"]["inference_kwargs"][
            "openai_model"
        ] = args.openai_model
        step["roles_config"]["evaluator"]["inference_kwargs"][
            "openai_api_base"
        ] = args.openai_api_base
        step["roles_config"]["evaluator"]["inference_kwargs"][
            "openai_key"
        ] = args.openai_key
        step["roles_config"]["evaluator"]["inference_kwargs"][
            "openai_model"
        ] = args.openai_model
    if step["step_type"] == "simple_multiple_choice":
        step["inference_config"]["output_path"] = args.output_path
    elif step["step_type"] == "interactive_evaluation":
        step["output_path"] = args.output_path

# Output the updated configuration to a JSON file
with open(args.generate_path, "w") as outfile:
    json.dump(config_template, outfile, indent=4)

print(f"Config file generated: {args.generate_path}")

"""
python scripts/generate-basic.py \
    --template ./config/template-basic.json \
    --dataset arc_challenge \
    --base_url http://your-tgi-url:port \
    --model_name llama-2-7b-chat-hf \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --openai_api_base https://api.openai.com/v1/ \
    --openai_key your_openai_key \
    --openai_model gpt-4-1106-preview \
    --output_path ./result \
    --generate_path ./config/generated.json
"""
