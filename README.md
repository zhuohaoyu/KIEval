# KIEval: A Knowledge-grounded Interactive Evaluation Framework for Large Language Models

This is the official repository for KIEval, all our code and datasets can be found here.

## Quick Start

To get started, first clone the repository and setup the enviroment:

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
```



### Interactive Evaluation

To reproduce KIEval results, first start a text-generation-inference(huggingface.co/docs/text-generation-inference/en/index) instance with your candidate model:

```bash
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model
```

 Then, generate an evaluation config with our script:

```bash
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
```

Finally, run the evaluation process:

```bash
python run.py -c ./config/generated.json
```


This repository provides all settings necessary for researchers to reproduce the results of KIEval, it also facilitates the reproduction of all metrics (from previous works) discussed in our paper. Please refer to `config/templates` for all supported evaluation methods.
