# KIEval: A Knowledge-grounded Interactive Evaluation Framework for Large Language Models

<div align="center">
  <a>Zhuohao Yu<sup>1</sup></a>&emsp;
  <a>Chang Gao<sup>1</sup></a>&emsp;
  <a>Wenjin Yao<sup>1</sup></a>&emsp;
  <a>Yidong Wang<sup>1</sup></a>&emsp; <br>
  <a>Wei Ye<sup>†1</sup></a>&emsp;
  <a>Jindong Wang<sup>2</sup></a>&emsp;
  <a>Xing Xie<sup>2</sup></a>&emsp;
  <a>Yue Zhang<sup>3</sup></a>&emsp;
  <a>Shikun Zhang<sup>1</sup></a>&emsp;
  <p> <sup>1</sup> Peking University, <sup>2</sup> Microsoft Research, <sup>3</sup> Westlake University.</p>
</div>



## Overview
<div align="center">

<a href="https://github.com/zhuohaoyu/KIEval/">
    <img src="figures/pipeline.png" alt="KIEval Pipeline" width="600" class="center">
</a>
</div>

This is the official repository for [KIEval: A Knowledge-grounded Interactive Evaluation Framework for Large Language Models](https://arxiv.org/abs/2402.15043), accepted to the main conference of 62nd Annual Meeting of the Association for Computational Linguistics (**ACL 2024**).

Automatic evaluation methods for large language models (LLMs) are hindered by data contamination, leading to inflated assessments of their effectiveness. Existing strategies, which aim to detect contaminated texts, focus on quantifying contamination status instead of accurately gauging model performance. In this paper, we introduce KIEval, a Knowledge-grounded Interactive Evaluation framework, which incorporates an LLM-powered "interactor" role for the first time to accomplish a dynamic contamination-resilient evaluation. Starting with a question in a conventional LLM benchmark involving domain-specific knowledge, KIEval utilizes dynamically generated, multi-round, and knowledge-focused dialogues to determine whether a model's response is merely a recall of benchmark answers or demonstrates a deep comprehension to apply knowledge in more complex conversations. Extensive experiments on seven leading LLMs across five datasets validate KIEval's effectiveness and generalization. We also reveal that data contamination brings no contribution or even negative effect to models' real-world applicability and understanding, and existing contamination detection methods for LLMs can only identify contamination in pre-training but not during supervised fine-tuning.


## Quick Start

To get started, first clone the repository and setup the environment:

```bash
git clone https://github.com/zhuohaoyu/KIEval.git
cd KIEval
pip install -r requirements.txt
```

We provide a modular implementation of our method, currently we support evaluating models locally with Huggingface's Transformers, and remote models with text-generation-inference or other APIs.

To reproduce results in our paper or evaluate new models with KIEval, we recommend starting a [text-generation-inference](https://huggingface.co/docs/text-generation-inference/en/index) instance with your model:

```bash
model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model
```

Then, generate an evaluation config file with our script:

```bash
python scripts/generate-basic.py \
    --template ./config/template-basic.json \ # a template config file we provide
    --dataset arc_challenge \ # dataset name, please refer to datasets/ for all supported datasets
    --base_url http://your-host-url:8080 \ # replace with your host url, if you start the text-generation-inference locally, use http://localhost:8080
    --model_name llama-2-7b-chat-hf \ # any name you like
    --model_path meta-llama/Llama-2-7b-chat-hf \ # Huggingface model ID or local model path
    --openai_api_base https://api.openai.com/v1/ \ # OpenAI API base url, you could replace with proxy URL if needed
    --openai_key your_openai_key \ # replace with your OpenAI API key
    --openai_model gpt-4-1106-preview \ 
    --output_path ./result \ # output path for evaluation results
    --generate_path ./config/generated.json # output path for generated config file
```

Finally, run the evaluation process with the generated config file and wait for the results :)

```bash
python run.py -c ./config/generated.json
```


This repository provides all settings necessary for researchers to reproduce the results of KIEval, it also facilitates the reproduction of all metrics (from previous works) discussed in our paper. Please refer to `config/templates` for all supported evaluation methods.


## Citation
✨ If you find our work helpful, please consider citing with:


```bibtex
@misc{yu2024kieval,
      title={KIEval: A Knowledge-grounded Interactive Evaluation Framework for Large Language Models}, 
      author={Zhuohao Yu and Chang Gao and Wenjin Yao and Yidong Wang and Wei Ye and Jindong Wang and Xing Xie and Yue Zhang and Shikun Zhang},
      year={2024},
      eprint={2402.15043},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```