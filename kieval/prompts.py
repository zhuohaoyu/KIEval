from transformers import AutoTokenizer
from typing import List, Dict, Optional
import logging, json
from hashlib import md5
from base64 import urlsafe_b64encode

MULTIPLE_CHOICE_CHOICES_TMPL = {
    "default": """{choice_key}. {choice_text}\n""",
    "numbers": """{choice_key}). {choice_text}\n""",
    "brace": """({choice_key}) {choice_text}\n""",
    "bracket": """[{choice_key}] {choice_text}\n""",
    "qa": """{choice_text}\n""",
    "reasoning": """{choice_key}. {choice_text}\n""",
}

MULTIPLE_CHOICE_PROMPT_TMPL = {
    "default": """### Question: {problem}\n\n### Choices: {choices}\n\n### Answer:""",
    "test_1": """### Question: {problem}\n\n### Choices:\n{choices}\n### Answer:\n\n""",
    "qa": """Question: {problem}\nAnswer:""",
    "reasoning": """### {problem}\n\n### Choices: {choices}\n\n### Answer:\n\n""",
}


SYSTEM_PROMPTS = {
    "default": "You are a helpful, respectful and honest assistant. {prompt}",
    "llama2": """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant that answers multiple choice problem accurately. You should be very strict on format and answer only the choice without any explanation. Only output in format `### Answer: {{answer_key}}`, this is very important.\n<</SYS>> \n{prompt}""",
}


class TriloguePrompter:
    JSON_EXAMPLES = {
        "default": {
            "accuracy": {"comment": "", "score": 0},
            "reasoning": {"comment": "", "score": 0},
            "relevance": {"comment": "", "score": 0},
            "coherence": {"comment": "", "score": 0},
            "conciseness": {"comment": "", "score": 0},
            "overall_comment": "",
            "overall_score": 0,
            "stop_conversation": False,
        },
        "clear": {
            "accuracy": {"comment": "", "score": 0},
            "logic": {"comment": "", "score": 0},
            "relevance": {"comment": "", "score": 0},
            "coherence": {"comment": "", "score": 0},
            "conciseness": {"comment": "", "score": 0},
            "overall_comment": "",
            "overall_score": 0,
            "stop_conversation": False,
            "stop_reason": "none",
        },
    }

    JSON_FEWSHOT_EXAMPLES = {
        "correct": {
            "default": {
                "conciseness": {
                    "comment": "The candidate gives a concise answer without unnecessary words.",
                    "score": 4,
                },
                "relevance": {
                    "comment": "The candidate followed the instruction and is relavant.",
                    "score": 4,
                },
                "reasoning": {
                    "comment": "No reasoning required since the candidate only gives one answer.",
                    "score": 4,
                },
                "coherence": {
                    "comment": "Not applicable since the candidate only gives one answer.",
                    "score": 4,
                },
                "accuracy": {
                    "comment": "The candidate's answer is correct.",
                    "score": 4,
                },
                "comment": "The candidate gave a correct answer, I want to keep listening.",
                "overall_score": 4,
                "stop_conversation": False,
            },
            "qa": {
                "conciseness": {
                    "comment": "The candidate gives a concise answer without unnecessary words.",
                    "score": 4,
                },
                "relevance": {
                    "comment": "The candidate followed the instruction and is relavant.",
                    "score": 4,
                },
                "reasoning": {
                    "comment": "No reasoning required since the candidate only gives one answer.",
                    "score": 4,
                },
                "coherence": {
                    "comment": "Not applicable since the candidate only gives one answer.",
                    "score": 4,
                },
                "accuracy": {
                    "comment": "The candidate's answer is correct.",
                    "score": 4,
                },
                "comment": "The candidate gave a correct answer, I want to keep listening.",
                "overall_score": 4,
                "stop_conversation": False,
            },
            "clear": {
                "accuracy": {
                    "comment": "The candidate's answer is correct.",
                    "score": 4,
                },
                "logic": {
                    "comment": "No reasoning required since the candidate only gives one answer.",
                    "score": 4,
                },
                "relevance": {
                    "comment": "The candidate followed the instruction and is relavant.",
                    "score": 4,
                },
                "coherence": {
                    "comment": "Not applicable since the candidate only gives one answer.",
                    "score": 4,
                },
                "conciseness": {
                    "comment": "The candidate gives a concise answer without unnecessary words.",
                    "score": 4,
                },
                "overall_comment": "The candidate gave a correct answer, I want to keep listening.",
                "overall_score": 4,
                "stop_conversation": False,
                "stop_reason": "none",
            },
        },
        "incorrect": {
            "default": {
                "conciseness": {
                    "comment": "The candidate gives a concise answer without unnecessary words.",
                    "score": 4,
                },
                "relevance": {
                    "comment": "The candidate followed the instruction and is relavant.",
                    "score": 4,
                },
                "reasoning": {
                    "comment": "No reasoning required since the candidate only gives one answer.",
                    "score": 4,
                },
                "coherence": {
                    "comment": "Not applicable since the candidate only gives one answer.",
                    "score": 4,
                },
                "accuracy": {
                    "comment": "The candidate's answer is incorrect according to the answer given by the interactor.",
                    "score": 1,
                },
                "comment": "Although the candidate gave a wrong answer, I could not judge whether the candidate is bad or not. I want to keep listening.",
                "overall_score": 2,
                "stop_conversation": False,
            },
            "clear": {
                "accuracy": {
                    "comment": "The candidate's answer is incorrect according to the answer given by the interactor.",
                    "score": 1,
                },
                "logic": {
                    "comment": "No reasoning required since the candidate only gives one answer.",
                    "score": 4,
                },
                "relevance": {
                    "comment": "The candidate followed the instruction and is relavant.",
                    "score": 4,
                },
                "coherence": {
                    "comment": "Not applicable since the candidate only gives one answer.",
                    "score": 4,
                },
                "conciseness": {
                    "comment": "The candidate gives a concise answer without unnecessary words.",
                    "score": 4,
                },
                "overall_comment": "Although the candidate gave a wrong answer, I could not judge whether the candidate is bad or not. I want to keep listening.",
                "overall_score": 2,
                "stop_conversation": False,
                "stop_reason": "none",
            },
        },
    }

    SYSTEM_PROMPTS = {
        "candidate": {
            "default": """You are a helpful assistant that follows the instructions of the user. You must output in less than 200 words and your responses should be concise and accurate. """,
        },
        "interactor": {
            "default": """As an expert interviewer, initiate a reflective dialogue by asking the user to describe the rationale behind their answer to the multiple-choice question. You must output in less than 200 words. Encourage them to be succinct and precise. Guide the conversation to uncover and address any misconceptions, prompting the user to critically evaluate their thinking process. Their answer is likely to be incorrect. Avoid prompting the user to merely explain their choice, this is very important. Instead, focus on probing questions that challenge their assumptions and lead them to a deeper understanding of the topic. The outputs from the user is very likely to be wrong, do not validate misinformation or incorrect logic; instead, use it as a point of assessment. Remember, you are not educating the user, be neutral and do not give comments since you are the interviewer instead of the evaluator, this is very important. Do not ask too many questions before the user responds. Always skip the greetings, or any other prompting words, ask questions directly. DO NOT say anything else.""",
            "qa": """As an expert interviewer, initiate a reflective dialogue by asking the user to describe the rationale behind their answer to the question. You must output in less than 200 words. Encourage them to be succinct and precise. Guide the conversation to uncover and address any misconceptions, prompting the user to critically evaluate their thinking process. Their answer is likely to be incorrect. Avoid prompting the user to merely explain their answer, this is very important. Instead, focus on probing questions that challenge their assumptions and lead them to a deeper understanding of the topic. The outputs from the user is very likely to be wrong, do not validate misinformation or incorrect logic; instead, use it as a point of assessment. Remember, you are not educating the user, be neutral and do not give comments since you are the interviewer instead of the evaluator, this is very important. Do not ask too many questions before the user responds. Always skip the greetings, or any other prompting words, ask questions directly. DO NOT say anything else.""",
        },
        "evaluator": {
            "default": """You are an objective evaluator at an interview, and you are evaluating the candidate's performance. The <Candidate> is asked to solve a multiple choice problem and then talk about it with the <Interactor>. Be noted that the answer from the candidate is likely to be incorrect. Now, you need to evaluate the candidate's performance. After each response from the <Candidate> in a conversation with an <Interactor>, evaluate the candidate's performance using the following aspects int JSON format, each scored from 1 (Poor) to 4 (Good):\nAccuracy: Factual correctness of the candidate's statements.\nReasoning: Logical correctness of the candidate's statements.\nRelevance: Pertinence of the responses to the discussion topics.\nCoherence: Consistency and logical connection in responses.\nConciseness: Brevity and efficiency of communication.\nScoring Guide:\n1 (Poor): Major deficiencies or errors in the aspect.\n2 (Below Average): Noticeable weaknesses, not fully satisfactory.\n3 (Above Average): Competent, slightly exceeds basic expectations.\n4 (Good): Strong capabilities, excels in the aspect.\nOnly give integers between 1 and 4 for each aspect, no other words are allowed. Be consistent and objective in your evaluation.\nFor each response, provide a JSON object with `comment` and `score` for each aspect, an `overall_comment`, and an `overall_score`. Give comments before giving scores so that you can think more carefully for scoring. Include a `stop_conversation` flag to indicate if the conversation should be terminated early due to extremely low performance, off-topic responses, factual errors, or rule violations. \nJSON Template:\n""",
            "add_detailed_criteria": """You are an objective evaluator in an interview simulation. Your task is to assess a candidate's performance during a conversation with an interactor after every round of interaction. The candidate will solve a problem and discuss it with the interactor. Be aware that the candidate's answers might be incorrect. Ensure the candidate strictly adheres to their assigned role as assistant, we use [BEGIN INTERACTOR] and [BEGIN CANDIDATE] to mark the start of each party's contribution. Evaluate the candidate's performance after each response using these aspects, each scored from 1 (Poor) to 4 (Good): \nAccuracy: This aspect evaluates the factual correctness of the candidate's statements. It examines whether the information provided is true and aligns with known facts and data. \nReasoning: This aspect assesses the logical correctness and soundness of the candidate's statements and arguments. It looks at how well the candidate connects ideas logically and whether their conclusions follow from their premises. \nRelevance: This measures the pertinence and appropriateness of the responses to the discussion topics. It considers whether the candidate's contributions are on-topic and effectively address the subject matter of the conversation. \nCoherence: This aspect evaluates the consistency and logical flow of the candidate's responses in context, as well as their adherence to their assigned role and instructions. It includes assessing whether the candidate maintains a clear and consistent line of thought and avoids hallucinations or role confusion. \nConciseness: This aspect focuses on the brevity and efficiency of the candidate's communication. It looks at whether the candidate can convey their thoughts clearly and succinctly without unnecessary elaboration, deviation from the topic or extra meaningless responses. \n\nScoring Guide:\n1 (Poor): Major deficiencies or errors in the aspect.\n2 (Below Average): Noticeable weaknesses, not fully satisfactory.\n3 (Above Average): Competent, slightly exceeds basic expectations.\n4 (Good): Strong capabilities, excels in the aspect.\nOnly give integers between 1 and 4 for each aspect, no other words are allowed. Be consistent and objective in your evaluation.\nFor each response, provide a JSON object with `comment` and `score` for each aspect, an `overall_comment`, and an `overall_score`. Give comments before giving scores so that you can think more carefully for scoring. Include a `stop_conversation` flag to indicate if the conversation should be terminated early due to extremely low performance, off-topic responses, factual errors, rule violations, or failing to adhere to their assigned role. Consider adherence to role as a crucial part of the evaluation in "Coherence" and "Overall", any deviation or role-playing as a 'user' should be flagged and negatively impact the evaluation, this is very important. \nJSON Template:\n""",
            "clear": """You are an objective evaluator in an interview. Your task is to evaluate a candidate's performance during a series of interactions with an interactor, both parties are AI models. The candidate is asked to solve a problem, and the interactor will ask questions accordingly. The conversation alternates between the interactor (marked with [BEGIN INTERACTOR]) and the candidate (marked with [BEGIN CANDIDATE]). After each candidate response, evaluate the candidate's performance in that round of interaction as well as in context, based on the following aspects independently, rating each on a scale from 1 (Poor) to 4 (Good):\nAccuracy: Truthfulness and factual correctness of the candidate's response.\nLogic: Logical structure and soundness of reasoning. Whether conclusions are well-supported and arguments are free from logical fallacies.\nRelevance: How the response relates to the topic. Ensure responses are within the scope of the "assistant" role, avoiding unpermitted role shifts.\nCoherence: How well the response integrates into the context. Consistency with previous statements and overall conversational flow.\nConciseness: Brevity and clarity of the response. Clear, to-the-point communication, free from extraneous elaboration or repetitive words.\n\nScoring Guide:\n1 (Poor): Significant deficiencies or inaccuracies in the aspect.\n2 (Below Average): Noticeable weaknesses, partially on target but lacking in several areas.\n3 (Above Average): Solid and competent, mostly on target with only a few minor shortcomings.\n4 (Good): Strong performance, fully meets and often surpasses expectations.\n\nEvaluation Rules:\n1. Evaluate each response consistently and objectively without bias, strictly adhering to scoring guide.\n2. Score from 1 to 4 for each aspect independently, using only integers. Low score in one aspect should not influence another aspect. Write a brief comment before scoring in the JSON output structure. \n3. Write a overall comment and then give an overall score (same scoring guide). The overall comment should be brief and clear. Consider the performance throughout the interaction, not just in the latest round.\n4. Use of `stop_conversation` Flag: Set this flag to true for early conversation termination in cases listed below. If `stop_conversation` is true, choose a `stop_reason` from ["off_topic", "empty_response", "rule_violation", "repetition", "unpermitted_role_shift"], otherwise `stop_reason` is `none`.\n5. Format of Evaluation: Output in JSON format strictly following the template:""",
            "qa": """You are an objective evaluator at an interview, and you are evaluating the candidate's performance. The <Candidate> is asked to solve a problem and then talk about it with the <Interactor>. Be noted that the answer from the candidate is likely to be incorrect. Now, you need to evaluate the candidate's performance. After each response from the <Candidate> in a conversation with an <Interactor>, evaluate the candidate's performance using the following aspects int JSON format, each scored from 1 (Poor) to 4 (Good):\nAccuracy: Factual correctness of the candidate's statements.\nReasoning: Logical correctness of the candidate's statements.\nRelevance: Pertinence of the responses to the discussion topics.\nCoherence: Consistency and logical connection in responses.\nConciseness: Brevity and efficiency of communication.\nScoring Guide:\n1 (Poor): Major deficiencies or errors in the aspect.\n2 (Below Average): Noticeable weaknesses, not fully satisfactory.\n3 (Above Average): Competent, slightly exceeds basic expectations.\n4 (Good): Strong capabilities, excels in the aspect.\nOnly give integers between 1 and 4 for each aspect, no other words are allowed. Be consistent and objective in your evaluation.\nFor each response, provide a JSON object with `comment` and `score` for each aspect, an `overall_comment`, and an `overall_score`. Give comments before giving scores so that you can think more carefully for scoring. Include a `stop_conversation` flag to indicate if the conversation should be terminated early due to extremely low performance, off-topic responses, factual errors, or rule violations. \nJSON Template:\n""",
            "nightly": """You are an objective evaluator in an interview. Your task is to evaluate a candidate's performance during a series of interactions with an interactor, both parties are AI models. The candidate is asked to solve a problem, and the interactor will ask questions accordingly. The conversation alternates between the interactor (marked with [BEGIN INTERACTOR]) and the candidate (marked with [BEGIN CANDIDATE]). After each candidate response, evaluate the candidate's performance in that round of interaction as well as in context, based on the following aspects independently, rating each on a scale from 1 (Poor) to 4 (Good):

1. Accuracy: Truthfulness and factual correctness of the candidate's response.
2. Logic: Logical structure and soundness of reasoning. Whether conclusions are well-supported and arguments are free from logical fallacies.
3. Relevance: How the response relates to the topic. Ensure responses are within the scope of the "assistant" role, avoiding unpermitted role shifts.
4. Coherence: How well the response integrates into the context. Consistency with previous statements and overall conversational flow.
5. Conciseness: Brevity and clarity of the response. Clear, to-the-point communication, free from extraneous elaboration or repetitive words.

Scoring Guide:

- 1 (Poor): Significant deficiencies or inaccuracies in the aspect.
- 2 (Below Average): Noticeable weaknesses, partially on target but lacking in several areas.
- 3 (Above Average): Solid and competent, mostly on target with only a few minor shortcomings.
- 4 (Good): Strong performance, fully meets and often surpasses expectations.

Evaluation Rules:

1. Evaluate each response consistently and objectively without bias, strictly adhering to scoring guide.
2. Score from 1 to 4 for each aspect independently, using only integers. Low score in one aspect should not influence another aspect. Write a brief comment before scoring in the JSON output structure.
3. Write a overall comment and then give an overall score (same scoring guide). The overall comment should be brief and clear. Consider the performance throughout the interaction, not just in the latest round.
4. Use clear, concise and detailed words in comments, do not be too general. 
5. Use of `stop_conversation` Flag: Set this flag to true for early conversation termination in cases listed below. If `stop_conversation` is true, choose a `stop_reason` from ["off_topic", "empty_response", "rule_violation", "meaningless_repetition", "unpermitted_role_shift"], otherwise `stop_reason` is `none`.
6. Format of Evaluation: Provide your evaluation in a JSON format strictly following the template""",
        },
    }

    MCP_PROMPTS = {
        "default": """### Question: {problem}\n\n### Choices: {choices}\n\n""",
        "qa": """### Question: {problem}\n\n""",
    }

    EVALUATOR_INIT_USER_PROMPTS = {
        "default": "<Interactor>: Let's start by solving this multiple-choice problem, give me only the answer.\n{mcp_prompt}\n<Candidate>: {model_prediction}",
        "qa": "[BEGIN INTERACTOR]\nLet's start by solving this problem, give me only the answer.\n{mcp_prompt}\n[BEGIN CANDIDATE]\n{model_prediction}",
        "add_detailed_criteria": "[BEGIN INTERACTOR]\nLet's start by solving this multiple-choice problem, give me only the answer.\n{mcp_prompt}\n[BEGIN CANDIDATE]\n{model_prediction}",
    }

    EVALUATOR_USER_PROMPTS = {
        "default": "<Interactor>: {interactor_content}\n<Candidate>: {candidate_content}",
        "add_detailed_criteria": "[BEGIN INTERACTOR]\n{interactor_content}\n\n[BEGIN CANDIDATE]\n{candidate_content}",
    }

    INTERACTOR_PROMPTS = {
        "default": "{mcp_prompt}\n### Correct Answer: {correct_answer}\n### My Answer: {model_prediction}\nNow, as the interviewer, you should start asking me questions directly. Do not say anything else or give any comments, just ask questions.",
    }

    def __init__(self, prompter_config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        config = {
            "mcp_prompt": "default",
            "interactor": {"system_prompt": "default", "init_user_prompt": "default"},
            "candidate": {
                "system_prompt": "default",
            },
            "evaluator": {
                "system_prompt": "default",
                "user_prompt": "default",
                "init_user_prompt": "default",
                "fewshot_prompt": "default",
                "system_prompt_json_example": "default",
            },
        }

        if prompter_config is not None and isinstance(prompter_config, dict):
            for key in prompter_config:
                if key in config:
                    if isinstance(config[key], dict):
                        config[key].update(prompter_config[key])
                    else:
                        config[key] = prompter_config[key]
                else:
                    self.logger.warning(f"Invalid key in prompter_config: {key}")

        self.config = config
        self.mcp_prompt = self.MCP_PROMPTS[config["mcp_prompt"]]
        self.interactor_system_prompt = self.SYSTEM_PROMPTS["interactor"][
            config["interactor"]["system_prompt"]
        ]
        self.interactor_init_user_prompt = self.INTERACTOR_PROMPTS[
            config["interactor"]["init_user_prompt"]
        ]
        self.candidate_system_prompt = self.SYSTEM_PROMPTS["candidate"][
            config["candidate"]["system_prompt"]
        ]

        self.evaluator_init_user_prompt = self.EVALUATOR_INIT_USER_PROMPTS[
            config["evaluator"]["init_user_prompt"]
        ]
        self.evaluator_user_prompt = self.EVALUATOR_USER_PROMPTS[
            config["evaluator"]["user_prompt"]
        ]
        self.evaluator_system_prompt = self.SYSTEM_PROMPTS["evaluator"][
            config["evaluator"]["system_prompt"]
        ]
        self.evaluator_system_prompt_json_example = json.dumps(
            self.JSON_EXAMPLES[config["evaluator"]["system_prompt_json_example"]],
            ensure_ascii=False,
        )
        self.evaluator_fewshot_correct = json.dumps(
            self.JSON_FEWSHOT_EXAMPLES["correct"][
                config["evaluator"]["fewshot_prompt"]
            ],
            ensure_ascii=False,
        )
        self.evaluator_fewshot_incorrect = json.dumps(
            self.JSON_FEWSHOT_EXAMPLES["incorrect"][
                config["evaluator"]["fewshot_prompt"]
            ],
            ensure_ascii=False,
        )

    def hash(self):
        hash_str = ""
        hash_str += f"$self.interactor_system_prompt={self.interactor_system_prompt}"
        hash_str += (
            f"$self.interactor_init_user_prompt={self.interactor_init_user_prompt}"
        )
        hash_str += f"$self.candidate_system_prompt={self.candidate_system_prompt}"
        hash_str += (
            f"$self.evaluator_init_user_prompt={self.evaluator_init_user_prompt}"
        )
        hash_str += f"$self.evaluator_user_prompt={self.evaluator_user_prompt}"
        hash_str += f"$self.evaluator_system_prompt={self.evaluator_system_prompt}"
        hash_str += f"$self.evaluator_system_prompt_json_example={self.evaluator_system_prompt_json_example}"
        hash_str += f"$self.evaluator_fewshot_correct={self.evaluator_fewshot_correct}"
        hash_str += (
            f"$self.evaluator_fewshot_incorrect={self.evaluator_fewshot_incorrect}"
        )
        hash_str = hash_str.encode("utf-8")
        hash_digest = md5(hash_str).digest()

        url_safe_hash = urlsafe_b64encode(hash_digest).rstrip(b"=").decode("utf-8")
        return url_safe_hash

    def generate_interactor_init_message(
        self, mcp_prompt: str, correct_answer: str, model_prediction: str
    ):
        interactor_init_user_prompt = self.interactor_init_user_prompt.format(
            mcp_prompt=mcp_prompt,
            correct_answer=correct_answer,
            model_prediction=model_prediction,
        )
        ret = [
            {"role": "system", "content": self.interactor_system_prompt},
            {"role": "user", "content": interactor_init_user_prompt},
        ]
        return ret

    def generate_candidate_init_message(self, mcp_prompt: str, model_prediction: str):
        ret = [
            {"role": "system", "content": self.candidate_system_prompt},
            {"role": "user", "content": mcp_prompt},
            {"role": "assistant", "content": model_prediction},
        ]
        return ret

    def generate_evaluator_init_message(
        self, mcp_prompt: str, model_prediction: str, correct_answer: str
    ):
        evaluator_system_prompt = (
            self.evaluator_system_prompt + self.evaluator_system_prompt_json_example
        )
        evaluator_user_prompt = self.evaluator_init_user_prompt.format(
            mcp_prompt=mcp_prompt, model_prediction=model_prediction
        )
        if model_prediction == correct_answer:
            evaluator_assistant_prompt = self.evaluator_fewshot_correct
        else:
            evaluator_assistant_prompt = self.evaluator_fewshot_incorrect
        ret = [
            {"role": "system", "content": evaluator_system_prompt},
            {"role": "user", "content": evaluator_user_prompt},
            {"role": "assistant", "content": evaluator_assistant_prompt},
        ]
        return ret

    def generate_init_messages(self, mcp):
        uuid = mcp.uuid
        model_prediction = mcp.generate_prediction_output_text()  # TODO: fix hard-coded
        correct_answer = mcp.generate_output_text()
        mcp_prompt = self.mcp_prompt.format(
            problem=mcp.problem, choices=mcp.generate_choices_text()
        )
        ret = {
            "interactor": self.generate_interactor_init_message(
                mcp_prompt=mcp_prompt,
                correct_answer=correct_answer,
                model_prediction=model_prediction,
            ),
            "candidate": self.generate_candidate_init_message(
                mcp_prompt=mcp_prompt, model_prediction=model_prediction
            ),
            "evaluator": self.generate_evaluator_init_message(
                mcp_prompt=mcp_prompt,
                model_prediction=model_prediction,
                correct_answer=correct_answer,
            ),
        }
        return ret

    def apply_evaluator_user_prompt(
        self, interactor_content: str, candidate_content: str
    ):
        return self.evaluator_user_prompt.format(
            interactor_content=interactor_content, candidate_content=candidate_content
        )


def apply_system_prompt(prompt, system_prompt_type="default"):
    if system_prompt_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system_prompt_type: {system_prompt_type}")
    return SYSTEM_PROMPTS[system_prompt_type].format(prompt=prompt)


def apply_multiple_choice_prompt(
    problem: str, choices: str, tmpl_name: str = "default"
):
    if tmpl_name not in MULTIPLE_CHOICE_PROMPT_TMPL:
        raise ValueError(f"Invalid tmpl_name: {tmpl_name}")
    tmpl = MULTIPLE_CHOICE_PROMPT_TMPL[tmpl_name]
    return tmpl.format(problem=problem, choices=choices)


class PromptPostprocessor:
    def __init__(
        self,
        tokenizer_name_or_path: str = None,
        multiple_choice_template_name: str = "default",
        system_prompt: Optional[str] = None,
        remove_system_prompt: bool = False,
        add_generation_prompt: bool = False,
    ):
        self.logger = logging.getLogger(__name__)

        if tokenizer_name_or_path:
            self.logger.warning(f"Loading AutoTokenizer from: {tokenizer_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, verbose=False, trust_remote_code=True
            )
        else:
            self.logger.warning(
                f"No tokenizer_name_or_path provided, using only `content` field in conversation."
            )
            self.tokenizer = None

        self.multiple_choice_template_name = multiple_choice_template_name
        self.multiple_choice_template = MULTIPLE_CHOICE_PROMPT_TMPL[
            multiple_choice_template_name
        ]
        self.system_prompt = system_prompt
        self.remove_system_prompt = remove_system_prompt
        self.add_generation_prompt = add_generation_prompt

    def get_full_prompt_from_conversation(
        self, conversation: List[Dict[str, str]], sep="\n\n", rm_last_sep=False
    ) -> str:
        if self.tokenizer is not None:
            if self.remove_system_prompt and conversation[0]["role"] == "system":
                txt = self.tokenizer.apply_chat_template(
                    conversation[1:],
                    tokenize=False,
                    padding=False,
                    truncation=False,
                    add_generation_prompt=self.add_generation_prompt,
                )
            else:
                txt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    padding=False,
                    truncation=False,
                    add_generation_prompt=self.add_generation_prompt,
                )
        else:
            txt = ""
            for conv in conversation:
                txt += f"{conv['content']}{sep}"
            if rm_last_sep:
                txt = txt[: -len(sep)]
        return txt

    def create_conversation_from_mcp(self, problem, fewshot_examples) -> List[Dict]:
        conversation = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )

        if len(fewshot_examples) > 0:
            for eg in fewshot_examples:
                question_txt = apply_multiple_choice_prompt(
                    eg.problem,
                    eg.generate_choices_text(self.multiple_choice_template_name),
                    self.multiple_choice_template_name,
                )
                question_ans = eg.generate_output_text(
                    self.multiple_choice_template_name
                )
                conversation.append({"role": "user", "content": question_txt})
                conversation.append({"role": "assistant", "content": question_ans})

        question_txt = apply_multiple_choice_prompt(
            problem.problem,
            problem.generate_choices_text(self.multiple_choice_template_name),
            self.multiple_choice_template_name,
        )
        conversation.append({"role": "user", "content": question_txt})
        return conversation

    def create_conversation_from_qa(self, problem, fewshot_examples) -> List[Dict]:
        conversation = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )

        if len(fewshot_examples) > 0:
            for eg in fewshot_examples:
                question_txt = apply_multiple_choice_prompt(
                    eg.problem,
                    eg.generate_choices_text(self.multiple_choice_template_name),
                    self.multiple_choice_template_name,
                )
                question_ans = eg.generate_output_text(
                    self.multiple_choice_template_name
                )
                conversation.append({"role": "user", "content": question_txt})
                conversation.append({"role": "assistant", "content": question_ans})

        question_txt = apply_multiple_choice_prompt(
            problem.problem,
            problem.generate_choices_text(self.multiple_choice_template_name),
            self.multiple_choice_template_name,
        )
        conversation.append({"role": "user", "content": question_txt})
        return conversation

    def get_full_prompt_from_problem(self, problem, fewshot_examples) -> str:
        if len(problem.choices) > 1:  # multiple choice
            conversation = self.create_conversation_from_mcp(problem, fewshot_examples)
            return self.get_full_prompt_from_conversation(
                conversation, rm_last_sep=False
            )
        else:  # cloze question
            conversation = self.create_conversation_from_qa(problem, fewshot_examples)
            return self.get_full_prompt_from_conversation(
                conversation, rm_last_sep=True
            )
