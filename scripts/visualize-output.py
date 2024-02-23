import gradio as gr
import json, json5
import os
import argparse


def parse_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def parse_json_data(content: str):
    if "{" not in content or "}" not in content:
        return {}
    content = content[content.find("{") : content.rfind("}") + 1]
    try:
        data = json5.loads(content)
        return data
    except Exception as e:
        return {}


def check_json(
    json_path, stop_conversation, lowest_overall_score, highest_overall_score
):
    data = parse_json(json_path)

    if data["request"]["current_party"] == "evaluator":
        eval_data = parse_json_data(
            data["response"]["choices"][0]["message"]["content"]
        )
        if eval_data == {}:
            return False
        overall_score = eval_data.get("overall_score", 0)
        if (
            eval_data.get("stop_conversation", False) == stop_conversation
            and lowest_overall_score <= overall_score <= highest_overall_score
        ):
            return True
    return False


FOLDER_ROOT = None


def visualize_conversation(json_file):
    global FOLDER_ROOT
    json_path = os.path.join(FOLDER_ROOT, json_file)
    data = parse_json(json_path)

    chat_log = []
    evaluation_log = []

    content0 = data["request"]["role_messages"]["candidate"][1]["content"]

    content1 = data["request"]["role_messages"]["candidate"][2]["content"]

    content0 = content0.replace("### Question: ", "\n")
    content0 = content0.replace("### Choices: ", "\n\n")

    chat_log.append([content0, content1])

    pair = []

    cur_round = 0

    for message in data["request"]["all_messages"]:
        if message["role"] == "interactor" or message["role"] == "candidate":
            # Adding conversation messages to chat log
            speaker = (
                f"Interactor(Round {cur_round})"
                if message["role"] == "interactor"
                else f"Candidate(Round {cur_round})"
            )
            # chat_log.append([f'## {speaker}\n{message["content"]}', None])
            if message["role"] == "interactor":
                pair.append(f'## {speaker}\n{message["content"]}')
            else:
                pair.append(f'## {speaker}\n{message["content"]}')
                chat_log.append(pair.copy())
                pair = []
        elif message["role"] == "evaluator":
            # Adding evaluation results to evaluation log
            eval_data = json.loads(message["content"])
            eval_str = ""
            for key, value in eval_data.items():
                if isinstance(value, dict):
                    eval_str += (
                        f" + {key.capitalize()}({value['score']}): {value['comment']}\n"
                    )
                else:
                    eval_str += f" + {key.capitalize()}: {value}\n"
            chat_log.append([f"## Evaluation(Round {cur_round})\n{eval_str}", None])
            cur_round += 1

    if data["request"]["current_party"] == "evaluator":
        eval_data = json.loads(data["response"]["choices"][0]["message"]["content"])
        eval_str = ""
        for key, value in eval_data.items():
            if isinstance(value, dict):
                eval_str += (
                    f" + {key.capitalize()}({value['score']}): {value['comment']}\n"
                )
            else:
                eval_str += f" + {key.capitalize()}: {value}\n"
        chat_log.append([f"## Evaluation\n{eval_str}", None])

    return chat_log


# Parsing CLI arguments
parser = argparse.ArgumentParser(description="AI Conversation Visualizer")
parser.add_argument(
    "--folder_path", type=str, help="Path to folder containing JSON files"
)
args = parser.parse_args()
FOLDER_ROOT = args.folder_path
# Gradio interface


def interface(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    sorted(json_files)

    with gr.Blocks() as demo:
        gr.Markdown(
            f"## Interactive Evaluation Conversation Visualizer({len(json_files)} conversations)"
        )

        with gr.Row():
            stop_conversation = gr.Checkbox(label="Stop Conversation", value=False)
            lowest_score = gr.Number(label="Lowest Overall Score", value=0)
            highest_score = gr.Number(label="Highest Overall Score", value=5)
            filter_button = gr.Button("Filter Conversations")
            clear_filter_button = gr.Button("Clear Filter")

        with gr.Row():
            json_dropdown = gr.Dropdown(
                label="Select JSON File", choices=json_files, allow_custom_value=True
            )
            visualize_button = gr.Button("Visualize Conversation")

        with gr.Row():
            conversation_output = gr.Chatbot(
                label="Conversation", height=1000, sanitize_html=False
            )

        def get_filtered_json_files(stop_conversation, lowest_score, highest_score):
            global FOLDER_ROOT
            print(len([f for f in os.listdir(FOLDER_ROOT)]))
            json_files = [
                f
                for f in os.listdir(FOLDER_ROOT)
                if f.endswith(".json")
                and check_json(
                    os.path.join(FOLDER_ROOT, f),
                    stop_conversation,
                    lowest_score,
                    highest_score,
                )
            ]
            print(stop_conversation, lowest_score, highest_score, len(json_files))
            return gr.Dropdown(
                label="Select JSON File", choices=json_files, allow_custom_value=True
            )

        def clear_filter():
            global FOLDER_ROOT
            json_files = [f for f in os.listdir(FOLDER_ROOT) if f.endswith(".json")]
            print(stop_conversation, lowest_score, highest_score, len(json_files))
            return gr.Dropdown(
                label="Select JSON File", choices=json_files, allow_custom_value=True
            )

        filter_button.click(
            fn=get_filtered_json_files,
            inputs=[stop_conversation, lowest_score, highest_score],
            outputs=json_dropdown,
        )

        clear_filter_button.click(
            fn=clear_filter,
            inputs=[stop_conversation, lowest_score, highest_score],
            outputs=json_dropdown,
        )

        visualize_button.click(
            fn=visualize_conversation,
            inputs=[json_dropdown],
            outputs=[conversation_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=32333)


if __name__ == "__main__":
    interface(args.folder_path)
