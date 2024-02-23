import math
import json
import argparse
import os

ATTRIBUTES = ["accuracy", "logic", "relevance", "coherence", "conciseness", "overall"]


def get_score_list(data, attribute):
    return (
        [
            [
                data[i]["evaluation_results"][j][attribute]["score"]
                for j in range(len(data[i]["evaluation_results"]))
                if data[i]["evaluation_results"][j][attribute]["score"] is not None
            ]
            for i in range(len(data))
        ]
        if attribute != "overall"
        else [
            [
                data[i]["evaluation_results"][j]["overall_score"]
                for j in range(len(data[i]["evaluation_results"]))
                if data[i]["evaluation_results"][j]["overall_score"] is not None
            ]
            for i in range(len(data))
        ]
    )


def get_attribute_score(data, attribute, scoring_function):
    score_list = get_score_list(data, attribute)
    return sum(scoring_function(scores) for scores in score_list) / len(score_list)


score_mapper = {
    0: 0.0,
    1: 1.0,
    2: 3.0,
    3: 7.0,
    4: 10.0,
}


def get_stop_reasons(data):
    stop_reason_counter = {}
    for item in data:
        try:
            stop_reason = item["evaluation_results"][-1]["stop_reason"]
            if stop_reason not in stop_reason_counter:
                stop_reason_counter[stop_reason] = 0
            stop_reason_counter[stop_reason] += 1
            if stop_reason == "unpermitted_role_shift":
                print(item["all_messages"])
        except:
            print(item["evaluation_results"][-1], item["uuid"])
    return stop_reason_counter


def weighted_mean(scores):
    weights = [math.exp(-0.2 * index) for index in range(len(scores))]
    return (
        sum(score_mapper[score] * weight for score, weight in zip(scores, weights))
        / sum(weights)
        * 10.0
    )


def get_all_scores(data) -> dict:
    scores = {}
    for attribute in ATTRIBUTES:
        scores[attribute] = get_attribute_score(data, attribute, weighted_mean)
    conversation_lengths = []
    for item in data:
        conversation_lengths.append(len(item["evaluation_results"]))
    scores["conversation_completeness"] = sum(conversation_lengths) / len(
        conversation_lengths
    )
    scores["stop_reasons"] = get_stop_reasons(data)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get scores from evaluation results.")
    parser.add_argument(
        "-r",
        "--results-dir",
        type=str,
        required=True,
        help="Specify the path to the evaluation results dir.",
        default="main_results",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Specify the path to the evaluation outputs dir.",
        default="main_outputs",
    )
    parser.add_argument(
        "-s",
        "--score-file",
        type=str,
        required=True,
        help="Specify the path to save the scores.",
        default="scores.json",
    )
    args = parser.parse_args()

    scores = {}
    for file in os.listdir(args.results_dir):
        if file.endswith(".json"):
            with open(os.path.join(args.results_dir, file), "r") as f:
                data = json.load(f)
                mcp_acc = -1
                for key in data:
                    if key.startswith("interactive_evaluation_step:"):
                        hash = data[key]["hash"]
                    if key.startswith("simple_multiple_choice") and mcp_acc < 0:
                        mcp_acc = data[key]["accuracy"]
                with open(
                    os.path.join(
                        args.output_dir, f"interactive_{hash}/interact_details.json"
                    ),
                    "r",
                ) as f:
                    data = json.load(f)
                scores[file] = get_all_scores(data)
                scores[file]["mcp_acc"] = mcp_acc * 100.0

    # sort scores according to the overall score
    scores = {
        k: v
        for k, v in sorted(
            scores.items(), key=lambda item: item[1]["overall"], reverse=True
        )
    }

    scores_by_dataset = {}
    for file in scores:
        model, dataset = file.split("@")
        if dataset not in scores_by_dataset:
            scores_by_dataset[dataset] = {}
        scores_by_dataset[dataset][model] = scores[file]

    with open(args.score_file, "w") as f:
        json.dump(scores_by_dataset, f, indent=4)
