import random
import re 

from simple_parsing import ArgumentParser
from datasets import load_dataset


# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/gpqa/utils.py
def gpqa_process_doc(doc):
    def preprocess(text):
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    choices = [
        preprocess(doc["Incorrect Answer 1"]),
        preprocess(doc["Incorrect Answer 2"]),
        preprocess(doc["Incorrect Answer 3"]),
        preprocess(doc["Correct Answer"]),
    ]

    random.shuffle(choices)
    correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/gpqa/_template_yaml
    formatted = f"What is the correct answer to this question:{doc['Question']}\nChoices:\n(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}"

    out_doc = {
        "choice1": choices[0],
        "choice2": choices[1],
        "choice3": choices[2],
        "choice4": choices[3],
        "answer": f"({chr(65 + correct_answer_index)})",
        "formatted_question": formatted,
    }
    return out_doc


def download_gpqa_diamond():
    random.seed(0)
    data = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    data = data.map(gpqa_process_doc)
    data.to_json("data/gpqa/gpqa_diamond.jsonl", lines=True)


def download_gpqa_main():
    data = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    data = data.map(gpqa_process_doc)
    data.to_json("data/gpqa/gpqa_main.jsonl", lines=True)


def download_hle():
    data = load_dataset("cais/hle", split="test")
    data = data.filter(lambda x: x['image'])
    data = data.remove_columns(["image", "image_preview", "rationale_image"])
    data.to_json("data/hle/hle.jsonl", lines=True)


def alpaca_eval():
    data = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    data.to_json("data/alpaca_eval/alpaca_eval.jsonl", lines=True)


def arena_hard():
    data = load_dataset("json", data_files="/scratch/gpfs/hyen/p-open-ended-retrieval/arena-hard-auto/data/arena-hard-v0.1/question.jsonl")['train']
    data.to_json('data/arena_hard/arena_hard.jsonl', lines=True)


mapping = {
    "gpqa_diamond": download_gpqa_diamond,
    "gpqa_main": download_gpqa_main,
    "hle": download_hle,
    "alpaca_eval": alpaca_eval,
    "arena_hard": arena_hard
}


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_argument("--dataset", type=str, required=True, help="dataset to download, all to download everything")
    args = parser.parse_args()

    if args.dataset == 'all':
        for k, v in mapping.items():
            v()
    elif args.dataset not in mapping:
        raise ValueError(f"Data name {args.dataset} not found in mapping.")
    else:
        mapping[args.dataset]()
 