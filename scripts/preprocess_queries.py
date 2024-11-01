import os
import json

from datasets import load_dataset

def process_alpaca_eval():
    os.makedirs("data/alpaca_eval", exist_ok=True)
    data = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    # gpt4o = load_dataset("json", data_files="/scratch/gpfs/hyen/p-open-ended-retrieval/alpaca_eval/results/gpt-4o-2024-05-13/model_outputs.json")['train']
    # gpt4o = {x['instruction']: x['output'] for x in gpt4o}

    # def get_query(d):
    #     # put all the queries together
    #     queries = {}
    #     queries['instruction'] = d['instruction']
    #     queries['davinci'] = d['output']
    #     queries['gpt4o'] = gpt4o[d['instruction']]
    #     queries['instruction_gpt4o'] = f"{d['instruction']}\n\n{gpt4o[d['instruction']]}"
    #     return {"query": queries}

    # output_data = data.map(get_query)
    # output_data.to_json(f"data/alpaca_eval/alpaca_eval_gpt4o_query.jsonl", orient="records", force_ascii=False)
    # print("Alpaca eval done")

    # 1: just the instruction
    def get_query(d):
        return {"query": d["instruction"]}
    output_data = data.map(get_query)
    output_data.to_json(f"data/alpaca_eval/alpaca_eval_instruction_query.jsonl", orient="records", force_ascii=False)

    # 2: text_davinci_003
    def get_query(d):
        return {"query": d['output']}
    output_data = data.map(get_query)
    assert set(data['generator']) == {"text_davinci_003"}
    output_data.to_json(f"data/alpaca_eval/alpaca_eval_davinci_query.jsonl", orient="records", force_ascii=False)

    # 3: gpt4o
    data = load_dataset("json", data_files="/scratch/gpfs/hyen/p-open-ended-retrieval/alpaca_eval/results/gpt-4o-2024-05-13/model_outputs.json")['train']
    def get_query(d):
        return {"query": d['output']}
    output_data = data.map(get_query)
    assert set(data['generator']) == {"gpt-4o-2024-05-13"}
    output_data.to_json(f"data/alpaca_eval/alpaca_eval_gpt4o_query.jsonl", orient="records", force_ascii=False)

    # 4: instruction + gpt4o
    def get_query(d):
        return {"query": f"{d['instruction']}\n\n{d['output']}"}
    output_data = data.map(get_query)
    output_data.to_json(f"data/alpaca_eval/alpaca_eval_gpt4o_query.jsonl", orient="records", force_ascii=False)
    
    print("Alpaca eval done")


def process_wild_bench():
    os.makedirs("data/wild_bench", exist_ok=True)
    data = load_dataset("allenai/WildBench", "v2", split="test")

    # 1: just the last turn
    def get_query(d):
        return {"query": d['conversation_input'][-1]['content']}
    output_data = data.map(get_query)
    output_data.to_json(f"data/wild_bench/wild_bench_last_turn_query.jsonl", orient="records", force_ascii=False)

    # 2: all turns
    def get_query(d):
        return {"query": "\n\n".join([turn['content'] for turn in d['conversation_input']])}
    output_data = data.map(get_query)
    output_data.to_json(f"data/wild_bench/wild_bench_all_turns_query.jsonl", orient="records", force_ascii=False)    

    # 3: gpt4
    def get_query(d):
        return {"query": d['references']['gpt-4']}
    output_data = data.map(get_query)
    output_data.to_json(f"data/wild_bench/wild_bench_gpt4_query.jsonl", orient="records", force_ascii=False)
    
    # 4: the intent
    def get_query(d):
        return {"query": d['intent']}
    output_data = data.map(get_query)
    output_data.to_json(f"data/wild_bench/wild_bench_intent_query.jsonl", orient="records", force_ascii=False)

    print("Wild bench done")

    # checklist? I feel like these could be useful but existing retrievers probably cannot handle this

def process_arena_hard():
    os.makedirs("data/arena_hard", exist_ok=True)
    data = load_dataset("json", data_files="/scratch/gpfs/hyen/p-open-ended-retrieval/arena-hard-browser/data/arena-hard-v0.1/question.jsonl")['train']

    # 1: just last turn
    def get_query(d):
        return {"query": d['turns'][-1]['content']}
    output_data = data.map(get_query)
    output_data.to_json(f"data/arena_hard/arena_hard_last_turn_query.jsonl", orient="records", force_ascii=False)
    
    # 2: gpt4o
    with open("/scratch/gpfs/hyen/p-open-ended-retrieval/arena-hard-browser/data/arena-hard-v0.1/model_answer/gpt-4o-2024-05-13.jsonl") as f:
        gpt4o_data = [json.loads(line) for line in f]
        gpt4o_data = {x["question_id"]: x['choices'][-1]['turns'][-1]['content'] for x in gpt4o_data}

    def get_query(d):
        return {"query": gpt4o_data[d['question_id']]}
    output_data = data.map(get_query)
    output_data.to_json(f"data/arena_hard/arena_hard_gpt4o_query.jsonl", orient="records", force_ascii=False)

    print("Arena hard done")


if __name__ == "__main__":
    process_alpaca_eval()
    process_wild_bench()
    process_arena_hard()