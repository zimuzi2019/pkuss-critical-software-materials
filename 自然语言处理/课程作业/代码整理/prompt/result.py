import json

def get_llmbar_res():
    with open('avg_results.json', 'r') as f:
        data = json.load(f)
    max_num = max(data.keys())
    return data[max_num]

def get_other_res():
    with open('result.json', 'r') as f:
        data = json.load(f)

    return [x for x in data if 'LLMBar' not in x]