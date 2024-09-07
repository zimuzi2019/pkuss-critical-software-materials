import os
from testset import PandaLM, Auto_J, LLMBar, MTBench
from prompt import prompt1, prompt1_cn, prompt_vanila, prompt_CoT, prompt_vanila_rule, prompt_CoT_rule
os.chdir(os.path.dirname(__file__))

# def main(testset, prompt_fn):
#     data = load_data(testset)
#     num_tokens = 0
#     # num_sample = len(data) * 3 # 3个标注人
#     TP = Counter()
#     P = Counter()
#     class_cnt = Counter()
#     outputs = []

#     with ThreadPoolExecutor(max_workers=4) as executor:
#         futures = {executor.submit(process_data, d, testset, prompt_fn): d for d in data}
#         for future in tqdm(as_completed(futures), total=len(data), desc='predict'):
#             try:
#                 result = future.result()
#                 with lock:  # 确保以下操作是线程安全的
#                     num_tokens += result[0]
#                     TP += result[1]
#                     P += result[2]
#                     class_cnt += result[3]
#                     outputs.append({"output": result[4]})

#             except Exception as exc:
#                 print(f'Generated an exception: {exc}')

#     cost = num_tokens / 1000 * 0.001
#     num_sample = sum(class_cnt.values())
#     print(f'cost: {cost}')
#     metrics(testset.name, num_sample, class_cnt, TP, P, cost)
#     save_output(testset, outputs, prompt_fn.__name__)

# Example usage
# main(testset=pandalm, prompt_fn=prompt1)

def test_all(pandalm, auto_j, llmbar, mt_bench, prompt_fn_list):
    for prompt_fn in prompt_fn_list:
        pandalm.eval(prompt_fn)
        auto_j.eval(prompt_fn)
        llmbar.eval(prompt_fn)
        mt_bench.eval(prompt_fn)

def repeat_test_llmbar(llmbar, prompt_fn_list, repeat_time):
    for i in range(repeat_time):
        for prompt_fn in prompt_fn_list:
            llmbar.eval(prompt_fn)
        calc_avg_llmbar()

def calc_avg_llmbar():
    import json
    with open('result.json', 'r') as f:
        results = json.load(f)
    results = [d for d in results if "LLMBar" in d]
    avg_results = {}
    cnt={}
    num=0
    for d in results:
        if d['prompt_type'] not in avg_results:
            avg_results[d['prompt_type']] = d['LLMBar']
            cnt[d['prompt_type']] = 1
        else:
            for set_name in avg_results[d['prompt_type']].keys():
                for metric in avg_results[d['prompt_type']][set_name].keys():
                    avg_results[d['prompt_type']][set_name][metric] += d['LLMBar'][set_name][metric]
            cnt[d['prompt_type']] += 1
            num = max(num, cnt[d['prompt_type']])
    avg_results_list = []
    for prompt_type in avg_results.keys():
        for set_name in avg_results[prompt_type].keys():
            for metric in avg_results[prompt_type][set_name].keys():
                avg_results[prompt_type][set_name][metric] /= cnt[prompt_type]
        avg_results_list.append({'LLMBar':avg_results[prompt_type], 'prompt_type': prompt_type})

    print(avg_results_list)
    with open('avg_results.json', 'r') as f:
        avg_results_old = json.load(f)

    with open('avg_results.json', 'w') as f:
        json.dump({**avg_results_old, **{num:avg_results_list}}, f, indent=4)



if __name__ == "__main__":
    # prompt_fn = prompt1
    # prompt_fn = prompt1_cn
    # prompt_fn = prompt_vanila
    # prompt_fn = prompt_vanila_rule

    prompt_fn = prompt_CoT_rule
    pandalm = PandaLM(path='testset-v1.json')
    auto_j = Auto_J(path='auto-j/testdata_pairwise.jsonl')
    llmbar = LLMBar(path='dataset.json', dirs=['LLMBar/Natural'] + [f'LLMBar/Adversarial/{set_name}' for set_name in [ 'GPTInst', 'GPTOut', 'Manual', 'Neighbor']])
    mt_bench = MTBench(path='mt-bench/human-00000-of-00001-25f4910818759289.parquet')
    # pandalm.eval(prompt_fn)
    # auto_j.eval(prompt_fn)
    # llmbar.eval(prompt_fn)
    # mt_bench.eval(prompt_fn)

    # test_all(pandalm, auto_j, llmbar, mt_bench, [prompt1_cn])
    # test_all(pandalm, auto_j, llmbar, mt_bench, [prompt_CoT])
    # calc_avg_llmbar()
    repeat_test_llmbar(llmbar, [prompt1, prompt1_cn, prompt_vanila, prompt_CoT, prompt_vanila_rule, prompt_CoT_rule], 2)