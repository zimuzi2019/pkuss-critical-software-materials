# import time
import json
from zhipuai import ZhipuAI
# from prompt import prompt1
from tqdm import tqdm
from collections import Counter
import os
# from testset import pandalm, auto_j_pairwise
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import random


lock = threading.Lock()  # 创建一个锁对象
client = ZhipuAI(api_key="e77a4fc1fc878c490d175aadd5f392b0.a6RDg5rsvHkgXmCR") # 请填写您自己的APIKey
os.chdir(os.path.dirname(__file__))
temperature = 0.3
debug = False

# def load_data(testset):
#     if testset.data_type == 0:
#         with open(testset.path, 'r') as f:
#             data = json.load(f)
#     elif testset.data_type == 1:
#         data = []
#         with open(testset.path, 'r') as f:
#             for line in f:
#                 data.append(json.loads(line.strip()))
#     elif testset.data_type == 2:
#         parquet_data = pd.read_parquet(testset.path)
#         data = parquet_data.to_dict('records')
#         data = random.sample(data, len(data)//5)

#     return data

def save_output(testset, output, prompt_name, set_name=None):
    file_name = f'{testset.name}_output_{prompt_name}_{set_name}.json' if set_name else f'{testset.name}_output_{prompt_name}.json'

    with open(f'outputs/{file_name}', 'w') as f:
        json.dump(output, f)


def metrics(name, num_sample, class_cnt, TP, P, cost):
    category = list(class_cnt.keys())
    accuracy = sum([cnt for cnt in TP.values()])/ num_sample
    # recall_list = {i:TP[i]/class_cnt[i] for i in range(3)}
    # precision_list = {i:TP[i]/P[i] for i in range(3)}
    # recall = sum([recall_list[i]*class_cnt[i] for i in range(3)]) / num_sample
    # precision = sum([precision_list[i]*class_cnt[i] for i in range(3)]) / num_sample
    recall_list = {i:TP[i]/class_cnt[i] for i in category}
    precision_list = {i:TP[i]/P[i] for i in category if P[i]!=0}
    recall = sum([recall_list[i]*class_cnt[i] for i in category if i in recall_list.keys()]) / num_sample
    precision = sum([precision_list[i]*class_cnt[i] for i in category if i in precision_list.keys()]) / num_sample
    f1 = 2*(precision*recall/(precision + recall))
    result = {'Accuracy': accuracy, 'Precision':precision, 'Recall': recall, 'F1': f1}
    # with open('result.json', 'r') as f:
    #     old_metrics = json.load(f)
    # old_metrics.append(result)
    # with open('result.json','w') as f:
    #     json.dump(old_metrics, f, indent=4)
    #     f.write(',\n')
    with open('mid_res.txt', 'a') as f:
        f.write(name+'\n')
        f.write(str(num_sample)+'\n')
        f.write(str(class_cnt)+'\n')
        f.write(str(P)+'\n')
        f.write(str(TP)+'\n')
        f.write(str(recall_list)+'\n')
        f.write(str(precision_list)+'\n')
        f.write(str(cost) + '\n\n')
    print(result)
    return result




def process_data(data, testset, prompt_fn):
    num_label = testset.num_label
    label, params = testset.build_data(data)
    TP, P = Counter(), Counter()
    class_cnt = Counter()
    num_tokens = 0
    res = None
    message, sign = prompt_fn(*params)
    if debug:
        print(message)
    try:
        response = client.chat.completions.create(
            model='glm-3-turbo',
            messages=message,
            temperature=temperature
        )
    except Exception as e:
        print(e)
    else:
        class_cnt = label
        num_tokens = response.usage.total_tokens
        res = response.choices[0].message.content

        if sign[1] in res:
            TP[1] += label[1]
            P[1] += num_label
        elif sign[2] in res:
            TP[2] += label[2]
            P[2] += num_label
        elif sign[0] in res:
            TP[0] += label[3]
            P[0] += num_label

    return num_tokens, TP, P, class_cnt, res

def main(testset, prompt_fn):
    data = testset.load_data()
    num_tokens = 0
    # num_sample = len(data) * 3 # 3个标注人
    TP = Counter()
    P = Counter()
    class_cnt = Counter()
    outputs = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_data, d, testset, prompt_fn): d for d in data}
        for future in tqdm(as_completed(futures), total=len(data), desc='predict'):
            try:
                result = future.result()
                with lock:  # 确保以下操作是线程安全的
                    num_tokens += result[0]
                    TP += result[1]
                    P += result[2]
                    class_cnt += result[3]
                    outputs.append({"output": result[4]})

            except Exception as exc:
                print(f'Generated an exception: {exc}')

    cost = num_tokens / 1000 * 0.001
    num_sample = sum(class_cnt.values())
    print(f'cost: {cost}')
    metrics(testset.name, num_sample, class_cnt, TP, P, cost)
    save_output(testset, outputs, prompt_fn.__name__)

# Example usage
# main(testset=pandalm, prompt_fn=prompt1)



# if __name__ == "__main__":
#     prompt_fn = prompt1

#     # main(pandalm, prompt_fn)

#     main(auto_j_pairwise, prompt_fn)