import time
import json
from zhipuai import ZhipuAI
from prompt import prompt1
from tqdm import tqdm
from collections import Counter
import os
from testset import pandalm
client = ZhipuAI(api_key="e77a4fc1fc878c490d175aadd5f392b0.a6RDg5rsvHkgXmCR") # 请填写您自己的APIKey
os.chdir(os.path.dirname(__file__))
temperature = 0.3
def metrics(name, num_sample, class_cnt, TP, P):
    accuracy = sum([cnt for cnt in TP.values()])/ 300
    recall_list = {i:TP[i]/class_cnt[i] for i in range(3)}
    precision_list = {i:TP[i]/P[i] for i in range(3)}
    recall = accuracy
    precision = sum([precision_list[i]*class_cnt[i] for i in range(3)]) / num_sample
    f1 = 2*(precision*recall/(precision + recall))
    result = {name:{'Accuracy': accuracy, 'Precision':precision, 'Recall': recall, 'F1': f1}}
    with open('result.json','a') as f:
        json.dump(result, f, indent=4)
    with open('mid_res.txt', 'a') as f:
        f.write(name+'\n')
        f.write(str(num_sample)+'\n')
        f.write(str(class_cnt)+'\n')
        f.write(str(P)+'\n')
        f.write(str(TP)+'\n')
        f.write(str(recall_list)+'\n')
        f.write(str(precision_list)+'\n')
    print(result)

def main(testset=pandalm, prompt_fn=prompt1):

    with open(testset['path'], 'r') as f:
        data=json.load(f)
    num_tokens = 0
    num_sample = len(data) * 3 # 3个标注人
    TP=Counter()
    P=Counter()
    class_cnt = Counter()

    # always: 0==tie
    for data in tqdm(data, desc='predict'):
        # label = Counter([data[f'annotator{i}'] for i in range(1,4)])
        label, params = testset['fn'](data)
        class_cnt += label

        message, sign = prompt_fn(*params)
        response = client.chat.completions.create(
            model='glm-3-turbo',
            messages=message,
            temperature=temperature
        )
        num_tokens += response.usage.total_tokens
        res = response.choices[0].message.content
        if sign[1] in res:
            TP[1] += label[1]
            P[1] += 3
        elif sign[2] in res:
            TP[2] += label[2]
            P[2] += 3
        elif sign[0] in res:
            TP[0] += label[3]
            P[0] += 3

    cost = num_tokens/1000*0.001
    print(f'cost:{cost}')
    metrics(testset['name'], num_sample, class_cnt, TP, P)
    # print('accuray', accuracy)
    # print(recalls, precisions)
    # print(avg_recall)
    # print(avg_precision)


if __name__ == "__main__":

    main(pandalm, prompt1)




