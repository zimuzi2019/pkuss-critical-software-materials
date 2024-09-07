import json
from prompt import prompt1
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils import process_data, metrics, save_output
# from utils import load_data
import os
import random
import pandas as pd
lock = threading.Lock()  # 创建一个锁对象

# [question, answer(list), ref, mode]
class TestSet(object):
    name = ''
    path = ''
    num_label = 1
    mode = 0
    data_type = 0
    def __init__(self, path, mode=0):
        self.path = path
        self.mode = mode

    def load_data(testset):
        if testset.data_type == 0:
            with open(testset.path, 'r') as f:
                data = json.load(f)
        elif testset.data_type == 1:
            data = []
            with open(testset.path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif testset.data_type == 2:
            parquet_data = pd.read_parquet(testset.path)
            data = parquet_data.to_dict('records')
            data = random.sample(data, len(data)//5)

        return data

    def eval(self, prompt_fn=prompt1):
        result, output = self.process(prompt_fn)
        result = { self.name : result, 'prompt_type': prompt_fn.__name__}
        self.save_metrics(result)
        # save_output(self, output, prompt_fn.__name__)


    def build_data(self, data):
        return Counter(), ()

    def process(self, prompt_fn=prompt1, set_name=None):
        self.data = self.load_data(self)
        num_tokens = 0
        # num_sample = len(data) * 3 # 3个标注人
        TP = Counter()
        P = Counter()
        class_cnt = Counter()
        outputs = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_data, d, self, prompt_fn): d for d in self.data}
            for future in tqdm(as_completed(futures), total=len(self.data), desc='predict'):
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
        save_output(self, outputs, prompt_fn.__name__, set_name)
        result = metrics(self.name, num_sample, class_cnt, TP, P, cost)  #{zhibiao:}


        return result, outputs


    def save_metrics(self, result):
        with open('result.json', 'r') as f:
            results = json.load(f)
        results.append(result)
        with open('result.json', 'w') as f:
            json.dump(results, f, indent=4)


class PandaLM(TestSet):
    name = 'PandaLM'
    num_label = 3
    data_type = 0


    def build_data(self, data):
        label = Counter([data[f'annotator{i}'] for i in range(1,4)])
        question = data['instruction'] + '\n' + data['input']
        answer = [data['response1'], data['response2']]
        ref = None
        mode = 0
        return label, (question, answer, ref, mode)


"""{'scenario': 'post_summarization',
 'label': 1,
 'prompt': "Summarize the following post\n\nTitle: I (23 F) had sex with a friend (27 M) who turns out has a girlfriend (27/28 F). Should I remain friends with him?\n\nPost: Hey,  r/relationships! Looks like I'm beginning to be a regular poster here if you include my primary account. Not the best feeling though, haha. Anyways. \n\nI met him (27 M) during the time that I was on the rocks with another relationship (LDR,  if that matters). During that time I never hit on him even though I felt attracted to him because of my relationship. When that relationship ended, he was one of the people who helped me get back on my feet and even went with me on a holiday because the ex (25 M) and I planned it out before breaking up. So as I tried to get over my ex, I had sex with the guy because based from his stories, he told me that he's been single for more than a year and I was beginning to trust him more. \n\nThat was a month ago, and just recently, have hooked up again, twice. The red flag I should have noticed is he refuses to be Facebook friends or even let me follow his Twitter as he's following mine. The only thing I follow him on is Instagram, which is how I learned that he has a girlfriend. \n\nThe girlfriend (27/28 F) knew about me THE ENTIRE TIME and decided to post about me on her blog, telling that I chose to play a losing battle. The thing is, I never knew. He's only told me of having a girl best friend but never the degree. \n\nI'm more hurt about this betrayal than the relationship I had that I just recently moved on from because I really thought that this guy would be honest and a special person in my life. I told my closest friends what to do and they told me to stick to a decision I can live through. \n\nSo this is why I'm here, yet again. I still want him to be a part of my life and he's told me he'd still stick around (I know, words are nothing) but I think that if only I didn't have sex with him then I would forgive him easily.",
 'response 1': ' Friend (27 M) of 3 months had sex with me (23 F) not knowing he has a girlfriend (27/28 F). Told me GF knows about it. Should I keep the friendship with him?',
 'response 2': ' I had sex with a friend who turned out to have a girlfriend. Should I remain friends with him? Should I just move on?'}"""
class Auto_J(TestSet):
    name = 'auto-j'
    num_label = 1
    data_type = 1

    # def __init__(self, mode):
    #     self.path = 'auto-j/testdata_pairwise.jsonl' if mode==0 else 'auto-j/testdata_critique.jsonl'
    #     self.mode = mode
    def build_data(self, data):
        # label = Counter([data['label']])
        label = Counter([(data['label']+1) % 3])
        question = 'scenario: {}\n'.format(data['scenario']) + data['prompt']
        answer = [data['response 1'], data['response 2']]
        ref = None

        return label, (question, answer, ref, self.mode)


class LLMBar(TestSet):
    name = 'LLMBar'
    num_label = 1
    data_type = 0

    def __init__(self, path, dirs, mode=0):
        super().__init__(path, mode)
        self.data_file = path
        self.dirs = dirs

    def eval(self, prompt_fn=prompt1):
        results = dict()
        for dir in self.dirs:
            set_name = dir.split('/')[-1]
            self.path = os.path.join(dir, self.data_file)
            result, output = self.process(prompt_fn=prompt_fn, set_name=set_name)
            # save_output(self, output, prompt_fn.__name__, set_name)
            results[set_name] = result
        results = { self.name:results, 'prompt_type': prompt_fn.__name__}
        self.save_metrics(results)


    def build_data(self, data):
        # label = Counter([(data['label']+1) % 3])
        label = Counter([data['label']])
        question = data['input']
        answer = [data['output_1'], data['output_2']]
        ref = None
        return label, (question, answer, ref, self.mode)


"""{'question_id': 81,
  'model_a': 'alpaca-13b',
  'model_b': 'gpt-3.5-turbo',
  'winner': 'model_b',
  'judge': 'author_2',
  'conversation_a': array([{'content': 'Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.', 'role': 'user'},
         {'content': 'I recently had the pleasure of visiting Hawaii and it quickly became one of my favorite places. From the stunning beaches to the lush mountains, this place has it all. The people are incredibly friendly and the culture is alive and well. One of the highlights of my trip was visiting the Polynesian Cultural Center. Here, I was able to learn about the culture of the native Hawaiian people and try my hand at traditional crafts and activities. I also had a chance to explore some of the natural wonders of the island, including the breathtaking Hanauma Bay and the majestic Waimea Canyon. Whether you’re looking for a relaxing beach vacation or an adventure filled with culture and nature, Hawaii is the perfect destination.', 'role': 'assistant'},
         {'content': 'Rewrite your previous response. Start every sentence with the letter A.', 'role': 'user'},
         {'content': 'Aloha! I recently had the pleasure of visiting Hawaii and it quickly became one of my favorite places. From the stunning beaches to the lush mountains, this place has it all. The people are incredibly friendly and the culture is alive and well. One of the highlights of my trip was visiting the Polynesian Cultural Center. Here, I was able to learn about the culture of the native Hawaiian people and try my hand at traditional crafts and activities. I also had a chance to explore some of the natural wonders of the island, including the breathtaking Hanauma Bay and the majestic Waimea Canyon. Whether you’re looking for a relaxing beach vacation or an adventure filled with culture and nature, Hawaii is the perfect destination.', 'role': 'assistant'}],
        dtype=object),
  'conversation_b': array([{'content': 'Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.', 'role': 'user'},
         {'content': "Aloha! I recently had the pleasure of embarking on a trip to the beautiful island of Hawaii, and let me tell you, the cultural experiences and must-see attractions did not disappoint.\n\nFirst on my list was a visit to the Polynesian Cultural Center. This interactive experience immerses you in the cultures of the Pacific Islands, from the intricate dances of the Maori people of New Zealand to the fire knife dancing of Samoa. The center also features a canoe pageant, where different island nations showcase their unique styles of canoeing. It was truly a feast for the senses and a fascinating insight into the diverse cultures of the Pacific.\n\nNext up was a trip to the North Shore, which boasts some of the best surf spots in the world. I watched in awe as surfers of all levels tackled the massive waves, and even had the chance to take a lesson myself. It was an exhilarating experience, and I left with a newfound respect for the power of the ocean.\n\nOf course, no trip to Hawaii would be complete without a visit to Pearl Harbor. The somber memorial serves as a reminder of the sacrifices made during World War II, and it was a deeply moving experience to pay my respects to the brave men and women who lost their lives on that fateful day.\n\nLast but not least, I made sure to indulge in some of the local cuisine. From poke bowls to shave ice, the flavors of Hawaii are truly unique and delicious. A personal favorite was the plate lunch, which typically consists of a protein, rice, and macaroni salad. It may not be the healthiest option, but it sure is tasty!\n\nOverall, my trip to Hawaii was an unforgettable experience. The cultural immersion, natural beauty, and delicious food all contributed to an incredible adventure. If you're looking for a destination that has it all, Hawaii should definitely be at the top of your list. Mahalo for reading!", 'role': 'assistant'},
         {'content': 'Rewrite your previous response. Start every sentence with the letter A.', 'role': 'user'},
         {'content': "Absolutely! A recent trip to the beautiful island of Hawaii was an adventure that I will never forget.\n\nAs a helpful assistant, I highly recommend a visit to the Polynesian Cultural Center as it immerses you in the cultures of the Pacific Islands.\n\nAlso, a trip to the North Shore is a must-see attraction, where you can witness surfers of all levels tackling the massive waves.\n\nAdditionally, paying respects to the brave men and women who lost their lives at Pearl Harbor was a deeply moving experience.\n\nAnd let's not forget about the delicious local cuisine, from poke bowls to shave ice, the flavors of Hawaii are truly unique and tasty.\n\nAll in all, Hawaii has it all - cultural immersion, natural beauty, and delicious food. As an assistant, I encourage you to put Hawaii at the top of your travel list. Aloha!", 'role': 'assistant'}],
        dtype=object),
  'turn': 1}"""
class MTBench(TestSet):
    name = 'MT-Bench'
    num_label = 1
    data_type = 2 # parquet

    def build_data(self, data):
        label_map = {'tie':0, 'model_a':1, 'model_b':2}
        label = Counter([label_map[data['winner']]])
        question = None
        answer = [data['conversation_a'].tolist(), data['conversation_b'].tolist()]
        ref = None
        return label, (question, answer, ref, self.mode, True)

# pandalm = PandaLM()
# auto_j_pairwise = Auto_J(mode=0)
# auto_j_critique = Auto_J(mode=2)
# pandalm = {'name':'PandaLM', 'path': 'testset-v1.json', 'fn': pandalm_fn}
# auto_j_pairwise = {'name': 'auto_j', 'path': 'auto-j/testdata_pairwise.jsonl', 'fn': auto_j_fn}
if __name__ == '__main__':
    dirs = ['LLMBar/Natual'] + [f'LLMBar/Adversarial/{set_name}' for set_name in [ 'GPTInst', 'GPTOut', 'Manual', 'Neighbor']]
    for dir in dirs:
        print(dir.split('/')[-1])
