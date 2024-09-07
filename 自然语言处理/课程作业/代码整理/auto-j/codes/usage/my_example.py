from vllm import LLM, SamplingParams
import torch
from constants_prompt import build_autoj_input


def extract_pariwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'):
            pred_label = 0
        elif pred_rest.startswith('response 2'):
            pred_label = 1
        elif pred_rest.startswith('tie'):
            pred_label = 2
    return pred_label


def extract_single_rating(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("Rating: [["):pos2].strip())
    else:
        return 0.0


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()    
    
    pairwise_data = []
    
    import json
    
    with open("/home2/xjw/auto-j/data/test/testdata_pairwise.jsonl", "r") as f:
      pairwise_data = [json.loads(line) for line in f.readlines()]
    
    input_pairwise = []
    
    for i in range(len(pairwise_data)):
        input_pairwise.append(
          build_autoj_input(
            prompt=pairwise_data[i]["prompt"],
            resp1=pairwise_data[i]["response 2"],
            resp2=pairwise_data[i]["response 1"],
            protocol="pairwise_tie"
          ))    
    
    model_name_or_dir = "/home2/xjw/LLaMA-Factory/models/llama3_lora_sft"
    llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

    input = input_pairwise  # or input_single

    outputs = llm.generate(input, sampling_params)
    
    judgment = []
    
    for i in range(len(outputs)):
        judgment.append(outputs[i].outputs[0].text)
        
    evaluation_result = []
    
    for i in range(len(judgment)):
      result = extract_pariwise_result(judgment[i])
      evaluation_result.append(result)
      
    with open("prediction_exchange.jsonl", "w") as f:
      for i in range(len(evaluation_result)):
        o = {"output": evaluation_result[i]}
        f.write(json.dumps(o) + "\n")