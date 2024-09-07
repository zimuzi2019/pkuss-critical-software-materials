import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # 定义一个函数逐条输出数据
def read_json_data(data):
    for item in data:
        input("按任意键读取下一条数据...")
        print(item)



def sample_json_data(data, sample_size):
    """
    从指定的JSON文件中采样指定数量的数据并返回。

    参数：
    file_path (str): JSON文件的路径。
    sample_size (int): 需要采样的数据数量。

    返回：
    list: 采样的数据列表。
    """
    import random

    # 检查采样数量是否超过数据总量
    if sample_size > len(data):
        raise ValueError("采样数量超过了数据总量")

    # 随机采样数据
    sampled_data = random.sample(data, sample_size)

    return sampled_data

if __name__ == '__main__':

    # 读取JSON文件
    with open('train.json', 'r') as file:
        data = json.load(file)


    # 计算条目数
    num_entries = len(data)
    print("数据条目数:", num_entries)
    # # 调用函数输出数据
    # read_json_data(data)
    sample_size = int(3.5e4)
    sampled_data = sample_json_data(data, sample_size)
    # print(len(sampled_data))
    # read_json_data(sampled_data)
    with open('sampled_train.json', 'w') as f:
        json.dump(sampled_data, f, indent=4, ensure_ascii=False)