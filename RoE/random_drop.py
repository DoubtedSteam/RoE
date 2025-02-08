import json
from tqdm import tqdm

with open('llava_v1_5_mix665k.json', 'r') as f:
    source_datas = json.load(f)

target_datas = []

for idx, data in enumerate(source_datas):
    if idx % 10 < 6:
        target_datas.append(data)
        
print(len(target_datas))

with open('llava_v1_5_mix665k_50.json', 'w') as f:
    json.dump(target_datas, f, indent=4)
