import os
import json

def calculate_skip_ratio():
    # 获取当前目录中的所有jsonl文件
    jsonl_files = [f for f in os.listdir('./skip_result') if f.endswith('.jsonl')]
    skip_count_all = 0
    total_count_all = 0

    # 打开输出文件进行追加写入
    with open('skip_ratio_output.txt', 'a', encoding='utf-8') as output_file:
        # 遍历所有jsonl文件并统计skip比例
        for file in jsonl_files:
            skip_count = 0
            total_count = 0

            with open(os.path.join('./skip_result', file), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()  # 去掉换行符和多余空格
                    if not line:  # 跳过空行
                        continue

                    # 解析JSON并判断内容是否为skip
                    data = json.loads(line)
                    if data == "skip":
                        skip_count += 1
                    total_count += 1

            # 计算并写入每个文件的skip比例
            total_count_all += total_count
            skip_count_all += skip_count
            if total_count > 0:
                skip_ratio = skip_count / total_count
                output_file.write(f"File: {file} skip ratio: {skip_ratio:.2%}\n")
            else:
                output_file.write(f"File: {file} has something wrong\n")

        skip_ratio = skip_count_all / total_count_all if total_count_all > 0 else 0
        output_file.write(f"Overall Skip Ratio: {skip_ratio:.2%}\n")
        output_file.write(f"====================================\n")
        print(f"Overall Skip Ratio: {skip_ratio:.2%}\n")

calculate_skip_ratio()
