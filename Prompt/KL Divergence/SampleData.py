import json
import random
import sys
# 使用utf-8编码
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

def load_jsonl(path):
    """逐行读取 jsonl 文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data

def write_jsonl(path, data_list):
    """将列表写入 jsonl 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main(file1, file2, n, output):
    # 读取两个 JSONL 数据集
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    print(f"Loaded {len(data1)} entries from {file1}")
    print(f"Loaded {len(data2)} entries from {file2}")

    # 合并
    merged = data1 + data2
    total = len(merged)

    # 随机打乱
    random.shuffle(merged)

    # 检查 N 是否合理
    if n > total:
        raise ValueError(f"N={n} 大于总数据量 {total}")

    # 随机抽取 N 条
    sampled = random.sample(merged, n)

    # 写入结果文件
    write_jsonl(output, sampled)

    print(f"Finished! Randomly sampled {n} instructions to {output}")


if __name__ == "__main__":
    file1 = rf"Prompt\KL Divergence\HumanEval_SFT.jsonl"
    file2 = rf"Prompt\KL Divergence\MBPP_SFT.jsonl"
    N = 1100
    output = rf"Prompt\KL Divergence\SFT_KL.jsonl"
    # main(file1=file1, file2=file2, n=N, output=output)
    print(-1e-4)
