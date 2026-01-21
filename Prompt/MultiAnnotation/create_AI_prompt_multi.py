import json
import os
from collections import defaultdict


# 题目的描述信息，如ID、题目名字、问题描述、算法标签等
# Complete the code annotation task described in this prompt.
info_path = "OriginalData\LeetCode_1_3000\question_information.json"
with open(info_path, "r", encoding="utf-8") as f:
    info_data = json.load(f)

# 已经完成的题目ID
COMPLETED_PATH = "OriginalData\LeetCode_1_3000\question_completion.json"
with open(COMPLETED_PATH, "r", encoding="utf-8") as f:
    completed_tasks = json.load(f)
s = set()
for item in completed_tasks:
    s.add(item)

# 解决方案代码，但是不是全部都可以选择
solutions_path = "OriginalData\LeetCode_1_3000\question_completion.json"
with open(solutions_path, "r", encoding="utf-8") as f:
    solutions_data = json.load(f)

# 采样得到的“低效代码-高效代码”对，包含了运行结果信息
sampled_solutions = "OriginalData\LeetCode_1_3000\sampled_evaluation_result\distributional_sample_TOTAL.json"
with open(sampled_solutions, "r", encoding="utf-8") as f:
    sampling_data = json.load(f) 

# 提示词的模板
prompt_path = "Prompt\MultiAnnotation\optimized_prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    base_prompt_lines = f.readlines() # 重命名以避免混淆

# problem_idx -> info item
info_map = {item.get('problem_idx'): item for item in info_data}

# problem_idx -> solutions item
solution_map = {item.get('problem_idx'): item for item in solutions_data}

# Group code pairs by problem_idx
grouped_pairs = defaultdict(list)
for item in sampling_data:
    grouped_pairs[item["problem_idx"]].append(item)

# Set max number of pairs per problem
N = 2

def safe_replace(lines, old, new):
    replacement = str(new) if new is not None else ""
    return [line.replace(old, replacement) for line in lines]

# 题目的数量（并非是代码对的数量）
count = 0

for problem_idx, all_pairs in grouped_pairs.items():
    PROMPT_DIR = f"Prompt/MultiAnnotation/problem_{problem_idx}"

    if problem_idx in s:
        print(f"Skipping completed problem_idx: {problem_idx}")
        if os.path.exists(PROMPT_DIR):
            for f in os.listdir(PROMPT_DIR):
                os.remove(os.path.join(PROMPT_DIR, f))
            os.rmdir(f"Prompt/MultiAnnotation/problem_{problem_idx}")
        continue

    info = info_map.get(problem_idx)
    solutions = solution_map.get(problem_idx)
    if info is None or solutions is None:
        continue

    total_pairs_for_problem = len(all_pairs)
    if total_pairs_for_problem == 0:
        continue

    # 为这个新批次复制基础模板
    prompt_text_lines = base_prompt_lines.copy()
    prompt_text_lines = safe_replace(prompt_text_lines, "<problem_idx>", str(problem_idx))
    prompt_text_lines = safe_replace(prompt_text_lines, "<task name>", info["task_name"])
    prompt_text_lines = safe_replace(prompt_text_lines, "<problem description>", info["description"])
    prompt_text_lines = safe_replace(prompt_text_lines, "<corresponding algorithm labels>", ", ".join(info["algorithms"]))
    prompt_text_lines = safe_replace(prompt_text_lines, "<programming language>", "Python")
    prompt_text_lines = safe_replace(prompt_text_lines, "<the beginning of solution code or null>", info["prompt"])
    prompt_text = "".join(prompt_text_lines) 

    if not os.path.exists(PROMPT_DIR):
        os.makedirs(PROMPT_DIR)

    with open(f"{PROMPT_DIR}/base_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt_text)

    
    # 按 N 的大小遍历所有代码对 (all_pairs)
    for i in range(0, total_pairs_for_problem, N):
        batch_pairs = all_pairs[i : i + N]
        
        # 计算文件名所需的 start 和 end 索引 (1-based)
        start_index = i + 1
        end_index = i + len(batch_pairs)

        # 构造多个 (低效-高效) 对的文本
        pairs_text = ""
        ID = 0
        for idx, item in enumerate(batch_pairs, start=1):
            ID += 1
            ineffi_id = item["ineffi_solution_id"]
            effi_id = item["effi_solution_id"]

            ineffi_sol = solutions.get(f"completion{ineffi_id}")
            effi_sol = solutions.get(f"completion{effi_id}")

            ineffi_time = str(round(item["ineffi_time"], 5)) + "s"
            effi_time = str(round(item["effi_time"], 5)) + "s"

            ineffi_memory = item["ineffi_memory"]
            effi_memory = item["effi_memory"]

            if item["ineffi_memory"] > 0.0:
                ineffi_memory = str(round(item["ineffi_memory"], 2)) + "MB"
            else:
                ineffi_memory = "The solution was too fast to measure."

            if  item["effi_memory"] > 0.0:
                effi_memory = str(round((item["effi_memory"]), 2)) + "MB"
            else:
                effi_memory = "The solution was too fast to measure."

            # idx 是批次内的相对索引 (1, 2, 3...)
            pairs_text += f"""
=== Pair {ID} ===
## Inefficient Code ({ID})
```
{ineffi_sol}
```
- Time: {ineffi_time}
- Memory: {ineffi_memory}

## Efficient Replacement ({ID})
```
{effi_sol}
```
- Time: {effi_time}
- Memory: {effi_memory}
"""     
        # 去除开头、末尾的多余换行符
        pairs_text = pairs_text.lstrip()
        pairs_text = pairs_text.rstrip()
        pairs_text = "".join(pairs_text)

        # 将pairs_text替代原有的"=== Pair (k) ==="
        final_prompt_text = prompt_text.replace("=== Pair (k) ===", pairs_text)

        with open(f"{PROMPT_DIR}/{start_index}-{end_index}.txt", "w", encoding="utf-8") as f:
            f.write(final_prompt_text)

    count += 1
    if count >= 15:
        break

print("Done: Multi-pair prompts generated.")