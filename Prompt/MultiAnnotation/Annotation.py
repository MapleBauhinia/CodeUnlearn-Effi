import json
import os
import re
import glob
import time
import requests
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------
# Annotation.py: 第一次进行数据标注
# -------------------------------------

# === API 配置（保持一致，但采用 Annotation.py 的调用方式） ===
BASE_URL = "https://yunwu.zeabur.app/v1"
API_KEY = "..."
MODEL = "claude-sonnet-4-5-20250929"
MAX_RETRY = 3
API_DELAY_SEC = 1.0

# === 数据路径配置 ===
INFO_PATH = "OriginalData\LeetCode_1_3000\question_information.json"
COMPLETED_PATH = "Prompt/MultiAnnotation/CompletedTasks.json"
SOLUTIONS_PATH = "OriginalData\LeetCode_1_3000\question_completion.json"
SAMPLED_SOLUTIONS_PATH = "OriginalData\LeetCode_1_3000\sampled_evaluation_result\distributional_sample_TOTAL.json"
PROMPT_PATH = "Prompt/MultiAnnotation/optimized_prompt.txt"
OUTPUT_JSON_PATH = "Prompt/MultiAnnotation/AI_results(15).json"

N = 2   # 每个提示词包含的 pair 数量
LIMIT = 15  # 只处理多少道题目


def extract_leading_number(filename: str) -> int:
    m = re.match(r"(\d+)", filename)
    return int(m.group(1)) if m else float("inf")

# =====================================
# 替换后的高速 annotate_text（核心改动）
# =====================================
def annotate_text(prompt):
    """
    使用 requests + yunwu.zeabur.app API 调用，替换 OpenAI SDK 调用
    保留 retry、max_tokens、messages 等全部逻辑
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a proficient code reviewer and annotator. Please complete the code annotation task described in this prompt."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": 4500 * N,
    }

    for attempt in range(MAX_RETRY):
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            data = response.json()

            # 正常返回的内容格式
            if "choices" not in data:
                raise ValueError(f"Invalid API response (missing choices): {data}")
            result = data["choices"][0]["message"]["content"].strip()
            return result

        except Exception as e:
            print(f"[WARNING] Attempt {attempt + 1}/{MAX_RETRY} failed: {e}")
            if attempt < MAX_RETRY - 1:
                time.sleep(3)
            else:
                print("[ERROR] API call failed after retries.")
                return None


def parse_model_output(raw_output, txt_number):
    # 解析模型输出，去除 Markdown 包裹并提取 JSON 对象
    code_block_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    matches = code_block_pattern.findall(raw_output)

    json_objects = []
    current_index = txt_number
    for idx, block in enumerate(matches, start=1):
        block = block.strip()
        if not block:
            continue
        try:
            obj = json.loads(block)
            obj["pair_idx"] = current_index + 1
            current_index += 1
            json_objects.append(obj)

        except json.JSONDecodeError as e:
            # 打印错误，但避免 UnicodeEncodeError
            print(f"[Warning] The {idx} Json block is invalid: {e}")
            safe_block = (
                f"--- Invalid Block Start ---\n{block}\n--- Invalid Block End ---"
            )
            # 防止 Windows 控制台编码报错
            print(safe_block.encode("utf-8", "ignore").decode("utf-8", "ignore"))
            continue

    return json_objects, current_index
 

def safe_replace(lines, old, new):
    """安全地替换列表中的占位符"""
    replacement = str(new) if new is not None else ""
    return [line.replace(old, replacement) for line in lines]

def main():
    """
    主执行函数: 加载数据、生成提示词、调用API、处理结果、保存已经处理的题目。
    """
    # 加载所有数据
    print("Loading data files...")
    try:
        with open(INFO_PATH, "r", encoding="utf-8") as f:
            info_data = json.load(f)

        with open(COMPLETED_PATH, "r", encoding="utf-8") as f:
            completed_tasks = json.load(f)
        s = set()
        for id in completed_tasks:
            s.add(id)

        with open(SOLUTIONS_PATH, "r", encoding="utf-8") as f:
            solutions_data = json.load(f)

        with open(SAMPLED_SOLUTIONS_PATH, "r", encoding="utf-8") as f:
            sampling_data = json.load(f)

        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            base_prompt_lines = f.readlines()

    except FileNotFoundError as e:
        print(f"[CRITICAL ERROR] Required data file not found: {e}. Exiting.")
        return
    except json.JSONDecodeError as e:
        print(f"[CRITICAL ERROR] Failed to parse JSON data file: {e}. Exiting.")
        return
    
    print("Data loaded successfully.")

    # 构建数据索引
    info_map = {item.get('problem_idx'): item for item in info_data}
    solution_map = {item.get('problem_idx'): item for item in solutions_data}
    
    grouped_pairs = defaultdict(list)
    for item in sampling_data:
        grouped_pairs[item["problem_idx"]].append(item)

    # 开始对每一道题目进行处理，从输出提示词开始
    count = 0
    for problem_idx, all_pairs in grouped_pairs.items():
        # 跳过已经完成的题目
        if problem_idx in s:
            # print(f"[Skipping] Problem {problem_idx} has already been processed. Skipping.")
            continue
        
        print(f"\n[Processing] Starting prompt generation for Problem {problem_idx}...")
        PROMPT_DIR = f"Prompt/MultiAnnotation/problem_{problem_idx}"
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
        prompt_text_lines = safe_replace(prompt_text_lines, "<programming language, e.g., Python, Java, C++>", "Python")
        prompt_text_lines = safe_replace(prompt_text_lines, "<the beginning of solution code or null>", info["prompt"])
        prompt_text = "".join(prompt_text_lines)

        if not os.path.exists(PROMPT_DIR):
            os.makedirs(PROMPT_DIR)

        with open(f"{PROMPT_DIR}/base_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt_text)

        # 按 N 的大小遍历所有代码对 (all_pairs)
        for i in range(0, total_pairs_for_problem, N):
            batch_pairs = all_pairs[i : i + N]
            start_index = i + 1
            end_index = i + len(batch_pairs)
            
            print(f"[Processing] Generate The Prompt of Problem {problem_idx}, pairs {start_index}-{end_index}")

            # 构造多个 (低效-高效) 代码对的文本
            pairs_text = ""
            ID = 0
            for item in batch_pairs:
                ID += 1
                ineffi_id = item["ineffi_solution_id"]
                effi_id = item["effi_solution_id"]

                ineffi_sol = solutions.get(f"completion{ineffi_id}")
                effi_sol = solutions.get(f"completion{effi_id}")

                if ineffi_sol is None or effi_sol is None:
                    # 跳过这个无效的 pair，但继续处理批次中的其他 pair
                    print(f"[Warning] Missing solution code for pair ({ineffi_id}, {effi_id}) in problem {problem_idx}. Skipping this pair.")
                    continue 

                ineffi_time = f"{round(item['ineffi_time'], 5)}s"
                effi_time = f"{round(item['effi_time'], 5)}s"

                if item["ineffi_memory"] > 0.0:
                    ineffi_memory = f"{round(item['ineffi_memory'], 2)}MB"
                else:
                    ineffi_memory = "The solution was too fast to measure."

                if item["effi_memory"] > 0.0:
                    effi_memory = f"{round(item['effi_memory'], 2)}MB"
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

            # 将pairs_text替代原有的"=== Pair (k) ==="
            final_prompt_text = prompt_text.replace("=== Pair (k) ===", pairs_text, 1)

            with open(f"{PROMPT_DIR}/{start_index}-{end_index}.txt", "w", encoding="utf-8") as f:
                f.write(final_prompt_text)
            
            print(f"[Succeed] Generated prompt for problem {problem_idx}, pairs {start_index}-{end_index}\n")

        print(f"[Processing] Starting annotation from problem_idx: {problem_idx}")
        # 读取所有待处理的 .txt 文件
        # 按照最后修改时间排序(1-3, 4-6, ...)
        files = glob.glob(os.path.join(PROMPT_DIR, "*.txt"))

        # sort by the leading number in filename
        files = sorted(files, key=lambda p: extract_leading_number(os.path.basename(p)))
        
        if not files:
            print(f"No .txt file is found in {PROMPT_DIR}")
            return
        
        if os.path.exists(OUTPUT_JSON_PATH):
            with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        else:
            os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
            all_results = []

        txt_number = 0
        for filepath in files:
            filename = os.path.basename(filepath)
            # filename is "base_prompt.txt", skip it
            if filename.startswith("base_prompt"):
                continue
            
            print(f"\n[Processing] {filename}")
            text = open(filepath, "r", encoding="utf-8").read().strip()
            if not text:
                print("Skipped empty file.")
                continue
            
            raw_output = annotate_text(text)
            if raw_output is None:
                print(f"[Failed] Skip {filename}")
                continue
            

            # 尝试解析为 JSON
            json_objects, txt_number = parse_model_output(raw_output, txt_number)

            all_results.extend(json_objects)

            print(f"[Succeed] Have annotated {filename}")
            time.sleep(1.5)

        # 更新已经完成的题目列表
        s.add(problem_idx)
        # 清理提示词文件夹
        if os.path.exists(PROMPT_DIR):
            for f in os.listdir(PROMPT_DIR):
                os.remove(os.path.join(PROMPT_DIR, f))
            os.rmdir(PROMPT_DIR)

        # 保存所有结果到 OUTPUT_DIR
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as fout:
            json.dump(all_results, fout, ensure_ascii=False, indent=2)
        print(f"\n[Completed] All results of Problem {problem_idx} have been saved to {OUTPUT_JSON_PATH}.")

        # 保存已经完成的题目列表
        completed_tasks = list(s)
        completed_tasks = sorted(completed_tasks)
        with open(COMPLETED_PATH, "w", encoding="utf-8") as f:
            json.dump(completed_tasks, f, ensure_ascii=False, indent=2)

        count += 1
        if count >= LIMIT:
            break
        
if __name__ == "__main__":
    main()