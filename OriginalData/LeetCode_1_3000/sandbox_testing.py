import os
import json
import subprocess
import time
import tempfile
import sys
import textwrap
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import re
python_exec = sys.executable

# ===============================
# 公共辅助类
# ===============================
LEETCODE_COMMON = textwrap.dedent("""
import random, functools, collections, string, math, datetime
from typing import *
from itertools import *
from heapq import *
from bisect import *
from operator import *
from collections import deque, defaultdict, Counter

inf = float('inf')

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values: list):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head

def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    elif not p or not q:
        return False
    elif p.val != q.val:
        return False
    else:
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
""")

# ===============================
# 沙箱执行函数（Windows-safe）
# ===============================
def run_solution_sandbox(code: str, entry_point: str, check_func: str, timeout_sec=10, mem_limit_mb=2048):
    py_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
            py_path = f.name
            f.write(LEETCODE_COMMON)
            f.write("\n")
            f.write(code)
            f.write("\n")
            f.write(f"def candidate(**kwargs):\n    return {entry_point}(**kwargs)\n\n")
            f.write(check_func)
            f.write("\nif __name__ == '__main__':\n    check(candidate)\n")

        
        proc = subprocess.Popen(
            [python_exec, py_path],
            # 避免 PIPE 堵塞
            # stdout=subprocess.PIPE -> stdout=subprocess.DEVNULL
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        max_memory = 0
        process = psutil.Process(proc.pid)
        start_time = time.perf_counter()
        # 循环监控进程状态与内存使用
        while True:
            if proc.poll() is not None:
                break
            try:
                mem = process.memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, mem)
                if mem > mem_limit_mb:
                    proc.kill()
                    return {"time": float("inf"), "memory": float("inf"), "error": f"Memory limit exceeded {mem:.4f}MB"}
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            elapsed = time.perf_counter() - start_time
            if elapsed < 1:
                time.sleep(0.001)
            elif elapsed < 2:
                time.sleep(0.005)
            else:
                time.sleep(0.01)

            if time.perf_counter() - start_time > timeout_sec:
                proc.kill()
                return {"time": float("inf"), "memory": float("inf"), "error": "Timeout"}

        elapsed = time.perf_counter() - start_time
        stdout, stderr = proc.communicate()

        return {"time": elapsed, "memory": max_memory, "error": stderr.strip()}
    except Exception as e:
        return {"time": float("inf"), "memory": float("inf"), "error": str(e)}
    finally:
        if py_path and os.path.exists(py_path):
            try:
                os.remove(py_path)
            except:
                pass

# ===============================
# 可序列化评估函数
# ===============================
def _evaluate_single_solution(args):
    code, entry_point, check_func, nums_runs = args
    total_time, total_mem, count = 0, 0, 0
    for _ in range(nums_runs):
        res = run_solution_sandbox(code=code, entry_point=entry_point, check_func=check_func)
        if res["time"] < float("inf") and res["memory"] < float("inf"):
            total_time += res["time"]
            total_mem += res["memory"]
            count += 1
            
    if count > 0:
        return total_time / count, total_mem / count
    else:
        return float("inf"), float("inf")

# ===============================
# 多解法评估（支持并行 + tqdm）
# ===============================
def evaluate_problem(problem_idx, entry_point, check_func, solutions, nums_runs=5, parallel=True):
    results = []
    task_args = [(code, entry_point, check_func, nums_runs) for code in solutions]
    MAX_PARALLEL_PROCESSES = os.cpu_count()

    if parallel and len(solutions) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(len(solutions), MAX_PARALLEL_PROCESSES)) as ex:
                futures = {ex.submit(_evaluate_single_solution, arg): i for i, arg in enumerate(task_args, 1)}
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Problem {problem_idx}"):
                    idx = futures[fut]
                    try:
                        avg_time, avg_mem = fut.result(timeout=10)
                    except Exception as e:
                        avg_time, avg_mem = float("inf"), float("inf")
                    if avg_time < float("inf") and avg_mem < float("inf"):
                        results.append({
                            "problem_idx": problem_idx,
                            "solution_id": idx,
                            "avg_time": avg_time,
                            "avg_memory": avg_mem
                        })
        except Exception as e:
            print(f"[Warning] Parallel execution failed ({e}), switching to sequential.")
            parallel = False

    if not parallel:
        for idx, arg in enumerate(task_args, 1):
            avg_time, avg_mem = _evaluate_single_solution(arg)
            if avg_time < float("inf") and avg_mem < float("inf"):
                results.append({
                    "problem_idx": problem_idx,
                    "solution_id": idx,
                    "avg_time": avg_time,
                    "avg_memory": avg_mem
                })

    # 归一化计算分数
    valid = [r for r in results if r["avg_time"] < float("inf") and r["avg_memory"] < float("inf")]
    if valid:
        max_t = max(r["avg_time"] for r in valid)
        max_m = max(r["avg_memory"] for r in valid)
        min_t = min(r["avg_time"] for r in valid)
        min_m = min(r["avg_memory"] for r in valid)

        for r in results:
            if r["avg_time"] < float("inf"):
                t_norm = (r["avg_time"] - min_t) / (max_t - min_t)
                m_norm = (r["avg_memory"] - min_m) / (max_m - min_m)
                r["score"] = 0.5 * t_norm + 0.5 * m_norm
            else:
                r["score"] = float("inf")
    else:
        for r in results:
            r["score"] = float("inf")

    return results

# ===============================
# 主程序
# ===============================
def main(start_idx=0, end_idx=None, testing_path=None, solutions_path=None, output_file=None):
    """读取测试数据与解法数据 [start_idx, end_idx) """

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            old_results = json.load(f)
    else:
        old_results = []

    if not os.path.exists(testing_path) or not os.path.exists(solutions_path):
        print("The required JSON files are missing. Please ensure they are in place.")
        sys.exit(1)

    with open(testing_path, "r", encoding="utf-8") as f:
        testing_data = [x for x in json.load(f) if "problem_idx" in x]
    with open(solutions_path, "r", encoding="utf-8") as f:
        solutions_data = [x for x in json.load(f) if "problem_idx" in x]

    # 构建解法映射字典
    sol_map = {item["problem_idx"]: item for item in solutions_data}

    all_results = []

    # 批量运行所有题目
    for i, test_item in enumerate(tqdm(testing_data[start_idx: end_idx], desc="Evaluating problems")):
        pid = test_item["problem_idx"]
        entry = test_item["test_entry"]
        check = test_item["test"]
        sols_item = sol_map.get(pid)
        if not sols_item:
            continue

        sols = [sols_item[k] for k in sols_item if k.startswith("completion")]
        res = evaluate_problem(pid, entry, check, sols, parallel=True)
        all_results.extend(sorted(res, key=lambda x: x["score"]))

    # 合并新旧结果
    merged_results = old_results + all_results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Before adding, there are {len(old_results)} solutions. We add {len(all_results)} problems solutions.")
    print(f"Total problems evaluated: {len(merged_results)}")

if __name__ == "__main__":
    OUTPUT_IDX = 1
    start_idx = 0
    end_idx = 100

    testing_path = rf"OriginalData\LeetCode_1_3000\question_testing.json"
    solutions_path = rf"OriginalData\LeetCode_1_3000\question_completion.json"
    output_file = rf"OriginalData\LeetCode_1_3000\evaluation_results\part_{OUTPUT_IDX}.json"

    # 上一次运行的
    main(start_idx, end_idx, testing_path, solutions_path, output_file) # 修改这里以评估不同范围的问题ID