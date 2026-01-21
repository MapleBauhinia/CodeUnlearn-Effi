import json
import os

def filter_time_consumption(input_path, time_threshold=5.0, mem_threshold=2048.0):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = []
    for item in data:
        if "problem_idx" not in item:
            continue

        if item["avg_time"] >= time_threshold or item["avg_memory"] >= mem_threshold:
            problem_idx = item["problem_idx"]
            solution_idx = item["solution_id"]
            avg_time = item["avg_time"]
            avg_memory = item["avg_memory"]

            cur = {
                "problem_idx": problem_idx,
                "solution_id": solution_idx,
                "avg_time": avg_time,
                "avg_memory": avg_memory
            }
            print(f"In problem {problem_idx}, the solution {solution_idx} has avg_time {avg_time} and avg_memory {avg_memory}")
            res.append(cur)

    print("Successfully filtered.")
    return res


def filter_and_rescore(input_file, output_file, time_threshold=5.0, mem_threshold=2048.0, time_rng=0.001, mem_rng=0.1):
    """
    过滤掉 avg_time 超过 time_threshold 的题解，
    并在每个 problem_idx 内重新归一化 score。
    """
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 按题目分组
    problem_ids = set()
    for item in data:
        problem_ids.add(item["problem_idx"])

    new_results = []
    for pid in problem_ids:
        res = []

        # 过滤掉超过时间 time_threshold 和内存限制 mem_threshold 的题解
        items = [x for x in data if x["problem_idx"] == pid and x["avg_time"] < time_threshold and x["avg_memory"] < mem_threshold]

        # 如果没有题解或者只有一个题解，则跳过
        if not items or len(items) == 1:
            continue

        max_t = max(x["avg_time"] for x in items)
        min_t = min(x["avg_time"] for x in items)
        max_m = max(x["avg_memory"] for x in items)
        min_m = min(x["avg_memory"] for x in items)

        if True:
            if max_t - min_t < time_rng:  # 如果变化尺度小于1毫秒，跳过
                continue
            if max_m - min_m < mem_rng:   # 如果变化尺度小于0.1MB，跳过
                continue

        for x in items:
            # 不再沿用原来的 score = 0.5 * (t / max_t + m / max_m)
            # 是根据 avg_time 和 avg_memory 重新计算 (min–max)
            t_norm = (x["avg_time"] - min_t) / (max_t - min_t)
            m_norm = (x["avg_memory"] - min_m) / (max_m - min_m)
            x["score"] = 0.5 * (t_norm + m_norm)
            res.append(x)

        #max_s = max(x["score"] for x in res)
        #min_s = min(x["score"] for x in res)
        #for i, _ in enumerate(res):
        #    res[i]["score"] = (res[i]["score"] - min_s) / (max_s - min_s)

        new_results.extend(sorted(res, key=lambda x: x["score"]))

    
    # 写入新文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_results, f, ensure_ascii=False, indent=2)

    print(f"\nFiltering and rescoring completed.")
    print(f"  Input file:  {input_file}")
    print(f"  Output file: {output_file}")
    print(f"  Filtered solutions (avg_time >= {time_threshold}s and avg_memory >= {mem_threshold}MB): {len(data) - len(new_results)}")
    print(f"  Total solutions after processing: {len(new_results)}\n")
    return len(new_results)



if __name__ == "__main__":
    time_threshold = 4.0
    mem_threshold = 35.0
    total = 0
    NUM = 14
    
    for num in range(1, NUM + 1):
        # path = f"..."
        # print("In part", num, "...")
        # filter_time_consumption(path, time_threshold, mem_threshold)

        input_file = rf"OriginalData\LeetCode_1_3000\evaluation_results\part_{num}.json"
        output_file = rf"OriginalData\LeetCode_1_3000\filtered_evaluation_results\part_{num}.json"
        total += filter_and_rescore(input_file, output_file, time_threshold, mem_threshold)

    print(f"Total solutions: {total}")
    print("Done")

    