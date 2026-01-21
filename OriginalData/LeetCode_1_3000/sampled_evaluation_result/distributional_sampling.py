"""
改进的分布式采样脚本（样式与 roulette_wheel_sample.py 类似）
目标：
  - 针对每个 problem，从已清洗的评测结果中采样若干解法（自适应数量），
  - 强制包含极端（最好/最差）样本，同时按四分位分配其余样本，
  - 构建对称的高效-低效代码对，顺序从中等差距到最大差距，便于课程式呈现。

输出 JSON：每个元素为一对 sample，包括：
  problem_idx, pair_type, left_solution_id, right_solution_id,
  left_score, right_score, left_time, right_time, left_memory, right_memory, score_gap
"""
from typing import List, Dict, Any, Tuple
import os
import json
import math
import random
from collections import defaultdict
from pprint import pprint

# -------------------------
# 工具函数（与 roulette_wheel_sample.py 风格一致）
# -------------------------
def clip(M: int, Min: int, Max: int) -> int:
    """Clip 操作：确保采样数 M 在 [Min, Max] 内"""
    # Pairs = clip(round(sqrt(M)), min_pairs, max_pairs)
    # N = 2 * Pairs = 2 * clip(round(sqrt(M)), min_pairs, max_pairs)
    return int(max(Min, min(Max, M)))

def quantile_index(n: int, q: float) -> int:
    """Return integer index in [0, n-1] that approximates q quantile."""
    if n <= 0:
        return 0
    pos = (n - 1) * q
    return int(math.floor(pos))

def quantile_indices_equidistant(length: int, k: int) -> List[int]:
    """
    Return k equidistant indices in [0, length-1], deterministic.
    Ensures strictly increasing indices.
    """
    if k <= 0:
        return []
    if k >= length:
        return list(range(length))
    indices = []
    last = -1
    for i in range(1, k + 1):
        pos = (i / (k + 1)) * length
        idx = int(math.floor(pos))
        if idx <= last:
            idx = last + 1
            if idx >= length:
                idx = length - 1
        indices.append(idx)
        last = idx
    return indices

def pick_from_bucket(bucket: List[Dict[str,Any]], count: int, deterministic: bool=True, rng: random.Random=None) -> List[Dict[str,Any]]:
    """Pick `count` items from bucket deterministically (equidistant) or randomly (without replacement)."""
    if count <= 0 or not bucket:
        return []
    L = len(bucket)
    if count >= L:
        return list(bucket)
    if deterministic:
        idxs = quantile_indices_equidistant(L, count)
        return [bucket[i] for i in idxs]
    else:
        if rng is None:
            rng = random.Random()
        return rng.sample(bucket, count)

# -------------------------
# 核心采样函数
# -------------------------
def sample_pairs_per_problem(
    solutions: List[Dict[str,Any]],
    min_pairs: int = 2,
    max_pairs: int = 12,
    force_extreme_k: int = 1,
    deterministic: bool = True,
    rng_seed: int = 0,
    rng_score: int = 0.5,
    c: int = 1
) -> List[Dict[str,Any]]:
    """
    For a single problem (solutions list), produce up to P pairs according to:
      - P = clip(round(sqrt(m)), min_pairs, max_pairs)
      - N = 2*P samples (even); if N > m then set N = m if m even else m-1
      - Always include top force_extreme_k best and worst solutions (if available)
      - Remaining samples allocated to quartiles (Q1..Q4) in proportions favoring middle quartiles
      - Build symmetric pairs from sampled list sorted best->worst:
            pairs: (N/2 -1, N-1), (N/2 -2, N-2), ... -> yields sequence from middle-gap to max-gap
    Returns list of pair dicts (annotated).
    """
    if not solutions:
        return []

    m = len(solutions)
    if m < 2:
        return []

    # choose sorting key (score preferred; lower is better)
    sols_sorted = sorted(solutions, key=lambda x: x.get('score', float('inf')))
    # if the gap is too small, we don't have enough variation to sample
    s_max = max([x.get('score', float('inf')) for x in sols_sorted])
    s_min = min([x.get('score', float('inf')) for x in sols_sorted])
    if s_max - s_min < rng_score:
        return []

    # compute P and N
    raw_p = int(round(math.sqrt(m)))
    if len(sols_sorted) < 10:
        c = 1.75
    elif len(sols_sorted) < 20:
        c = 1.5
    elif len(sols_sorted) < 50:
        c = 1.25
    else:
        c = 1.0

    P = clip(c * raw_p, min_pairs, max_pairs)
    if P < 1:
        P = 1
    N = 2 * P
    # don't ask more samples than available; ensure N even
    if N > m:
        N = m if m % 2 == 0 else m - 1
        if N < 2:
            return []

    # rng
    rng = random.Random(rng_seed)

    # force include extremes
    k = max(0, min(force_extreme_k, N // 2))  # cannot exceed half sampled
    sampled = []

    # include best k
    if k > 0:
        sampled.extend(sols_sorted[:k])
    # include worst k
    if k > 0:
        sampled.extend(sols_sorted[-k:])

    remaining_to_pick = N - len(sampled)
    # if remaining <=0 then we already have all needed (rare)
    if remaining_to_pick <= 0:
        # ensure unique and maintain order best->worst
        unique_by_id = {}
        for it in sampled:
            sid = it.get('solution_id')
            if sid not in unique_by_id:
                unique_by_id[sid] = it
        sampled = sorted(unique_by_id.values(), key=lambda x: x.get('score', float('inf')))
    else:
        # split into quartiles
        q1_idx = quantile_index(m, 0.25)
        q2_idx = quantile_index(m, 0.50)
        q3_idx = quantile_index(m, 0.75)
        # ensure indices increasing and non-empty chunks
        # define ranges: [0:q1_idx+1), [q1_idx+1:q2_idx+1), [q2_idx+1:q3_idx+1), [q3_idx+1:m)
        q1 = sols_sorted[0:q1_idx+1]
        q2 = sols_sorted[q1_idx+1:q2_idx+1]
        q3 = sols_sorted[q2_idx+1:q3_idx+1]
        q4 = sols_sorted[q3_idx+1:m]

        quartiles = [q1, q2, q3, q4]

        # allocation strategy for remaining_to_pick:
        # favor middle quartiles: weights = [1, 3, 3, 1] normalized
        base_weights = [1, 3, 3, 1]
        total_w = sum(base_weights)
        raw_alloc = [max(0, int(math.floor(remaining_to_pick * w / total_w))) for w in base_weights]
        allocated = sum(raw_alloc)
        # distribute leftover slots to middle quartiles preference Q2 then Q3 then Q1 then Q4
        leftover = remaining_to_pick - allocated
        order_pref = [1, 2, 0, 3]
        idx = 0
        while leftover > 0:
            raw_alloc[order_pref[idx % 4]] += 1
            idx += 1
            leftover -= 1

        # now for each quartile, pick raw_alloc[i] items deterministically/randomly
        for i in range(4):
            cnt = raw_alloc[i]
            bucket = quartiles[i]
            if not bucket or cnt <= 0:
                continue
            picks = pick_from_bucket(bucket, cnt, deterministic=deterministic, rng=rng)
            # avoid duplicates with already sampled extremes
            # use IDs to compare
            existing_ids = set(it['solution_id'] for it in sampled)
            picks = [p for p in picks if p['solution_id'] not in existing_ids]
            # if picks fewer than desired due to duplicates, try to borrow from neighboring buckets
            if len(picks) < cnt:
                need = cnt - len(picks)
                # neighbor order depending on i (prefer nearest)
                neighbors = []
                if i == 0:
                    neighbors = [1,2,3]
                elif i == 1:
                    neighbors = [2,0,3]
                elif i == 2:
                    neighbors = [1,3,0]
                else:
                    neighbors = [2,1,0]
                for nb in neighbors:
                    if need <= 0:
                        break
                    # candidates in nb not already in sampled and not already picked
                    cand = [c for c in quartiles[nb] if c['solution_id'] not in existing_ids and c not in picks]
                    take = pick_from_bucket(cand, need, deterministic=deterministic, rng=rng)
                    picks.extend(take)
                    for t in take:
                        existing_ids.add(t['solution_id'])
                    need = cnt - len(picks)
            sampled.extend(picks)

        # if still short (rare), fill from global best/worst pool
        if len(sampled) < N:
            existing_ids = set(it['solution_id'] for it in sampled)
            for item in sols_sorted:
                if item['solution_id'] in existing_ids:
                    continue
                sampled.append(item)
                existing_ids.add(item['solution_id'])
                if len(sampled) >= N:
                    break

    # final sampled list: ensure unique and exactly N items
    # maintain order best->worst by score
    unique = {}
    for it in sampled:
        unique[it['solution_id']] = it
    sampled_unique = sorted(list(unique.values()), key=lambda x: x.get('score'))
    if len(sampled_unique) > N:
        sampled_unique = sampled_unique[:N]
    # if less (shouldn't happen) pad
    if len(sampled_unique) < N:
        for it in sols_sorted:
            if it['solution_id'] not in {s['solution_id'] for s in sampled_unique}:
                sampled_unique.append(it)
            if len(sampled_unique) >= N:
                break

    # now build symmetric pairs:
    # indices 0..N-1; pairs: (N/2 -1 - k, N-1 - k) for k=0..P-1
    pairs = []
    half = N // 2
    L = len(sampled_unique) 

    for k in range(P):
        left_idx = half - 1 - k
        right_idx = L - 1 - k
        # 越界保护，不改变原算法的 pair-matching 逻辑
        if left_idx < 0 or right_idx < 0 or left_idx >= right_idx:
            continue
        if left_idx >= L or right_idx >= L:
            continue

        left = sampled_unique[left_idx]
        right = sampled_unique[right_idx]
        
        gap = None
        try:
            left_score = float(left.get('score', left.get('avg_time', 0.0)))
            right_score = float(right.get('score', right.get('avg_time', 0.0)))
            gap = right_score - left_score
        except Exception:
            gap = None
        pairs.append({
            'problem_idx': left.get('problem_idx'),
            'effi_solution_id': left.get('solution_id'),
            'ineffi_solution_id': right.get('solution_id'),
            'effi_score': left.get('score'),
            'ineffi_score': right.get('score'),
            'effi_time': left.get('avg_time'),
            'ineffi_time': right.get('avg_time'),
            'effi_memory': left.get('avg_memory'),
            'ineffi_memory': right.get('avg_memory'),
            'score_gap': gap
        })
    pairs = sorted(pairs, key=lambda x: x.get('effi_score', float('inf')), reverse=True)
    return pairs

# -------------------------
# 批量处理函数（与 roulette 风格类似）
# -------------------------
def sample_all_problems_from_file(
    input_file: str,
    output_file: str,
    min_pairs:int = 2,
    max_pairs:int = 12,
    force_extreme_k:int = 1,
    deterministic: bool = True,
    rng_seed: int = 0,
    rng_score: int = 0.5,
    global_limit: int = None, 
    c: int = 1
) -> List[Dict[str,Any]]:
    """
    Read input JSON (list of solution dicts), group by problem_idx and sample pairs for each problem.
    Writes the combined pairs list to output_file and returns it.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # group by problem
    problems = defaultdict(list)
    for it in data:
        if 'problem_idx' not in it or 'solution_id' not in it:
            continue
        problems[it['problem_idx']].append(it)

    all_pairs = []
    # sort problems by decreasing m so that big problems sampled first (optional)
    for pid, sols in sorted(problems.items(), key=lambda x: -len(x[1])):
        pairs = sample_pairs_per_problem(
            solutions=sols,
            min_pairs=min_pairs,
            max_pairs=max_pairs,
            force_extreme_k=force_extreme_k,
            deterministic=deterministic,
            rng_seed=rng_seed,
            rng_score=rng_score,
            c=c
        )
        all_pairs.extend(pairs)
        if global_limit and len(all_pairs) >= global_limit:
            all_pairs = all_pairs[:global_limit]
            break

    # save
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    return all_pairs

# -------------------------
# 命令行示例/入口
# -------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Distributional sampling improved (quartile + forced extremes)')
    parser.add_argument('--min_pairs', type=int, default=2, help='minimum pairs per problem')
    parser.add_argument('--max_pairs', type=int, default=50, help='maximum pairs per problem')
    parser.add_argument('--force_extreme_k', type=int, default=1, help='force include top-k best and worst solutions if available')
    parser.add_argument('--deterministic', action='store_true', default=True, help='deterministic sampling (True) or random')
    parser.add_argument('--rng_seed', type=int, default=42)
    parser.add_argument('--rng_score', type=float, default=0.5, help='minimum score gap to sample')
    parser.add_argument('--global_limit', type=int, default=None, help='global cap on number of pairs')
    parser.add_argument('--c', type=int, default=1.25, help='c-fold sampling')
    args = parser.parse_args()
    total = 0

    for num in range(1, 14 + 1):
        input_path = rf"OriginalData\LeetCode_1_3000\filtered_evaluation_results\part_{num}.json"
        output_path = rf"OriginalData\LeetCode_1_3000\sampled_evaluation_result\distributional_sample_part_{num}.json"
        pairs = sample_all_problems_from_file(
            input_file=input_path,
            output_file=output_path,
            min_pairs=args.min_pairs,
            max_pairs=args.max_pairs,
            force_extreme_k=args.force_extreme_k,
            deterministic=args.deterministic,
            rng_seed=args.rng_seed,
            rng_score=args.rng_score,
            global_limit=args.global_limit, 
            c=args.c
        )
        print(f"Saved {len(pairs)} pairs (min_pairs={args.min_pairs}, max_pairs={args.max_pairs}) to {output_path}")
        total += len(pairs)

    print(f"Total pairs: {total}")
