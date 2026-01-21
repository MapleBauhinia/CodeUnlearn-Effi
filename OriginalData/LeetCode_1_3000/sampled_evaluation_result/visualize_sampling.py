import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json
import numpy as np
import sys

def normalize_ranking(ranking: list[int]):
    # [实际的solution_id-排名]
    dict = {}
    i = 1
    for num in ranking:
        dict[num] = i
        i += 1
    return dict
    
def visualize_pairs_on_bar(N, pairs, dict=None, figsize=(15,4), cmap_name='tab20'):
    """
    可视化一个长度为 N 的序列，并高亮显示 M 个区间。
    参数：
        N : int
            整个序列长度
        intervals : list of tuple
            每个元组为 (start, end)
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Main green bar (light fill, dark edge)
    bar_y = 0.4
    bar_h = 0.2
    ax.add_patch(patches.Rectangle((0, bar_y), N, bar_h, color='lightgreen', ec='green', zorder=0))
    
    # 随机生成不同的颜色
    colors = [(random.random(), random.random(), random.random()) for _ in range(200)]

    # Plot each pair: two markers (slightly vertically offset so both are visible)
    marker_y_offsets = (bar_y + bar_h/2)  # center of the bar
    x_vals_out_of_range = []

    for idx, [a, b] in enumerate(pairs):
        if dict is not None:
            a1, b1 = a, b
            a, b = dict[a], dict[b]
        else:
            a1, b1 = a, b

        color = colors[idx]
        # Validate/clip positions
        for val in [a, b]:
            if val < 0 or val > N:
                x_vals_out_of_range.append((idx, val))

        # Draw markers: place one slightly above and one slightly below the bar center so they don't overlap.
        ax.plot([a], [marker_y_offsets + 0.06], marker='o', markersize=10, markeredgecolor='black', markerfacecolor=color, zorder=2)
        ax.plot([b], [marker_y_offsets - 0.06], marker='s', markersize=10, markeredgecolor='black', markerfacecolor=color, zorder=2)
        # Label the pair above the markers (centered between them)

        # 颜色块说明：颜色块-id1-id2
        """
        mid = (a + b) / 2.0
        if dict is None:
            ax.text(mid, marker_y_offsets + 0.18, f"{a},{b}", ha='center', va='bottom', fontsize=9, zorder=3)
        else:
            ax.text(mid, marker_y_offsets + 0.18, f"{a1},{b1}", ha='center', va='bottom', fontsize=9, zorder=3)
        """
        legend_x = N + 2
        legend_y = 0.85 - idx * 0.06
        ax.add_patch(patches.Rectangle((legend_x, legend_y), 0.6, 0.04, color=color, ec='black'))
        ax.text(legend_x + 0.8, legend_y + 0.02, f"{a1}, {b1}", va='center', fontsize=9, color='black')
        # Add small legend-like square on the right to indicate pair index/color
        # ax.add_patch(patches.Rectangle((N + 0.5 + idx*0.6, bar_y), 0.4, bar_h, color=color, ec='black', zorder=1))
        # ax.text(N + 0.5 + idx*0.6 + 0.2, bar_y + bar_h/2, f"#{idx}", ha='center', va='center', fontsize=8, color='white' if np.mean(color[:3])<0.6 else 'black')
    
    # Axis formatting
    ax.set_xlim(-1, N + 0.5 + len(pairs)*0.6 + 0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Index Position", fontsize=12)
    ax.set_title(f"Green bar length = {N}; {len(pairs)} pairs marked (each pair uses one color)", fontsize=13)
    ax.set_xticks(range(0, N+1, max(1, N//10)))
    ax.grid(False)
    plt.tight_layout()
    
    # Warnings for out-of-range positions
    if x_vals_out_of_range:
        print("Warning: some positions are outside the range [0, N]. They are still plotted at their numeric positions.")
        for idx, val in x_vals_out_of_range:
            print(f" - pair index {idx} contains value {val} (outside [0, {N}])")
    
    plt.show()

# 示例
N = 50
#visualize_pairs_on_bar(N, intervals, None)

if __name__ == "__main__":
    problem_idx = 290
    input_file = rf"OriginalData\LeetCode_1_3000\filtered_evaluation_results\part_2.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ranking = [x["solution_id"] for x in data if x["problem_idx"] == problem_idx]
    if not ranking:
        print("No solution found for problem", problem_idx)
        sys.exit(1)

    # mapping: <实际的solution_id-排名i>
    mapping = normalize_ranking(ranking)
    N = len(mapping)

    input_file = rf"OriginalData\LeetCode_1_3000\sampled_evaluation_result\distributional_sample_TOTAL.json"
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # pair = <ineffi_id, effi_id>
    pairs = [[x["effi_solution_id"], x["ineffi_solution_id"]] for x in data if x["problem_idx"] == problem_idx]

    # ===== 分批处理，每批 batch_size 个 =====
    batch_size = len(pairs)
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    for i in range(total_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(pairs))
        batch_pairs = pairs[start:end]
        print(f"Batch {i + 1}/{total_batches} has {len(batch_pairs)} pairs")
        visualize_pairs_on_bar(N, batch_pairs, mapping)