import json
import matplotlib.pyplot as plt

input_file = rf"OriginalData\LeetCode_1_3000\question_information.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

dict = {}
i = 0
for item in data:
    if "algorithms" not in item:
        continue

    algorithms = item["algorithms"]
    for algorithm in algorithms:
        if algorithm not in dict:
            dict[f"{algorithm}"] = 1
        else:
            dict[f"{algorithm}"] += 1

# 设置窗口大小
plt.figure(figsize=(20, 10))
plt.bar(dict.keys(), dict.values())
plt.xlabel("Algorithm Types")
plt.ylabel("Number of Algoritm")
plt.title("Distribution of Algorithm Types")

plt.show()