import json
res = []

for num in range(1, 14 + 1):
    input_path = rf"OriginalData\LeetCode_1_3000\sampled_evaluation_result\distributional_sample_part_{num}.json"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        res.append(item)
    
output_path = rf"OriginalData\LeetCode_1_3000\sampled_evaluation_result\distributional_sample_TOTAL.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print("Done")
