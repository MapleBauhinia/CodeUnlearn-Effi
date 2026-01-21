import json
import os

INPUT_PATH = "Prompt/MultiAnnotation/CompletedTasks.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for id in data:
    DIR = f"Prompt/MultiAnnotation/problem_{id}"

    if os.path.exists(DIR):
        for f in os.listdir(DIR):
            os.remove(os.path.join(DIR, f))
        os.rmdir(DIR)

print("Done!")