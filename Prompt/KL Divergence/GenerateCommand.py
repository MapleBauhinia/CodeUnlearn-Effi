import json

input_path = rf"Prompt\KL Divergence\mbpp.jsonl"        # 原始 HumanEval 文件
output_path = rf"Prompt\KL Divergence\MBPP_SFT.jsonl"   # 输出的 SFT 文件

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    
    for line in fin:
        item = json.loads(line)

        prompt = item.get("text", "").strip()
        solution = item.get("code", "").strip()

        # 构造 SFT 数据格式
        sft_item = {
            "instruction": "Write a correct and efficient Python function to solve the following problem. Respond with Python code only.",
            "input": prompt,
            "output": solution
        }

        # 写入 .jsonl（每个样本一行）
        fout.write(json.dumps(sft_item, ensure_ascii=False) + "\n")

print("Done to", output_path)

str = "# Copyright (c) \"Neo4j\"\n# Neo4j Sweden AB [https://neo4j.com]\n#\n# This file is part of Neo4j.\n#\n# Licensed under the Apache License, Version 2.0 (the \"License\");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     https://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an \"AS IS\" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n\n\nfrom ...._spatial import (\n    Point,\n    srid_table,\n)\nfrom ...packstream import Structure\n\n\ndef hydrate_point(srid, *coordinates):\n    \"\"\" Create a new instance of a Point subclass from a raw\n    set of fields. The subclass chosen is determined by the\n    given SRID; a ValueError will be raised if no such\n    subclass can be found.\n    \"\"\"\n    try:\n        point_class, dim = srid_table[srid]\n    except KeyError:\n        point = Point(coordinates)\n        point.srid = srid\n        return point\n    else:\n        if len(coordinates) != dim:\n            raise ValueError(\"SRID %d requires %d coordinates (%d provided)\" % (srid, dim, len(coordinates)))\n        return point_class(coordinates)\n\n\ndef dehydrate_point(value):\n    \"\"\" Dehydrator for Point data.\n\n    :param value:\n    :type value: Point\n    :return:\n    \"\"\"\n    dim = len(value)\n    if dim == 2:\n        return Structure(b\"X\", value.srid, *value)\n    elif dim == 3:\n        return Structure(b\"Y\", value.srid, *value)\n    else:\n        raise ValueError(\"Cannot dehydrate Point with %d dimensions\" % dim)\n\n\n__all__ = [\n    \"hydrate_point\",\n    \"dehydrate_point\",\n]\n"
print(str)