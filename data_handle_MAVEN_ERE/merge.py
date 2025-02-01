import os
import json
import random


if __name__ == "__main__":
    file_path = os.path.dirname(__file__)
    project_path = os.path.abspath(os.path.join(file_path, ".."))
    print(project_path)
    random.seed(0)

    dataset_name = "MAVEN_ERE"
    origin_data_path = os.path.join(project_path, f"data/converted/{dataset_name}")
    merge_data_path = os.path.join(project_path, f"data/converted/{dataset_name}/full")

    if not os.path.exists(merge_data_path):
        os.makedirs(merge_data_path)

    with open(os.path.join(origin_data_path, f"temporal/train.json")) as f:
        data_temporal = json.load(f)
    with open(os.path.join(origin_data_path, f"causal/train.json")) as f:
        data_causal = json.load(f)
    with open(os.path.join(origin_data_path, f"subevent/train.json")) as f:
        data_subevent = json.load(f)
    with open(os.path.join(origin_data_path, f"coref/train.json")) as f:
        data_coref = json.load(f)

    expansion_ratio = 2
    examples = []
    examples += data_temporal
    examples += data_causal
    examples += data_subevent * expansion_ratio
    examples += data_coref

    random.shuffle(examples)
    with open(os.path.join(merge_data_path, "train.json"), 'w') as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)

    print("finish")
