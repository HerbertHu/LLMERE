import os
import random


if __name__ == "__main__":
    file_dir = os.path.dirname(__file__)
    project_path = os.path.abspath(os.path.join(file_dir, ".."))
    print(project_path)
    random.seed(42)

    split = "train"
    if not os.path.exists(os.path.join(project_path, "data/MAVEN_ERE_split")):
        os.makedirs(os.path.join(project_path, "data/MAVEN_ERE_split"))
    with open(os.path.join(project_path, f"data/MAVEN_ERE/{split}.jsonl")) as f:
        lines = f.readlines()

    split_ratio = 0.8
    random.shuffle(lines)
    split_point = int(len(lines) * split_ratio)
    train_set = lines[:split_point]
    valid_set = lines[split_point:]

    with open(os.path.join(project_path, f"data/MAVEN_ERE_split/train.jsonl"), "w") as f:
        f.writelines(train_set)
    with open(os.path.join(project_path, f"data/MAVEN_ERE_split/valid.jsonl"), "w") as f:
        f.writelines(valid_set)
    
    with open(os.path.join(project_path, f"data/MAVEN_ERE/valid.jsonl")) as f:
        lines = f.readlines()
    with open(os.path.join(project_path, f"data/MAVEN_ERE_split/test.jsonl"), "w") as f:
        f.writelines(lines)

    print("finish")
