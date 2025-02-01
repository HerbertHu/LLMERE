import os
import json
import copy
import random
import math
from tqdm import tqdm

from template import TASK_DESC_TEMPORAL


class Document:
    def __init__(self, data, dataname):
        self.id = data["fid"]
        self.text = data["text"]
        self.events = data["events"]
        if dataname.lower() == "tb-dense":
            self.events += data["timexes"]
        self.relations = data["relations"]
        self.dataname = dataname.lower()

        self.sort_events()
        self.map_events()
        self.get_id2mention()
        self.new_relations = self.get_new_relation(self.relations)
        self.temporal_labels_dict = self.get_choices(self.new_relations)
        self.candidate_events = self.get_candidate_events(self.new_relations)
        # self.get_labels(ignore_nonetype)

    def sort_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.events_clean = []
        for e in self.events:
            if "eiid" not in e.keys():
                pass
            else:
                self.events_clean.append(e)
        self.events_sorted = sorted(self.events_clean, key=lambda x: (x["sent_id"], x["offset"][0]))

    def get_id2mention(self):
        self.events_all_id2mention = {}
        for index, event in enumerate(self.events_sorted):
            mention = self.text[event["sent_id"]][event["offset"][0]:event["offset"][1]]
            self.events_all_id2mention[event["eiid"]] = mention

    def map_events(self):
        self.event_num2id = {}
        self.event_id2num = {}
        self.event_num2id = {f"e{index}": e["eiid"] for index, e in enumerate(self.events_sorted)}
        self.event_id2num = {e["eiid"]: f"e{index}" for index, e in enumerate(self.events_sorted)}
        self.enum2events = {f"e{index}": e for index, e in enumerate(self.events_sorted)}

    def get_new_relation(self, relations):
        new_relations = copy.deepcopy(relations)
        for rel in ["BEFORE", "EQUAL", "VAGUE"]:
            if rel not in new_relations.keys():
                new_relations[rel] = []

        for rel in relations:
            if rel == "AFTER":
                for pair in relations[rel]:
                    new_relations["BEFORE"].append([pair[1], pair[0]])
                del new_relations["AFTER"]
            if rel in ["VAGUE", "EQUAL"]:
                for pair in relations[rel]:
                    new_relations[rel].append([pair[1], pair[0]])

        return new_relations

    def get_choices(self, labels):
        rel_choices = copy.deepcopy(labels)
        for rel in labels.keys():
            choices = {}
            for pair in labels[rel]:
                e1 = self.event_id2num[pair[0]]
                e2 = self.event_id2num[pair[1]]
                if e1 not in choices:
                    choices[e1] = [e2]
                else:
                    choices[e1].append(e2)

            choices_sorted = copy.deepcopy(choices)
            for key in choices:
                choices_sorted[key] = sorted(choices[key], key=lambda x: int(x[1:]))

            new_choices_sorted = {}
            for key in choices_sorted:
                for event in choices_sorted[key]:
                    map_id = self.event_num2id[event]
                    mention = self.events_all_id2mention[map_id]
                    if key not in new_choices_sorted:
                        new_choices_sorted[key] = ["<" + event + " " + mention + ">"]
                    else:
                        new_choices_sorted[key].append("<" + event + " " + mention + ">")

            rel_choices[rel] = new_choices_sorted
        return rel_choices

    def get_candidate_events(self, new_relations):
        candidate_events = {}
        for rel in new_relations.keys():
            for pair in new_relations[rel]:
                e1 = self.event_id2num[pair[0]]
                e2 = self.event_id2num[pair[1]]
                if e1 not in candidate_events:
                    candidate_events[e1] = [e2]
                else:
                    candidate_events[e1].append(e2)

        return candidate_events


def filter_golden_events(doc, split_events_sorted):
    tagged_events = []
    for map_id in split_events_sorted:
        event = doc.enum2events[map_id]
        event_id = event["eiid"]
        mention = doc.events_all_id2mention[event_id]
        tagged_events.append("<" + map_id + " " + mention + ">")
    return tagged_events


def choose_choices(map_id, choices, tagged_events):
    res = []
    if map_id in choices.keys():
        temp_res = choices[map_id]
        for i in temp_res:
            if i in tagged_events:
                res.append(i)
    else:
        res = []

    if len(res) > 0:
        res_text = ", ".join(res)
    else:
        res_text = ", ".join(['none'])
    return res_text, res


def convert_data(data_path, new_data_path, split):
    examples = []
    examples_pos, examples_neg = [], []
    doc_split_num = []
    with open(os.path.join(data_path, f"{split}.json")) as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line.strip())
        doc = Document(data, "MATRES")
        pass

        for item in doc.candidate_events:
            map_id = item

            # Document Partitioning Strategy
            n = len(doc.candidate_events[item])
            if n == 0:
                pass
            k = 30
            m = math.ceil(n / k) if n > 0 else 1
            min_size = n // m
            extra = n % m
            sizes = [min_size + 1 if i < extra else min_size for i in range(m)]
            doc_split_num.append(m)

            new_events = copy.deepcopy(doc.candidate_events[item])
            random.shuffle(new_events)

            index = 0
            for size in sizes:
                split_events = new_events[index:index + size]
                index += size
                split_events.append(item)
                split_events_sorted = sorted(split_events, key=lambda x: int(x[1:]))

                new_sent_list = []
                for sent_id, sent in enumerate(doc.text):
                    new_sent = []
                    offset = 0
                    for event_num in split_events_sorted:
                        event = doc.enum2events[event_num]
                        if sent_id == event["sent_id"]:
                            event_map = event_num
                            sp1 = event["offset"][0]
                            sp2 = event["offset"][1]
                            new_sent.extend(sent[offset: sp1])
                            new_sent.extend(["<" + event_map + " "])
                            new_sent.extend(sent[sp1: sp2])
                            new_sent.extend([">"])
                            offset = sp2
                    new_sent.extend(sent[offset:])
                    new_sent_list.append("".join(new_sent))
                text = " ".join(new_sent_list)

                instruction = TASK_DESC_TEMPORAL
                mention = doc.events_all_id2mention[doc.enum2events[item]['eiid']]
                sample_desc = f"Please identify the events in the document that have temporal relations " \
                              f"with the given event <{map_id} {mention}>."

                tagged_events = filter_golden_events(doc, split_events_sorted)
                temporal_labels_list = [
                    f"EQUAL: {choose_choices(map_id, doc.temporal_labels_dict['EQUAL'], tagged_events)[0]}",
                    f"BEFORE: {choose_choices(map_id, doc.temporal_labels_dict['BEFORE'], tagged_events)[0]}"
                ]
                relation_label = "; ".join(temporal_labels_list)

                item_input = "Document content: " + text + "\n" + sample_desc
                item_output = relation_label
                item_dict = {"instruction": instruction, "input": item_input, "output": item_output}

                no_relation_text = "EQUAL: none; BEFORE: none"
                if split == "train":
                    if relation_label == no_relation_text:
                        examples_neg.append(item_dict)
                    else:
                        examples_pos.append(item_dict)
                else:
                    examples.append(item_dict)

    if split == "train":
        print(f"all_pos_num: {len(examples_pos)}")
        print(f"all_neg_num: {len(examples_neg)}")
        examples = examples_pos + examples_neg
        random.shuffle(examples)
        keep_num = 100000
        examples = examples[:keep_num]
        print(f"keep_num: {len(examples)}")
    else:
        file = os.path.join(new_data_path, f"{split}_doc_split_num.json")
        with open(file, "w") as f:
            json.dump(doc_split_num, f)

    new_data_file = os.path.join(new_data_path, f"{split}.json")
    with open(new_data_file, 'w') as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)
    print(f"Convert {split} finish")


if __name__ == "__main__":
    file_path = os.path.dirname(__file__)
    project_path = os.path.abspath(os.path.join(file_path, ".."))
    print(project_path)
    random.seed(42)

    if not os.path.exists(os.path.join(project_path, "data/converted")):
        os.makedirs(os.path.join(project_path, "data/converted"))

    dataset_name = "MATRES"
    data_path = os.path.join(project_path, f"data/processed/MATRES")
    new_data_path = os.path.join(project_path, f"data/converted/{dataset_name}")

    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    split_list = ["train", "dev", "test"]
    for split in split_list:
        convert_data(data_path, new_data_path, split)

    print("finish")
