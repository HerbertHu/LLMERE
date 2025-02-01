import os
import json
import copy
import itertools
import random
import math
from tqdm import tqdm

from constant import BIDIRECTIONAL_REL
from template import TASK_DESC_COREF


class Document:
    def __init__(self, data):
        self.id = data["id"]
        self.words = data["tokens"]
        self.mentions = []
        self.events = []
        self.eid2mentions = {}

        if "events" in data:
            for e in data["events"]:
                self.events += e["mention"]
            for e in data["events"]:
                self.eid2mentions[e["id"]] = e["mention"]
        else:
            self.events = copy.deepcopy(data['event_mentions'])

        self.timexes = data["TIMEX"]
        for t in data["TIMEX"]:
            self.eid2mentions[t["id"]] = [t]
        self.events_all = self.events + self.timexes

        self.sort_events()
        self.map_events()
        self.map_timexes()

        if "events" in data:
            self.temporal_relations = data["temporal_relations"]
            self.causal_relations = data["causal_relations"]
            self.subevent_relations = {"subevent": data["subevent_relations"]}
            self.coref_relations = None

            self.temporal_labels = self.get_relation_labels(self.temporal_relations)
            self.causal_labels = self.get_relation_labels(self.causal_relations)
            self.subevent_labels = self.get_relation_labels(self.subevent_relations)
            self.coref_labels = self.get_coref_labels(data)
        else:
            self.temporal_relations = {}
            self.causal_relations = {}
            self.subevent_relations = {}
            self.coref_relations = {}

        self.get_id2mention()

        self.temporal_labels_dict = self.get_choices(self.temporal_labels)
        self.causal_labels_dict = self.get_choices(self.causal_labels)
        self.subevent_labels_dict = self.get_choices(self.subevent_labels)
        self.coref_labels_dict = self.get_choices(self.coref_labels)

    def sort_events(self):
        self.events_all = sorted(self.events_all, key=lambda x: (x["sent_id"], x["offset"][0]))

    def get_id2mention(self):
        self.events_all_id2mention = {}
        for index, event in enumerate(self.events_all):
            if event["id"].startswith("TIME"):
                mention = event["mention"]
            else:
                mention = event["trigger_word"]
            self.events_all_id2mention[event["id"]] = mention

    def map_events(self):
        self.events_sorted = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.event_num2id = {f"e{index}": e["id"] for index, e in enumerate(self.events_sorted)}
        self.event_id2num = {e["id"]: f"e{index}" for index, e in enumerate(self.events_sorted)}

    def map_timexes(self):
        self.timexes_sorted = sorted(self.timexes, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.timex_num2id = {f"t{index}": e["id"] for index, e in enumerate(self.timexes_sorted)}
        self.timex_id2num = {e["id"]: f"t{index}" for index, e in enumerate(self.timexes_sorted)}

    def get_relation_labels(self, relations):
        new_relations = copy.deepcopy(relations)
        for rel in relations:
            pair_set = set()
            for pair in relations[rel]:
                for e1 in self.eid2mentions[pair[0]]:
                    for e2 in self.eid2mentions[pair[1]]:
                        if e1["id"].startswith("TIME"):
                            e1_map_id = self.timex_id2num[e1["id"]]
                        else:
                            e1_map_id = self.event_id2num[e1["id"]]

                        if e2["id"].startswith("TIME"):
                            e2_map_id = self.timex_id2num[e2["id"]]
                        else:
                            e2_map_id = self.event_id2num[e2["id"]]

                        pair_set.add((e1_map_id, e2_map_id))
                        if rel in BIDIRECTIONAL_REL:
                            pair_set.add((e2_map_id, e1_map_id))
            new_relations[rel] = list(pair_set)

        return new_relations

    def get_coref_labels(self, data):
        pair_list = []
        for event in data["events"]:
            for mention1, mention2 in itertools.permutations(event["mention"], 2):
                # coref is bidirectional
                m1_map_id = self.event_id2num[mention1["id"]]
                m2_map_id = self.event_id2num[mention2["id"]]
                pair_list.append((m1_map_id, m2_map_id))
        relations = {"coreference": pair_list}

        return relations

    def get_choices(self, labels):
        rel_choices = copy.deepcopy(labels)
        for rel in labels.keys():
            choices = {}
            for pair in labels[rel]:
                if pair[0] not in choices:
                    choices[pair[0]] = [pair[1]]
                else:
                    choices[pair[0]].append(pair[1])

            choices_sorted = copy.deepcopy(choices)
            for key in choices:
                choices_sorted[key] = sorted(choices[key], key=lambda x: int(x[1:]))

            new_choices_sorted = {}
            for key in choices_sorted:
                for event in choices_sorted[key]:
                    if event.startswith("t"):
                        map_id = self.timex_num2id[event]
                    else:
                        map_id = self.event_num2id[event]

                    mention = self.events_all_id2mention[map_id]
                    if key not in new_choices_sorted:
                        new_choices_sorted[key] = ["<" + event + " " + mention + ">"]
                    else:
                        new_choices_sorted[key].append("<" + event + " " + mention + ">")

            rel_choices[rel] = new_choices_sorted
        return rel_choices


def filter_golden_events(doc, split_events_sorted):
    tagged_events = []
    for event in split_events_sorted:
        event_id = event["id"]
        if event_id.startswith("TIME"):
            map_id = doc.timex_id2num[event_id]
        else:
            map_id = doc.event_id2num[event_id]
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
    with open(os.path.join(data_path, f"{split}.jsonl")) as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line.strip())
        doc = Document(data)

        for item in doc.events_all:
            if item["id"].startswith("TIME"):
                continue
            else:
                map_id = doc.event_id2num[item["id"]]

            # Document Partitioning Strategy
            n = len(doc.events_sorted) - 1
            if n == 0:
                pass
            k = 30
            m = math.ceil(n / k) if n > 0 else 1
            min_size = n // m
            extra = n % m
            sizes = [min_size + 1 if i < extra else min_size for i in range(m)]
            doc_split_num.append(m)

            new_events = copy.deepcopy(doc.events_sorted)
            new_events.remove(item)
            random.shuffle(new_events)

            index = 0
            for size in sizes:
                split_events = new_events[index:index + size]
                index += size
                split_events.append(item)
                split_events_sorted = sorted(split_events, key=lambda x: (x["sent_id"], x["offset"][0]))

                new_sent_list = []
                for sent_id, sent in enumerate(doc.words):
                    new_sent = []
                    offset = 0
                    for event in split_events_sorted:
                        if sent_id == event["sent_id"]:
                            event_map = doc.timex_id2num[event["id"]] if event["id"].startswith("TIME") else \
                                doc.event_id2num[event["id"]]
                            sp1 = event["offset"][0]
                            sp2 = event["offset"][1]
                            new_sent.extend(sent[offset: sp1])
                            new_sent.extend(["<" + event_map])
                            new_sent.extend([" ".join(sent[sp1: sp2]) + ">"])
                            offset = sp2
                    new_sent.extend(sent[offset:])
                    new_sent_list.append(" ".join(new_sent))
                text = " ".join(new_sent_list)

                instruction = TASK_DESC_COREF
                sample_desc = f"Please identify the events in the document that have the coreference relation " \
                              f"with the given event <{map_id} {doc.events_all_id2mention[item['id']]}>."

                tagged_events = filter_golden_events(doc, split_events_sorted)
                coref_labels_list = [
                    f"COREFERENCE: {choose_choices(map_id, doc.coref_labels_dict['coreference'], tagged_events)[0]}"
                ]
                relation_list = ["; ".join(coref_labels_list)]

                item_input = "Document content: " + text + "\n" + sample_desc
                item_output = "\n\n".join(relation_list)
                item_dict = {"instruction": instruction, "input": item_input, "output": item_output}

                no_relation_text = "COREFERENCE: none"
                if split == "train":
                    if item_output == no_relation_text:
                        examples_neg.append(item_dict)
                    else:
                        examples_pos.append(item_dict)
                else:
                    examples.append(item_dict)

    if split == "train":
        print(f"all_pos_num: {len(examples_pos)}")
        print(f"all_neg_num: {len(examples_neg)}")
        neg_num = int(len(examples_pos) / 2.0 * 3)
        print(f"keep_neg_num: {neg_num}")
        examples = examples_pos + random.sample(examples_neg, neg_num)
        random.shuffle(examples)
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

    dataset_name = "MAVEN_ERE"
    data_path = os.path.join(project_path, f"data/MAVEN_ERE_split")
    new_data_path = os.path.join(project_path, f"data/converted/{dataset_name}/coref")

    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    split_list = ["train", "valid", "test"]
    for split in split_list:
        convert_data(data_path, new_data_path, split)

    print("finish")
