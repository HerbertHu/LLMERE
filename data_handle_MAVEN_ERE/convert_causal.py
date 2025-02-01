import os
import json
import copy
import itertools
import random
import math
import collections
import networkx as nx
from tqdm import tqdm

from constant import BIDIRECTIONAL_REL
from template import TASK_DESC_CAUSAL


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


def map_triple2text(head_node, tail_node, relation_type):
    text = None
    if relation_type == "coreference":
        text = f"{head_node} and {tail_node} are identical"
    elif relation_type == "CAUSE":
        text = f"{head_node} causes {tail_node} to occur"
    elif relation_type == "PRECONDITION":
        text = f"{head_node} is a precondition for {tail_node}"
    return text


def get_whole_triple(map_id, mention, doc, tagged_events):
    queue = collections.deque()
    queue.append(f"<{map_id} {mention}>")
    visited_set = set()
    visited_set.add(f"<{map_id} {mention}>")
    hop = 1

    triple_list = []
    while queue:
        for i in range(len(queue)):
            head_node = queue.popleft()
            head_node_id = head_node.split(" ")[0].lstrip("<")

            next_node_dict = {
                "CAUSE": choose_choices(head_node_id, doc.causal_labels_dict["CAUSE"], tagged_events)[1],
                "PRECONDITION": choose_choices(head_node_id, doc.causal_labels_dict["PRECONDITION"], tagged_events)[
                    1]}

            for relation_type in next_node_dict.keys():
                for tail_node in next_node_dict[relation_type]:
                    triple_list.append((head_node, tail_node, relation_type))
                    if tail_node not in visited_set:
                        queue.append(tail_node)
                        visited_set.add(tail_node)
        hop += 1

    return triple_list


def get_coref_text(map_id, mention, doc, tagged_events):
    head_node_id = map_id
    head_node = f"<{map_id} {mention}>"

    next_node_dict = {
        "coreference": choose_choices(head_node_id, doc.coref_labels_dict["coreference"], tagged_events)[1]
    }
    relation_text = []
    for tail_node in next_node_dict["coreference"]:
        relation_text.append(map_triple2text(head_node, tail_node, "coreference"))
    coref_text = ", ".join(relation_text)

    return coref_text


def check_logic(whole_graph, paths):
    def infer_relation(relation1, relation2):
        if relation1 == "CAUSE":
            if relation2 == "CAUSE":
                return "CAUSE"
            elif relation2 == "PRECONDITION":
                return "PRECONDITION"
        elif relation1 == "PRECONDITION":
            if relation2 == "CAUSE":
                return "PRECONDITION"
            elif relation2 == "PRECONDITION":
                return "PRECONDITION"
        else:
            return None

    if not paths:
        return []

    right_paths = []
    for path in paths:
        if len(path) <= 2:
            continue
        path_edges = []
        for i in range(0, len(path) - 1):
            edge_data = whole_graph.get_edge_data(path[i], path[i + 1])
            path_edges.append(edge_data["edge_type"])

        current_relation = None
        for i in range(len(path_edges) - 1):
            relation1 = path_edges[i]
            relation2 = path_edges[i + 1]

            if current_relation is None:
                current_relation = infer_relation(relation1, relation2)
            else:
                current_relation = infer_relation(current_relation, relation2)
        direct_edge = whole_graph.get_edge_data(path[0], path[-1])
        if current_relation == direct_edge["edge_type"]:
            right_paths.append(path)

    return right_paths


def get_whole_graph(triple_list):
    G = nx.DiGraph()

    for source, target, edge_type in triple_list:
        G.add_edge(source, target, edge_type=edge_type)

    return G


def get_multi_hop_path(whole_graph, center_node):
    if whole_graph.number_of_edges() > 0:
        one_hop_nodes = list(whole_graph.successors(center_node))
    else:
        one_hop_nodes = []
    all_paths = []
    for end_node in one_hop_nodes:
        node_paths = list(nx.all_simple_paths(whole_graph, source=center_node, target=end_node))
        right_paths = check_logic(whole_graph, node_paths)
        if right_paths:
            all_paths.extend(random.sample(right_paths, 1))

    paths_with_edge_info = []
    for path in all_paths:
        path_edges = []
        for i in range(0, len(path) - 1):
            edge_data = whole_graph.get_edge_data(path[i], path[i + 1])
            path_edges.append((path[i], path[i + 1], edge_data["edge_type"]))
        paths_with_edge_info.append(path_edges)

    return paths_with_edge_info


def get_path_text(paths_with_edge_info):
    paths_text_list = []
    for path in paths_with_edge_info:
        item_text_list = []
        for triple in path:
            item_text_list.append(map_triple2text(triple[0], triple[1], triple[2]))
        item_text = ", ".join(item_text_list)
        paths_text_list.append(item_text)
    paths_text = "; ".join(paths_text_list)

    return paths_text, paths_text_list


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
                            # Events are marked with special symbols
                            new_sent.extend(sent[offset: sp1])
                            new_sent.extend(["<" + event_map])
                            new_sent.extend([" ".join(sent[sp1: sp2]) + ">"])
                            offset = sp2
                    new_sent.extend(sent[offset:])
                    new_sent_list.append(" ".join(new_sent))
                text = " ".join(new_sent_list)

                instruction = TASK_DESC_CAUSAL
                mention = doc.events_all_id2mention[item['id']]
                sample_desc = f"Please identify the events in the document that have causal relations " \
                              f"with the given event <{map_id} {mention}>."

                tagged_events = filter_golden_events(doc, split_events_sorted)
                cuasal_labels_list = [
                    f"CAUSE: {choose_choices(map_id, doc.causal_labels_dict['CAUSE'], tagged_events)[0]}",
                    f"PRECONDITION: {choose_choices(map_id, doc.causal_labels_dict['PRECONDITION'], tagged_events)[0]}"
                ]
                relation_list = ["; ".join(cuasal_labels_list)]

                # Multi-hop subgraph
                triple_list = get_whole_triple(map_id, mention, doc, tagged_events)
                whole_graph = get_whole_graph(triple_list)
                paths_with_edge_info = get_multi_hop_path(whole_graph, f"<{map_id} {mention}>")
                paths_text, paths_text_list = get_path_text(paths_with_edge_info)
                coref_text = get_coref_text(map_id, mention, doc, tagged_events)
                if coref_text == "":
                    coref_info = "Coreference information: none"
                else:
                    coref_info = f"Coreference information: {coref_text}"

                if paths_text_list:
                    relevant_info = f"Relevant reasoning information: {'; '.join(paths_text_list)}"
                else:
                    relevant_info = f"Relevant reasoning information: none"

                item_input = "Document content: " + text + "\n" + sample_desc
                item_output = "\n\n".join(relation_list) + "\n" + coref_info + "\n" + relevant_info
                item_dict = {"instruction": instruction, "input": item_input, "output": item_output}

                no_relation_text = "CAUSE: none; PRECONDITION: none"
                if split == "train":
                    if item_output.split("\n")[0] == no_relation_text:
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
    new_data_path = os.path.join(project_path, f"data/converted/{dataset_name}/causal")

    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    split_list = ["train", "valid", "test"]
    for split in split_list:
        convert_data(data_path, new_data_path, split)

    print("finish")
