import argparse
import os
import json
import copy
from sklearn.metrics import classification_report

from constant import *


class Document:
    def __init__(self, data, dataname, ignore_nonetype=False):
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
        self.temporal_dict = self.load_relation_dict(self.relations, REL2ID_DICT["matres"])
        self.new_relations = self.get_new_relation(self.relations)
        self.candidate_events = self.get_candidate_events(self.new_relations)

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

    def load_relation_dict(self, relations, REL2ID):
        pair2rel = {}
        for rel in relations:
            for pair in relations[rel]:
                pair2rel[(pair[0], pair[1])] = REL2ID[rel]

        return pair2rel

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


def get_golden_relation_labels(doc, pair2rel, ignore_nonetype=True):
    labels = []
    for i, e1 in enumerate(doc.events_clean):
        for j, e2 in enumerate(doc.events_clean):
            if i == j:
                continue
            if doc.dataname in ["matres", "tcr"]:
                if ignore_nonetype:
                    labels.append(pair2rel.get((e1["eiid"], e2["eiid"]), -100))
                else:
                    labels.append(pair2rel.get((e1["eiid"], e2["eiid"]), REL2ID_DICT["matres"]["NONE"]))
    assert len(labels) == len(doc.events_clean) ** 2 - len(doc.events_clean)
    return labels


def get_pred_relation_labels(doc, pair2rel, ignore_nonetype=True):
    labels = []
    for i, e1 in enumerate(doc.events_clean):
        for j, e2 in enumerate(doc.events_clean):
            if i == j:
                continue
            if ignore_nonetype:
                labels.append(pair2rel.get((e1["eiid"], e2["eiid"]), REL2ID_DICT["matres"]["VAGUE"]))
            else:
                labels.append(pair2rel.get((e1["eiid"], e2["eiid"]), REL2ID_DICT["matres"]["NONE"]))
    assert len(labels) == len(doc.events_clean) ** 2 - len(doc.events_clean)
    count = len([x for x in labels if x != 3])
    try:
        assert count == len(pair2rel)
    except:
        print("count not equal pair2rel")

    return labels


def filter_label(label_list, pred_list):
    new_label_list = []
    new_pred_list = []
    for label, pred in zip(label_list, pred_list):
        if label != -100:
            new_label_list.append(label)
            new_pred_list.append(pred)

    return new_label_list, new_pred_list


def convert_and_evaluate(dataset_golden, dataset_predict, doc_split_num):
    rel_label_list = []
    rel_pred_list = []

    pairs_dict_list = []
    for data_predict in dataset_predict:
        # get predict event pairs
        data_json = json.loads(data_predict)
        predict_text = data_json["predict"]
        predict_text = predict_text.split("\n")[0]
        relation_text_list = predict_text.strip().rstrip(".").split("; ")

        pairs_dict = {}
        for index, relation_text in enumerate(relation_text_list):
            try:
                relation_label = relation_text.split(": ")[0].lstrip(" ")
                pairs = relation_text.split(": ")[1]
                none_list = ["none", "none.", "NONE", "NONE.", "None", "None."]
                if pairs in none_list:
                    pairs_list = []
                else:
                    pairs_list = pairs.split(">, <")
            except:
                pairs_list = []
                print("*" * 10)
                print(relation_text_list)
                print(relation_text)
                print("*" * 10)

            pairs_list_new = [x.lstrip("<").rstrip(">").split(" ")[0] for x in pairs_list]
            pairs_dict[relation_label] = pairs_list_new
        pairs_dict_list.append(pairs_dict)
    assert len(dataset_predict) == len(pairs_dict_list)
    doc_split_num_sum = sum(doc_split_num)
    assert doc_split_num_sum == len(dataset_predict)

    start_index_events, start_index_examples = 0, 0
    for data_golden in dataset_golden:
        data_golden_json = json.loads(data_golden)
        doc = Document(data_golden_json, "MATRES", ignore_nonetype=True)

        rel_label_list.extend(get_golden_relation_labels(doc, doc.temporal_dict))

        end_index_events = start_index_events + len(doc.candidate_events)
        doc_split_num_item = doc_split_num[start_index_events: end_index_events]
        start_index_events = end_index_events

        rel_pred_dict = {}
        for event, examples_num_item in zip(doc.candidate_events, doc_split_num_item):
            event = doc.enum2events[event]
            end_index_examples = start_index_examples + examples_num_item
            pairs_preds = pairs_dict_list[start_index_examples: end_index_examples]
            start_index_examples = end_index_examples
            for pairs_pred in pairs_preds:
                e1_id = event["eiid"]

                for label in REL2ID_DICT["matres"].keys():
                    if label != "NONE" and label != "AFTER" and label != "VAGUE":
                        if label not in pairs_pred:
                            print(f"label {label} does not exist")
                            print(pairs_pred)
                            continue

                        for e2_map_id in pairs_pred[label]:
                            try:
                                e2_id = doc.event_num2id[e2_map_id]
                                if int(e1_id[2:]) > int(e2_id[2:]):
                                    if label == "BEFORE":
                                        if (e2_id, e1_id) not in rel_pred_dict.keys():
                                            rel_pred_dict[(e2_id, e1_id)] = REL2ID_DICT["matres"]["AFTER"]
                                    elif label == "EQUAL" or label == "VAGUE":
                                        if (e2_id, e1_id) not in rel_pred_dict.keys():
                                            rel_pred_dict[(e2_id, e1_id)] = REL2ID_DICT["matres"][label]
                                else:
                                    if (e1_id, e2_id) not in rel_pred_dict.keys():
                                        rel_pred_dict[(e1_id, e2_id)] = REL2ID_DICT["matres"][label]
                            except:
                                print(f"event {e2_map_id} does not exist")
        rel_pred_list.extend(get_pred_relation_labels(doc, rel_pred_dict, ignore_nonetype=True))
    assert start_index_examples == len(pairs_dict_list)

    result_collection = {}
    rel_label_list, rel_pred_list = filter_label(rel_label_list, rel_pred_list)
    assert len(rel_label_list) == len(rel_pred_list)
    temporal_res = classification_report(rel_label_list, rel_pred_list, output_dict=True,
                                         target_names=TEMP_REPORT_CLASS_NAMES, labels=TEMP_REPORT_CLASS_LABELS,
                                         zero_division=0)
    result_collection["Temporal"] = temporal_res
    print(f"Temporal: precision={temporal_res['micro avg']['precision'] * 100:.5f}, "
          f"recall={temporal_res['micro avg']['recall'] * 100:.5f}, "
          f"f1={temporal_res['micro avg']['f1-score'] * 100:.5f}")

    return result_collection


def do_convert(args):
    if not os.path.exists(args.predict_file):
        raise ValueError("Please input the correct path of predict file.")
    if not os.path.exists(args.golden_file):
        raise ValueError("Please input the correct path of predict file.")

    with open(args.golden_file, "r", encoding="utf-8") as f:
        dataset_golden = f.readlines()
    with open(args.predict_file, "r", encoding="utf-8") as f:
        dataset_predict = f.readlines()
    with open(args.split_num_file, "r", encoding="utf-8") as f:
        doc_split_num = json.load(f)

    result_collection = convert_and_evaluate(dataset_golden, dataset_predict, doc_split_num)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as outfile:
        json.dump(result_collection, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_version", type=str, default="MATRES_temporal",
                        help="The predict results, only support the JSON format.")
    args = parser.parse_args()

    file_dir = os.path.dirname(__file__)
    args.project_path = os.path.abspath(os.path.join(file_dir, "../.."))
    args.predict_file = os.path.join(args.project_path, "output/predict", args.result_version,
                                     "generated_predictions.jsonl")
    args.golden_file = os.path.join(args.project_path,
                                    f"data/processed/MATRES/test.json")
    args.split_num_file = os.path.join(args.project_path,
                                       f"data/converted/MATRES/test_doc_split_num.json")
    args.output_dir = os.path.join(args.project_path, "output/eval", args.result_version)

    do_convert(args)
    print("finish")
