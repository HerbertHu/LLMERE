import argparse
import os
import json
import copy
import itertools
from sklearn.metrics import classification_report

from constant import *


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
            self.temporal_dict = self.load_relation_dict(data["temporal_relations"], TEMPREL2ID)
            self.causal_dict = self.load_relation_dict(data["causal_relations"], CAUSALREL2ID)
            self.subevent_dict = self.load_relation_dict({"SUBEVENT": data["subevent_relations"]}, SUBEVENTREL2ID)
            self.coref_dict = self.load_coref_dict(data)

    def sort_events(self):
        self.events_all = sorted(self.events_all, key=lambda x: (x["sent_id"], x["offset"][0]))

    def map_events(self):
        self.events = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.event_num2id = {f"e{index}": e["id"] for index, e in enumerate(self.events)}
        self.event_id2num = {e["id"]: f"e{index}" for index, e in enumerate(self.events)}

    def map_timexes(self):
        self.timexes = sorted(self.timexes, key=lambda x: (x["sent_id"], x["offset"][0]))
        self.timex_num2id = {f"t{index}": t["id"] for index, t in enumerate(self.timexes)}
        self.timex_id2num = {t["id"]: f"t{index}" for index, t in enumerate(self.timexes)}

    def load_coref_dict(self, data):
        pair2rel = {}
        for event in data["events"]:
            for mention1, mention2 in itertools.permutations(event["mention"], 2):
                # coref is bidirectional
                pair2rel[(mention1["id"], mention2["id"])] = COREFREL2ID["COREFERENCE"]
        return pair2rel

    def load_relation_dict(self, relations, REL2ID):
        pair2rel = {}
        for rel in relations:
            for pair in relations[rel]:
                # first mention as the event mention
                for e1 in self.eid2mentions[pair[0]]:
                    for e2 in self.eid2mentions[pair[1]]:
                        pair2rel[(e1["id"], e2["id"])] = REL2ID[rel]
                        if rel in BIDIRECTIONAL_REL:
                            pair2rel[(e2["id"], e1["id"])] = REL2ID[rel]
        return pair2rel


def get_relation_labels(doc, pair2rel, ignore_timex=True):
    labels = []
    for i, e1 in enumerate(doc.events_all):
        for j, e2 in enumerate(doc.events_all):
            if i == j or e1["id"] == e2["id"]:
                continue
            if ignore_timex:
                if e1["id"].startswith("TIME") or e2["id"].startswith("TIME"):
                    labels.append(-100)
                    continue
            labels.append(pair2rel.get((e1["id"], e2["id"]), 0))
    assert len(labels) == len(doc.events_all) ** 2 - len(doc.events_all)
    return labels


def filter_label(label_list):
    new_label_list = []
    for label in label_list:
        if label != -100:
            new_label_list.append(label)

    return new_label_list


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
                    pairs_list = pairs.split(", ")
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
        doc = Document(data_golden_json)

        rel_label_list.extend(get_relation_labels(doc, doc.subevent_dict, ignore_timex=True))

        end_index_events = start_index_events + len(doc.events)
        doc_split_num_item = doc_split_num[start_index_events: end_index_events]
        start_index_events = end_index_events

        rel_pred_dict = {}
        for event, examples_num_item in zip(doc.events, doc_split_num_item):
            end_index_examples = start_index_examples + examples_num_item
            pairs_preds = pairs_dict_list[start_index_examples: end_index_examples]
            start_index_examples = end_index_examples
            for pairs_pred in pairs_preds:
                e1_id = event["id"]

                for label in SUBEVENTREL2ID.keys():
                    if label != "NONE":
                        if label not in pairs_pred:
                            print(f"label {label} does not exist")
                            print(pairs_pred)
                            continue

                        for e2_map_id in pairs_pred[label]:
                            try:
                                e2_id = doc.event_num2id[e2_map_id] if e2_map_id.startswith("e") else doc.timex_num2id[
                                    e2_map_id]
                                if (e1_id, e2_id) not in rel_pred_dict.keys():
                                    rel_pred_dict[(e1_id, e2_id)] = SUBEVENTREL2ID[label]
                            except:
                                print(f"event {e2_map_id} does not exist")
        rel_pred_list.extend(get_relation_labels(doc, rel_pred_dict, ignore_timex=True))
    assert start_index_examples == len(pairs_dict_list)

    result_collection = {}
    rel_label_list = filter_label(rel_label_list)
    rel_pred_list = filter_label(rel_pred_list)
    assert len(rel_label_list) == len(rel_pred_list)
    print(f"rel_label_list: {len(rel_label_list)}")
    subevent_res = classification_report(rel_label_list, rel_pred_list, output_dict=True,
                                         target_names=SUBEVENT_REPORT_CLASS_NAMES, labels=SUBEVENT_REPORT_CLASS_LABELS,
                                         zero_division=0)
    result_collection["Subevent"] = subevent_res
    print(f"Subevent: precision={subevent_res['micro avg']['precision'] * 100:.5f}, "
          f"recall={subevent_res['micro avg']['recall'] * 100:.5f}, "
          f"f1={subevent_res['micro avg']['f1-score'] * 100:.5f}")

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
    parser.add_argument("--result_version", type=str, default="MAVEN_ERE_subevent",
                        help="The predict results, only support the JSON format.")
    args = parser.parse_args()

    file_dir = os.path.dirname(__file__)
    args.project_path = os.path.abspath(os.path.join(file_dir, "../.."))
    args.predict_file = os.path.join(args.project_path, "output/predict", args.result_version,
                                     "generated_predictions.jsonl")
    args.golden_file = os.path.join(args.project_path,
                                    f"data/MAVEN_ERE_split/test.jsonl")
    args.split_num_file = os.path.join(args.project_path,
                                       f"data/converted/MAVEN_ERE/subevent/test_doc_split_num.json")
    args.output_dir = os.path.join(args.project_path, "output/eval", args.result_version)

    do_convert(args)
    print("finish")
