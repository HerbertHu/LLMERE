import argparse
import os
import json
from sklearn.metrics import classification_report, confusion_matrix

from constant import *


class Document:
    def __init__(self, data, dataname):
        self.id = data["id"]
        self.text = data["text"]
        self.events = data["events"]
        self.relations = data["relations"]
        self.dataname = dataname.lower()

        self.sort_events()
        self.map_events()
        self.get_id2mention()
        self.subevent_dict = self.load_relation_dict(self.relations, REL2ID_DICT[self.dataname])

    def sort_events(self):
        self.events_sorted = sorted(self.events, key=lambda x: (x["sent_id"], x["offset"][0]))

    def get_id2mention(self):
        self.events_all_id2mention = {}
        for index, event in enumerate(self.events_sorted):
            self.events_all_id2mention[event["id"]] = event["mention"]

    def map_events(self):
        self.event_num2id = {}
        self.event_id2num = {}
        self.event_num2id = {f"e{index}": e["id"] for index, e in enumerate(self.events_sorted)}
        self.event_id2num = {e["id"]: f"e{index}" for index, e in enumerate(self.events_sorted)}
        self.enum2events = {f"e{index}": e for index, e in enumerate(self.events_sorted)}

    def load_relation_dict(self, relations, REL2ID):
        pair2rel = {}
        for rel in relations:
            for pair in relations[rel]:
                pair2rel[(pair[0], pair[1])] = REL2ID[rel]

        return pair2rel


def get_relation_labels(doc, pair2rel, ignore_nonetype=False):
    labels = []
    for i, e1 in enumerate(doc.events_sorted):
        for j, e2 in enumerate(doc.events_sorted):
            if i == j:
                continue
            if i > j:  # only consider e1 before e2
                labels.append(-100)
                continue
            if ignore_nonetype:
                labels.append(pair2rel.get((e1["id"], e2["id"]), -100))
            else:
                labels.append(pair2rel.get((e1["id"], e2["id"]), REL2ID_DICT[doc.dataname]["NONE"]))
    assert len(labels) == len(doc.events_sorted) ** 2 - len(doc.events_sorted)
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
        doc = Document(data_golden_json, "hievents")

        rel_label_list.extend(get_relation_labels(doc, doc.subevent_dict, ignore_nonetype=False))

        end_index_events = start_index_events + len(doc.events_sorted)
        doc_split_num_item = doc_split_num[start_index_events: end_index_events]
        start_index_events = end_index_events

        rel_pred_dict = {}
        for event, examples_num_item in zip(doc.events_sorted, doc_split_num_item):
            end_index_examples = start_index_examples + examples_num_item
            pairs_preds = pairs_dict_list[start_index_examples: end_index_examples]
            start_index_examples = end_index_examples
            for pairs_pred in pairs_preds:
                e1_id = event["id"]
                e1_map_id = doc.event_id2num[e1_id]

                for label in REL2ID_DICT["hievents"].keys():
                    if label != "NONE" and label != "Coref":
                        if label not in pairs_pred:
                            print(f"label {label} does not exist")
                            print(pairs_pred)
                            continue

                        for e2_map_id in pairs_pred[label]:
                            try:
                                e2_id = doc.event_num2id[e2_map_id]
                                if int(e1_map_id[1:]) > int(e2_map_id[1:]):
                                    if label == "SuperSub" and (e2_id, e1_id) not in rel_pred_dict.keys():
                                        rel_pred_dict[(e2_id, e1_id)] = REL2ID_DICT["hievents"]["SubSuper"]
                                    if label == "SubSuper" and (e2_id, e1_id) not in rel_pred_dict.keys():
                                        rel_pred_dict[(e2_id, e1_id)] = REL2ID_DICT["hievents"]["SuperSub"]
                                    if label == "Coref" and (e2_id, e1_id) not in rel_pred_dict.keys():
                                        rel_pred_dict[(e2_id, e1_id)] = REL2ID_DICT["hievents"]["Coref"]
                                else:
                                    if (e1_id, e2_id) not in rel_pred_dict.keys():
                                        rel_pred_dict[(e1_id, e2_id)] = REL2ID_DICT["hievents"][label]
                            except:
                                print(f"event {e2_map_id} does not exist")
        rel_pred_list.extend(get_relation_labels(doc, rel_pred_dict, ignore_nonetype=False))
    assert start_index_examples == len(pairs_dict_list)

    result_collection = {}
    rel_label_list, rel_pred_list = filter_label(rel_label_list, rel_pred_list)
    assert len(rel_label_list) == len(rel_pred_list)
    print(f"rel_label_list: {len(rel_label_list)}")
    subevent_res = classification_report(rel_label_list, rel_pred_list, output_dict=True,
                                         target_names=REPORT_CLASS_NAMES_subevent, labels=REPORT_CLASS_LABELS_subevent,
                                         zero_division=0)
    # print(confusion_matrix(rel_label_list, rel_pred_list))
    # print(subevent_res)
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
    parser.add_argument("--result_version", type=str, default="Hieve_subevent",
                        help="The predict results, only support the JSON format.")
    args = parser.parse_args()

    file_dir = os.path.dirname(__file__)
    fold = 0
    args.project_path = os.path.abspath(os.path.join(file_dir, "../.."))
    args.predict_file = os.path.join(args.project_path, "output/predict", args.result_version,
                                     "generated_predictions.jsonl")
    args.golden_file = os.path.join(args.project_path,
                                    f"data/processed/hievents/test.json")
    args.split_num_file = os.path.join(args.project_path,
                                       f"data/converted/Hieve/test_split.json")
    args.output_dir = os.path.join(args.project_path, "output/eval", args.result_version)

    do_convert(args)
    print("finish")
