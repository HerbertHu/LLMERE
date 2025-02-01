REL2ID_DICT = {  # always put None type at the last
    "hievents": {
        "NONE": 0,
        "SuperSub": 1,
        "SubSuper": 2,
        "Coref": 3
    }
}

NONE_REL_DICT = {
    "hievents": "NONE",
}

EID_DICT = {
    "hievents": "id",
}

BIDIRECTIONAL_REL_DICT = {
    "hievents": []
}

REL2ID = REL2ID_DICT["hievents"]
ID2REL = {v: k for k, v in REL2ID.items()}
EVAL_EXCLUDE_ID = [REL2ID["NONE"]]
REPORT_CLASS_NAMES = [ID2REL[i] for i in range(len(ID2REL)) if i not in EVAL_EXCLUDE_ID]
REPORT_CLASS_LABELS = [i for i in range(len(ID2REL)) if i not in EVAL_EXCLUDE_ID]

REPORT_CLASS_NAMES_subevent = ["SuperSub", "SubSuper"]
REPORT_CLASS_LABELS_subevent = [1, 2]
