REL2ID_DICT = {  # always put None type at the last
    "matres": {
        "BEFORE": 0,
        "AFTER": 1,
        "EQUAL": 2,
        "VAGUE": 3,
        "NONE": 4,
    },
    "tb-dense": {
        "BEFORE": 0,
        "AFTER": 1,
        "INCLUDES": 2,
        "IS_INCLUDED": 3,
        "SIMULTANEOUS": 4,
        "VAGUE": 5,
    },
    "tcr": {
        "BEFORE": 0,
        "AFTER": 1,
        "SIMULTANEOUS": 2,
        "VAGUE": 3,
        "NONE": 4,
    }
}

INVERSE_REL_DICT = {
    "matres": {
        "BEFORE": "AFTER",
        "AFTER": "BEFORE"
    },
    "tb-dense": {
        "BEFORE": "AFTER",
        "AFTER": "BEFORE",
        "INCLUDES": "IS_INCLUDED",
        "IS_INCLUDED": "INCLUDES"
    }
}

INVERSE_REL_DICT["tcr"] = INVERSE_REL_DICT["matres"]

BIDIRECTIONAL_REL_DICT = {
    "matres": ["EQUAL"],
    "tb-dense": ["SIMULTANEOUS"],
    "tcr": ["SIMULTANEOUS"]
}

NONE_REL_DICT = {
    "matres": "NONE",
    "tb-dense": "VAGUE"
}

NONE_REL_DICT["tcr"] = NONE_REL_DICT["matres"]

EVAL_EXCLUDE = {
    "matres": ["NONE", "VAGUE"],
    "tb-dense": ["VAGUE"]
}

EVAL_EXCLUDE["tcr"] = EVAL_EXCLUDE["matres"]

ERROR = 0

REL2ID = REL2ID_DICT["matres"]
ID2REL = {v: k for k, v in REL2ID.items()}
EVAL_EXCLUDE_ID = [REL2ID[r] for r in EVAL_EXCLUDE["matres"] if r in REL2ID]
TEMP_REPORT_CLASS_NAMES = [ID2REL[i] for i in range(len(ID2REL)) if i not in EVAL_EXCLUDE_ID]
TEMP_REPORT_CLASS_LABELS = [i for i in range(len(ID2REL)) if i not in EVAL_EXCLUDE_ID]
