CLASS_NAMES = [
    "neutrophil",
    "epithelial",
    "lymphocyte",
    "plasma",
    "eosinophil",
    "connective",
]

# Class ids are the CoNIC ids (0 background, 1..6 classes).
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (242, 104, 104),
    2: (67, 160, 71),
    3: (66, 133, 244),
    4: (171, 71, 188),
    5: (255, 167, 38),
    6: (0, 150, 136),
}

COUNT_COLUMNS = [f"count_{name}" for name in CLASS_NAMES]
SHORT_COUNT_COLUMNS = CLASS_NAMES
