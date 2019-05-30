import enum


class LabelType(enum.Enum):
    ONE_LABELS_TO_ALL_TASK = 0
    ONE_LABEL_TO_ONE_CLASS = 1


class UnitType(enum.Enum):
    FILTER = 0
    NEURON = 1
    NONE = 2

    def __str__(self):
        return self.name


class MaskType(enum.Enum):
    ADAPTIVE = 0
    HARD = 1
    INDEPENDENT = 2

    def __str__(self):
        return self.name
