import enum


class LabelType(enum.Enum):
    ONE_LABELS_TO_ALL_TASK = 0
    ONE_LABEL_TO_ONE_CLASS = 1


class UnitType(enum.Enum):
    FILTER = 0
    NEURON = 1

    def __str__(self):
        return self.name
