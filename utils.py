import numpy as np
from tables import *


def read_train_data(f, start, end):
    x = f.root.utterance[start:end]
    y = f.root.label[start:end]
    return x, y


def read_valid_data(f, start, end):
    x = f['utterance'][start:end]
    y = f['label'][start:end]
    return x, y



