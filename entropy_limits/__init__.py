import numpy as np


def ar_min_entropy_limit(beta):
    return -np.log2(1 - beta / 2)
