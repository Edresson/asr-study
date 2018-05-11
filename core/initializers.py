import numpy as np

from tensorflow.python.keras import backend as K



def k_init(k):
    def init(shape, name=None):
        return K.variable(k*np.ones(shape), dtype='float32',
                   name=name)
    return init
