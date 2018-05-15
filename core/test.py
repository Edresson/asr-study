import keras.layers as keras_layers
import numpy as np
import keras.backend as K
import tensorflow as tf
 
# compatibility keras update
try:
    from keras import initializers
 
    initializations = initializers
except:
    from keras import initializations
 
from keras.layers import Bidirectional
from keras import activations, regularizers
from keras.regularizers import l1, l2
import keras.layers as keras_layers
from keras.layers.recurrent import Recurrent
import logging
 
# LSTM costumise function
class LSTM(keras_layers.LSTM):
    def __init__(self, units, zoneout_h=0., zoneout_c=0., implementation=2,
                 layer_norm=None, mi=None, **kwargs):
        self.layer_norm = layer_norm
        self.mi = mi
        output_dim = units
        self.zoneout_c = zoneout_c
        self.zoneout_h = zoneout_h
        self.consume_less = implementation
        try:  # python3
            super().__init__(output_dim, **kwargs)
        except:  # python2
            super(LSTM, self).__init__(output_dim, **kwargs)
     
    def get_config(self):
        config = {'layer_norm': self.layer_norm,
                  'mi': self.mi,
                  'zoneout_h': self.zoneout_h,
                  'zoneout_c': self.zoneout_c
                  }
 
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

# keras LSTM: OK
b = Bidirectional(keras_layers.LSTM(256,return_sequences=True,
                                   kernel_regularizer=l2(1e-4),
                                   recurrent_regularizer=l2(1e-4),
                                   dropout=0,
                                   recurrent_dropout=0,
                                    activation='tanh'))
 
# customizer LSTM: Error
test =  Bidirectional(LSTM(256,return_sequences=True,
                                   kernel_regularizer=l2(1e-4),
                                   recurrent_regularizer=l2(1e-4),
                                   dropout=0,
                                   recurrent_dropout=0,
                                   zoneout_c=0,
                                   zoneout_h=0,
                                   mi=None,
                                   layer_norm=None,
                                    activation='tanh'))


