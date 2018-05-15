# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np


import keras.backend as K
import tensorflow as tf

# compatibility keras update
try:
    from keras import initializers

    initializations = initializers
except:
    from keras import initializations

from keras import activations, regularizers

import keras.layers as keras_layers
from keras.layers.recurrent import Recurrent
from keras.engine import Layer, InputSpec

from core.layers_utils import highway_bias_initializer
from core.layers_utils import layer_normalization
from core.layers_utils import multiplicative_integration_init
from core.layers_utils import multiplicative_integration
from core.layers_utils import zoneout

from core.initializers import k_init

import logging

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec

from keras.utils.conv_utils import conv_output_length

import theano
import theano.tensor as T


def _dropout(x, level, noise_shape=None, seed=None):
    x = K.dropout(x, level, noise_shape, seed)
    x *= (1. - level) # compensate for the scaling by the dropout
    return x


class QRNN(Layer):
    '''Quasi RNN

    # Arguments
        units: dimension of the internal projections and the final output.

    # References
        - [Quasi-recurrent Neural Networks](http://arxiv.org/abs/1611.01576)
        
    Credits: https://github.com/DingKe/nn_playground
    '''
    def __init__(self, units, window_size=2, stride=1,
                 return_sequences=False, go_backwards=False, 
                 stateful=False, unroll=False, activation='tanh',
                 kernel_initializer='uniform', bias_initializer='zero',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, 
                 dropout=0, use_bias=True, input_dim=None, input_length=None,
                 **kwargs):
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.units = units 
        self.window_size = window_size
        self.strides = (stride, 1)

        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = dropout
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(QRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        kernel_shape = (self.window_size, 1, self.input_dim, self.units * 3)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', 
                                        shape=(self.units * 3,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        length = input_shape[1]
        if length:
            length = conv_output_length(length + self.window_size - 1,
                                        self.window_size, 'valid',
                                        self.strides[0])
        if self.return_sequences:
            return (input_shape[0], length, self.units)
        else:
            return (input_shape[0], self.units)

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called '
                               'and thus has no states.')

        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError('If a QRNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')

        if self.states[0] is None:
            self.states = [K.zeros((batch_size, self.units))
                           for _ in self.states]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 'state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, self.units)) +
                                     ', found shape=' + str(value.shape))
                K.set_value(state, value)

    def __call__(self, inputs, initial_state=None, **kwargs):
        # If `initial_state` is specified,
        # and if it a Keras tensor,
        # then add it to the inputs and temporarily
        # modify the input spec to include the state.
        if initial_state is not None:
            if hasattr(initial_state, '_keras_history'):
                # Compute the full input spec, including state
                input_spec = self.input_spec
                state_spec = self.state_spec
                if not isinstance(state_spec, list):
                    state_spec = [state_spec]
                self.input_spec = [input_spec] + state_spec

                # Compute the full inputs, including state
                if not isinstance(initial_state, (list, tuple)):
                    initial_state = [initial_state]
                inputs = [inputs] + list(initial_state)

                # Perform the call
                output = super(QRNN, self).__call__(inputs, **kwargs)

                # Restore original input spec
                self.input_spec = input_spec
                return output
            else:
                kwargs['initial_state'] = initial_state
        return super(QRNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, initial_state=None, training=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                            initial_states,
                                            go_backwards=self.go_backwards,
                                            mask=mask,
                                            constants=constants,
                                            unroll=self.unroll,
                                            input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout < 1:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, inputs, training=None):
        if self.window_size > 1:
            inputs = K.temporal_padding(inputs, (self.window_size-1, 0))
        inputs = K.expand_dims(inputs, 2)  # add a dummy dimension

        output = K.conv2d(inputs, self.kernel, strides=self.strides,
                          padding='valid',
                          data_format='channels_last')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.dropout is not None and 0. < self.dropout < 1.:
            z = output[:, :, :self.units]
            f = output[:, :, self.units:2 * self.units]
            o = output[:, :, 2 * self.units:]
            f = K.in_train_phase(1 - _dropout(1 - f, self.dropout), f, training=training)
            return K.concatenate([z, f, o], -1)
        else:
            return output

    def step(self, inputs, states):
        prev_output = states[0]

        z = inputs[:, :self.units]
        f = inputs[:, self.units:2 * self.units]
        o = inputs[:, 2 * self.units:]

        z = self.activation(z)
        f = f if self.dropout is not None and 0. < self.dropout < 1. else K.sigmoid(f)
        o = K.sigmoid(o)

        output = f * prev_output + (1 - f) * z
        output = o * output

        return output, [output]

    def get_constants(self, inputs, training=None):
        return []
 
    def get_config(self):
        config = {'units': self.units,
                  'window_size': self.window_size,
                  'stride': self.strides[0],
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'use_bias': self.use_bias,
                  'dropout': self.dropout,
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(QRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class LayerNormalization(Layer):
    '''Normalize from all of the summed inputs to the neurons in a layer on
    a single training case. Unlike batch normalization, layer normalization
    performs exactly the same computation at training and tests time.

    # Arguments
        epsilon: small float > 0. Fuzz parameter
        num_var: how many tensor are condensed in the input
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
            Note that the order of this list is [gain, bias]
        gain_init: name of initialization function for gain parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        bias_init: name of initialization function for bias parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
    # Input shape

    # Output shape
        Same shape as input.

    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
    '''
    def __init__(self, epsilon=1e-5, weights=None, gain_init='one',
                 bias_init='zero', **kwargs):
        self.epsilon = epsilon
        self.gain_init = initializations.get(gain_init)
        self.bias_init = initializations.get(bias_init)
        self.initial_weights = weights
        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[-1],)

        self.g = self.gain_init(shape, name='{}_gain'.format(self.name))
        self.b = self.bias_init(shape, name='{}_bias'.format(self.name))

        self.trainable_weights = [self.g, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):
        return LN(x, self.g, self.b, epsilon=self.epsilon)

    def get_config(self):
        config = {"epsilon": self.epsilon,
                  'num_var': self.num_var,
                  'gain_init': self.gain_init.__name__,
                  'bias_init': self.bias_init.__name__}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RHN(Recurrent):
    '''Recurrent Highway Network - Julian Georg Zilly, Rupesh Kumar Srivastava,
    Jan Koutník, Jürgen Schmidhuber - 2016.
    For a step-by-step description of the network, see
    [this paper](https://arxiv.org/abs/1607.03474).
    # Arguments
        units: dimension of the internal projections and the final output.
        depth: recurrency depth size.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see:
            [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        bias_init: initialization function of the bias.
            (see [this
            post](http://people.idsia.ch/~rupesh/very_deep_learning/)
            for more information)
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        recurrent_activation: activation function for the inner cells.
        coupling: if True, carry gate will be coupled to the transform gate,
            i.e., c = 1 - t
        kernel_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights
            matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights
            matrices.
        bias_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop
        for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop
        for recurrent connections.
    # References
        - [Recurrent Highway Networks](https://arxiv.org/abs/1607.03474)
        (original paper)
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural
        Networks](http://arxiv.org/abs/1512.05287)
    # TODO: different dropout rates for each layer
    '''
    def __init__(self, units, depth=1,
                 init='glorot_uniform', inner_init='orthogonal',
                 bias_init=highway_bias_initializer,
                 activation='tanh', recurrent_activation='hard_sigmoid',
                 coupling=True, layer_norm=False, ln_gain_init='one',
                 ln_bias_init='zero', mi=False,
                 kernel_regularizer=None, U_regularizer=None,
                 bias_regularizer=None, dropout_W=0., dropout_U=0., **kwargs):
        self.units = units
        self.depth = depth
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.bias_init = initializations.get(bias_init)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.coupling = coupling
        self.has_layer_norm = layer_norm
        self.ln_gain_init = initializations.get(ln_gain_init)
        self.ln_bias_init = initializations.get(ln_bias_init)
        self.mi = mi
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True

        super(RHN, self).__init__(**kwargs)

        if not self.consume_less == "gpu":
            self._logger.warning("Ignoring consume_less=%s. Setting to 'gpu'." % self.consume_less)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            self.states = [None]

        self.W = self.init((self.input_dim, (2 + (not self.coupling)) *
                            self.units), name='{}_W'.format(self.name))
        self.Us = [self.inner_init(
            (self.units, (2 + (not self.coupling)) * self.units),
            name='%s_%d_U' % (self.name, i)) for i in xrange(self.depth)]

        bias_init_value = K.get_value(self.bias_init((self.units,)))
        b = [np.zeros(self.units),
             np.copy(bias_init_value)]

        if not self.coupling:
            b.append(np.copy(bias_init_value))

        self.bs = [K.variable(np.hstack(b),
                              name='%s_%d_b' % (self.name, i)) for i in
                   xrange(self.depth)]

        self.trainable_weights = [self.W] + self.Us + self.bs

        if self.mi:
            self.mi_params = [multiplicative_integration_init(
                ((2 + (not self.coupling)) * self.units,),
                name='%s_%d' % (self.name, i),
                has_input=(i == 0)) for i in xrange(self.depth)]

            for p in self.mi_params:
                if type(p) in {list, tuple}:
                    self.trainable_weights += p
                else:
                    self.trainable_weights += [p]

        if self.has_layer_norm:
            self.ln_weights = []
            ln_names = ['h', 't', 'c']
            for l in xrange(self.depth):

                ln_gains = [self.ln_gain_init(
                    (self.units,), name='%s_%d_ln_gain_%s' %
                    (self.name, l, ln_names[i])) for i in xrange(1)]

                ln_biases = [self.ln_bias_init(
                    (self.units,), name='%s_%d_ln_bias_%s' %
                    (self.name, l, ln_names[i])) for i in xrange(1)]
                self.ln_weights.append([ln_gains, ln_biases])
                self.trainable_weights += ln_gains + ln_biases

        self.regularizers = []
        if self.kernel_regularizer:
            self.kernel_regularizer.set_param(self.W)
            self.regularizers.append(self.kernel_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.bias_regularizer:
            self.bias_regularizer.set_param(self.b)
            self.regularizers.append(self.bias_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch \
                            size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
        else:
            self.states = [K.zeros((input_shape[0], self.units))]

    def step(self, x, states):
        s_tm1 = states[0]

        for layer in xrange(self.depth):
            B_U = states[layer + 1][0]
            U, b = self.Us[layer], self.bs[layer]

            if layer == 0:
                B_W = states[layer + 1][1]
                Wx = K.dot(x * B_W, self.W)
            else:
                Wx = 0

            Us = K.dot(s_tm1 * B_U, U)

            if self.mi:
                a = multiplicative_integration(Wx, Us,
                                               self.mi_params[layer]) + b
            else:
                a = Wx + Us + b

            a0 = a[:, :self.units]
            a1 = a[:, self.units: 2 * self.units]
            if not self.coupling:
                a2 = a[:, 2 * self.units:]

            if self.has_layer_norm:
                ln_gains, ln_biases = self.ln_weights[layer]
                a0 = LN(a0, ln_gains[0], ln_biases[0])
                # a1 = LN(a1, ln_gains[1], ln_biases[1])
                # if not self.coupling:
                #     a2 = LN(a2, ln_gains[2], ln_biases[2])

            # Equation 7
            h = self.activation(a0)
            # Equation 8
            t = self.recurrent_activation(a1)
            # Equation 9
            if not self.coupling:
                c = self.recurrent_activation(a2)
            else:
                c = 1 - t  # carry gate was coupled to the transform gate

            s = h * t + s_tm1 * c
            s_tm1 = s

        return s, [s]

    def get_constants(self, x):
        constants = []

        for layer in xrange(self.depth):
            constant = []
            if 0 < self.dropout_U < 1:
                ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.units))
                B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
                constant.append(B_U)
            else:
                constant.append(K.cast_to_floatx(1.))

            if layer == 0:
                if 0 < self.dropout_W < 1:
                    input_shape = self.input_spec[0].shape
                    input_dim = input_shape[-1]
                    ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
                    ones = K.tile(ones, (1, input_dim))
                    B_W = K.in_train_phase(K.dropout(ones,
                                                     self.dropout_W), ones)
                    constant.append(B_W)
                else:
                    constant.append(K.cast_to_floatx(1.))

            constants.append(constant)

        return constants

    def get_config(self):
        config = {'units': self.units,
                  'depth': self.depth,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'bias_init': self.bias_init.__name__,
                  'activation': self.activation.__name__,
                  'recurrent_activation': self.recurrent_activation.__name__,
                  'coupling': self.coupling,
                  'layer_norm': self.has_layer_norm,
                  'ln_gain_init': self.ln_gain_init.__name__,
                  'ln_bias_init': self.ln_bias_init.__name__,
                  'mi': self.mi,
                  'kernel_regularizer': self.kernel_regularizer.get_config() if
                  self.kernel_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if
                  self.U_regularizer else None,
                  'bias_regularizer': self.bias_regularizer.get_config() if
                  self.bias_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(RHN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTM(keras_layers.LSTM):
    """
    # Arguments

        ln: None, list of float or list of list of floats. Determines whether will apply LN or not. If list of floats, the same init will be applied to every LN; otherwise will be individual
        mi: list of floats or None. If list of floats, the multiplicative integration will be active and initialized with these values.
        zoneout_h: float between 0 and 1. Fraction of the hidden/output units to maintain their previous values.
        zoneout_c: float between 0 and 1. Fraction of the cell units to maintain their previous values.
    # References
        - [Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations](https://arxiv.org/abs/1606.01305)
    """
    def __init__(self, units, zoneout_h=0., zoneout_c=0.,implementation = 2 ,
                 layer_norm=None, mi=None, **kwargs):
        '''implementation Argument is consume_less in Keras 1 values:
         implementation= 1 :cpu
         implementation = 2 :gpu
         '''
        try:#python3
            super().__init__(units, **kwargs)
        except:#python2
            super(LSTM, self).__init__(units, **kwargs)

        self._logger = logging.getLogger('%s.%s' % (__name__,
                                                    self.__class__.__name__))
        
        self.layer_norm = layer_norm
        self.mi = mi

        self.zoneout_c = zoneout_c
        self.zoneout_h = zoneout_h
         

        if self.zoneout_h or self.zoneout_c:
            self.uses_learning_phase = True
            

    def build(self, input_shape):
        
        try:#python3
            super().build(input_shape)
        except:#python2
            super(LSTM, self).build(input_shape)

        """ ToDo:  reimplement

        if self.mi is not None:
            alpha_init, beta1_init, beta2_init = self.mi

            self.mi_alpha = self.add_weight(
                shape=(4 * self.units, ),
                initializer=k_init(alpha_init),
                name='{}_mi_alpha'.format(self.name))
            self.mi_beta1 = self.add_weight(
                shape=(4 * self.units, ),
                initializer=k_init(beta1_init),
                name='{}_mi_beta1'.format(self.name))
            self.mi_beta2 = self.add_weight(
                shape=(4 * self.units, ),
                initializer=k_init(beta2_init),
                name='{}_mi_beta2'.format(self.name))

        if self.layer_norm is not None:
            ln_gain_init, ln_bias_init = self.layer_norm

            self.layer_norm_params = {}
            for n, i in {'Uh': 4, 'Wx': 4, 'new_c': 1}.items():

                gain = self.add_weight(
                    shape=(i*self.units, ),
                    initializer=k_init(ln_gain_init),
                    name='%s_ln_gain_%s' % (self.name, n))
                bias = self.add_weight(
                    shape=(i*self.units, ),
                    initializer=k_init(ln_bias_init),
                    name='%s_ln_bias_%s' % (self.name, n))

                self.layer_norm_params[n] = [gain, bias]

    def _layer_norm(self, x, param_name):
        if self.layer_norm is None:
            return x

        gain, bias = self.layer_norm_params[param_name]

        return layer_normalization(x, gain, bias)"""

    """ToDo:  reimplement
    def step(self, inputs, states):
        
        x = inputs
        
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        
        # self.U to self.recurrent_kernel
            
        Uh = self._layer_norm(K.dot(h_tm1 * B_U[0], self.recurrent_kernel), 'Uh')

        # self.W to self.kernel
        Wx = self._layer_norm(K.dot(x * B_W[0], self.kernel), 'Wx')

        
        
        if self.mi is not None:
            z = self.mi_alpha * Wx * Uh + self.mi_beta1 * Uh + \
                self.mi_beta2 * Wx + self.bias
            
        else:
            z = Wx + Uh + self.bias

        z_i = z[:, :self.units]
        z_f = z[:, self.units: 2 * self.units]
        z_c = z[:, 2 * self.units: 3 * self.units]
        z_o = z[:, 3 * self.units:]

        i = self.recurrent_activation(z_i)
        f = self.recurrent_activation(z_f)
        c = f * c_tm1 + i * self.activation(z_c)
        o = self.recurrent_activation(z_o)

        if 0 < self.zoneout_c < 1:
            c = zoneout(self.zoneout_c, c_tm1, c,
                        noise_shape=(self.units,))

        # this is returning a lot of nan
        new_c = self._layer_norm(c, 'new_c')

        h = o * self.activation(new_c)
        if 0 < self.zoneout_h < 1:
            h = zoneout(self.zoneout_h, h_tm1, h,
                        noise_shape=(self.units,))

        return h, [h, c]

    def get_config(self):
        config = {'layer_norm': self.layer_norm,
                  'mi': self.mi,
                  'zoneout_h': self.zoneout_h,
                  'zoneout_c': self.zoneout_c
                  }



        base_config = super(LSTM, self).get_config()


        return dict(list(base_config.items()) + list(config.items()))"""
    



def recurrent(units, model='keras_lstm', activation='tanh',
              regularizer=None, dropout=0., **kwargs):
    if model == 'rnn':
        return keras_layers.SimpleRNN(units, activation=activation,
                                      kernel_regularizer=regularizer,
                                      U_regularizer=regularizer,
                                      dropout_W=dropout, dropout_U=dropout, consume_less='gpu',
                                      **kwargs)
    if model == 'gru':
        return keras_layers.GRU(units, activation=activation,
                                kernel_regularizer=regularizer,
                                U_regularizer=regularizer, dropout_W=dropout,
                                dropout_U=dropout,
                                consume_less='gpu', **kwargs)
    if model == 'keras_lstm':
        return keras_layers.LSTM(units, activation=activation,
                                 kernel_regularizer=regularizer,
                                 U_regularizer=regularizer,
                                 dropout_W=dropout, dropout_U=dropout,
                                 consume_less='gpu', **kwargs)
    if model == 'rhn':
        return RHN(units, depth=1,
                   bias_init=highway_bias_initializer,
                   activation=activation, layer_norm=False, ln_gain_init='one',
                   ln_bias_init='zero', mi=False,
                   kernel_regularizer=regularizer, U_regularizer=regularizer,
                   dropout_W=dropout, dropout_U=dropout, consume_less='gpu',
                   **kwargs)

    if model == 'lstm':
        return LSTM(units, activation=activation,
                    kernel_regularizer=regularizer, U_regularizer=regularizer,
                    dropout_W=dropout, dropout_U=dropout,
                    consume_less='gpu', **kwargs)
    raise ValueError('model %s was not recognized' % model)


if __name__ == "__main__":
    from keras.models import Sequential
    #from keras.utils.visualize_util import plot

    model = Sequential()
    #model.add(RHN(10, input_dim=2, depth=2, layer_norm=True))
    model.add(LSTM( 128))
    # plot(model)
