import matplotlib.pyplot as plt
from preprocessing import Preprocessing
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, concatenate,Dropout,ConvLSTM2D, MaxPooling1D,Bidirectional,Conv1D, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import pickle
from keras.utils import plot_model
import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.engine import InputSpec

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

class AttentionDecoder(Recurrent):

    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space

        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(x)

    def get_initial_state(self, inputs):
        print('inputs shape:', inputs.get_shape())

        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1-zt)*stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Model_factory():
    def lstm(self, word_number, embedding_dim, outputdim, embedding_matrix):
        input_scene = Input(shape=(36,))
        input_before_sents = Input(shape=(32,))
        input_sents = Input(shape=(32,))
        '''
        embedding_layer_scene = Embedding(
            word_number,
            embedding_dim,
            input_length=16,
            weights=[embedding_matrix]
        )(input_scene)
        lstm_scene = LSTM(64, input_shape=(16, embedding_dim),implementation=1)(embedding_layer_scene)
        '''

        input_scene = Dense(32,activation='tanh')(input_scene)


        embedding_layer_bs = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix]
        )(input_before_sents)
        lstm_bs = LSTM(128, input_shape=(32, embedding_dim),implementation=1)(embedding_layer_bs)

        embedding_layer_sents = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix]
        )(input_sents)
        lstm_sents = LSTM(128, input_shape=(32, embedding_dim),implementation=1)(embedding_layer_sents)

        lstm_all = concatenate([input_scene,lstm_bs,lstm_sents])
        output = Dense(outputdim, activation='softmax')(lstm_all)
        model = Model(inputs=[input_scene,input_before_sents,input_sents],outputs=output)


        return model

    def lstm_without_embedding(self, word_number, embedding_dim, outputdim):
        input_scene = Input(shape=(16,))
        input_before_sents = Input(shape=(32,))
        input_sents = Input(shape=(32,))

        embedding_layer_scene = Embedding(
            word_number,
            embedding_dim,
            input_length=16,
            trainable=False

        )(input_scene)
        lstm_scene = LSTM(64, input_shape=(16, embedding_dim),implementation=1)(embedding_layer_scene)

        embedding_layer_bs = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            trainable=False

        )(input_before_sents)
        lstm_bs = LSTM(128, input_shape=(32, embedding_dim),implementation=1)(embedding_layer_bs)

        embedding_layer_sents = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            trainable=False
        )(input_sents)
        lstm_sents = LSTM(128, input_shape=(32, embedding_dim),implementation=1)(embedding_layer_sents)

        lstm_all = concatenate([lstm_scene,lstm_bs,lstm_sents])
        output = Dense(outputdim, activation='softmax')(lstm_all)
        model = Model(inputs=[input_scene,input_before_sents,input_sents],outputs=output)


        return model

    def lstm_beforechar(self, word_number, embedding_dim, outputdim, embedding_matrix):
        input_scene = Input(shape=(36,))
        input_before_sents = Input(shape=(32,))
        input_sents = Input(shape=(32,))
        input_before_char = Input(shape=(outputdim*3,))

        '''
         embedding_layer_scene = Embedding(
            word_number,
            embedding_dim,
            input_length=16,
            weights=[embedding_matrix]
        )(input_scene)
        lstm_scene = LSTM(64, input_shape=(16, embedding_dim),implementation=1)(embedding_layer_scene)       
        '''
        scene = Dense(32,input_shape=(36,),activation='tanh')(input_scene)
        scene = Dense(32,activation='tanh')(scene)
        before_char = Dense(16, input_shape=(outputdim*3,),activation='tanh')(input_before_char)
        before_char = Dense(16,activation='tanh')(before_char)


        embedding_layer_bs = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_before_sents)
        lstm_bs = LSTM(128, input_shape=(32, embedding_dim),implementation=1)(embedding_layer_bs)

        embedding_layer_sents = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_sents)
        lstm_sents = LSTM(128, input_shape=(32, embedding_dim),implementation=1)(embedding_layer_sents)



        lstm_all = concatenate([scene,lstm_bs,lstm_sents,before_char])
        output_2 = Dense(128,activation='relu')(lstm_all)
        output_2 = Dropout(0.2)(output_2)
        output_2 = Dense(32,activation='relu')(output_2)
        output_2 = Dropout(0.2)(output_2)
        output = Dense(outputdim, activation='softmax')(output_2)
        model = Model(inputs=[input_scene,input_before_sents,input_sents, input_before_char],outputs=output)


        return model

    def textCNN(self, word_number, embedding_dim, outputdim, embedding_matrix):
        input_scene = Input(shape=(36,))
        input_before_sents = Input(shape=(32,))
        input_sents = Input(shape=(32,))
        input_before_char = Input(shape=(outputdim*3,))

        '''
         embedding_layer_scene = Embedding(
            word_number,
            embedding_dim,
            input_length=16,
            weights=[embedding_matrix]
        )(input_scene)
        lstm_scene = LSTM(64, input_shape=(16, embedding_dim),implementation=1)(embedding_layer_scene)       
        '''
        scene = Dense(32,input_shape=(36,),activation='tanh')(input_scene)
        scene = Dense(32,activation='tanh')(scene)
        before_char = Dense(16, input_shape=(outputdim*3,),activation='tanh')(input_before_char)
        before_char = Dense(16,activation='tanh')(before_char)


        embedding_layer_bs = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_before_sents)
        lstm_bs = Conv1D(filters=32,kernel_size=3, padding="valid",activation='relu',input_shape=(32, embedding_dim))(embedding_layer_bs)
        lstm_bs = MaxPooling1D(pool_size=2,padding="valid")(lstm_bs)
        lstm_bs = BatchNormalization()(lstm_bs)
        lstm_bs = Conv1D(filters=5, kernel_size=3, padding="valid", activation='relu',input_shape=(32, embedding_dim))(lstm_bs)
        lstm_bs = MaxPooling1D(pool_size=2, padding="valid")(lstm_bs)
        lstm_bs = BatchNormalization()(lstm_bs)
        lstm_bs = Flatten()(lstm_bs)

        embedding_layer_sents = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_sents)
        lstm_sents =  Conv1D(filters=32,kernel_size=3,padding="valid", activation='relu',input_shape=(32, embedding_dim))(embedding_layer_sents)
        lstm_sents = MaxPooling1D(pool_size=2,padding="valid")(lstm_sents)
        lstm_sents = BatchNormalization()(lstm_sents)
        lstm_sents = Conv1D(filters=5,kernel_size=3, padding="valid", activation='relu',input_shape=(32, embedding_dim))(
            lstm_sents)
        lstm_sents = MaxPooling1D(pool_size=2, padding="valid")(lstm_sents)
        lstm_sents = BatchNormalization()(lstm_sents)
        lstm_sents = Flatten()(lstm_sents)



        lstm_all = concatenate([scene,lstm_bs,lstm_sents,before_char])
        output_2 = Dense(128,activation='relu')(lstm_all)
        output_2 = Dropout(0.1)(output_2)
        output_2 = Dense(32,activation='relu')(output_2)
        output_2 = Dropout(0.1)(output_2)
        output = Dense(outputdim, activation='softmax')(output_2)
        model = Model(inputs=[input_scene,input_before_sents,input_sents, input_before_char],outputs=output)
        model.summary()


        return model

    def textRNN(self, word_number, embedding_dim, outputdim, embedding_matrix):
        input_scene = Input(shape=(36,))
        input_before_sents = Input(shape=(32,))
        input_sents = Input(shape=(32,))
        input_before_char = Input(shape=(outputdim*3,))

        '''
         embedding_layer_scene = Embedding(
            word_number,
            embedding_dim,
            input_length=16,
            weights=[embedding_matrix]
        )(input_scene)
        lstm_scene = LSTM(64, input_shape=(16, embedding_dim),implementation=1)(embedding_layer_scene)       
        '''
        scene = Dense(32,input_shape=(36,),activation='tanh')(input_scene)
        scene = Dense(32,activation='tanh')(scene)
        before_char = Dense(16, input_shape=(outputdim*3,),activation='tanh')(input_before_char)
        before_char = Dense(16,activation='tanh')(before_char)


        embedding_layer_bs = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_before_sents)
        lstm_bs = Bidirectional(LSTM(128,return_sequences=True,implementation=1))(embedding_layer_bs)
        lstm_bs = Bidirectional(LSTM(64,return_sequences=True,implementation=1))(lstm_bs)
        lstm_bs = Flatten()(lstm_bs)

        embedding_layer_sents = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_sents)
        lstm_sents =  Bidirectional(LSTM(128,return_sequences=True,implementation=1))(embedding_layer_sents)
        lstm_sents = Bidirectional(LSTM(64,return_sequences=True,implementation=1))(lstm_sents)
        lstm_sents = Flatten()(lstm_sents)



        lstm_all = concatenate([scene,lstm_bs,lstm_sents,before_char])
        output_2 = Dense(128,activation='relu')(lstm_all)
        output_2 = Dropout(0.1)(output_2)
        output_2 = Dense(32,activation='relu')(output_2)
        output_2 = Dropout(0.1)(output_2)
        output = Dense(outputdim, activation='softmax')(output_2)
        model = Model(inputs=[input_scene,input_before_sents,input_sents, input_before_char],outputs=output)


        return model

    def textRNN_attention(self, word_number, embedding_dim, outputdim, embedding_matrix):
        input_scene = Input(shape=(36,))
        input_before_sents = Input(shape=(32,))
        input_sents = Input(shape=(32,))
        input_before_char = Input(shape=(outputdim*3,))

        '''
         embedding_layer_scene = Embedding(
            word_number,
            embedding_dim,
            input_length=16,
            weights=[embedding_matrix]
        )(input_scene)
        lstm_scene = LSTM(64, input_shape=(16, embedding_dim),implementation=1)(embedding_layer_scene)       
        '''
        scene = Dense(32,input_shape=(36,),activation='tanh')(input_scene)
        scene = Dense(32,activation='tanh')(scene)
        before_char = Dense(16, input_shape=(outputdim*3,),activation='tanh')(input_before_char)
        before_char = Dense(16,activation='tanh')(before_char)


        embedding_layer_bs = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_before_sents)
        lstm_bs = Bidirectional(LSTM(128,return_sequences=True,implementation=1))(embedding_layer_bs)
        lstm_bs = AttentionDecoder(64, 4, name='1')(lstm_bs)
        lstm_bs = Bidirectional(LSTM(64,return_sequences=True,implementation=1))(lstm_bs)
        lstm_bs = AttentionDecoder(32, 4, name='2')(lstm_bs)
        lstm_bs = Flatten()(lstm_bs)

        embedding_layer_sents = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_sents)
        lstm_sents =  Bidirectional(LSTM(128,return_sequences=True,implementation=1))(embedding_layer_sents)
        lstm_sents = AttentionDecoder(64,4, name='3')(lstm_sents)
        lstm_sents = Bidirectional(LSTM(64,return_sequences=True,implementation=1))(lstm_sents)
        lstm_sents = AttentionDecoder(64, 4, name='4')(lstm_sents)
        lstm_sents = Flatten()(lstm_sents)



        lstm_all = concatenate([scene,lstm_bs,lstm_sents,before_char])
        output_2 = Dense(128,activation='relu')(lstm_all)
        output_2 = Dropout(0.1)(output_2)
        output_2 = Dense(32,activation='relu')(output_2)
        output_2 = Dropout(0.1)(output_2)
        output = Dense(outputdim, activation='softmax')(output_2)
        model = Model(inputs=[input_scene,input_before_sents,input_sents, input_before_char],outputs=output)


        return model

    def textRCNN_attention(self, word_number, embedding_dim, outputdim, embedding_matrix):
        input_scene = Input(shape=(36,))
        input_before_sents = Input(shape=(32,))
        input_sents = Input(shape=(32,))
        input_before_char = Input(shape=(outputdim*3,))

        '''
         embedding_layer_scene = Embedding(
            word_number,
            embedding_dim,
            input_length=16,
            weights=[embedding_matrix]
        )(input_scene)
        lstm_scene = LSTM(64, input_shape=(16, embedding_dim),implementation=1)(embedding_layer_scene)       
        '''
        scene = Dense(32,input_shape=(36,),activation='tanh')(input_scene)
        scene = Dense(32,activation='tanh')(scene)
        before_char = Dense(16, input_shape=(outputdim*3,),activation='tanh')(input_before_char)
        before_char = Dense(16,activation='tanh')(before_char)


        embedding_layer_bs = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_before_sents)
        lstm_bs = Bidirectional(LSTM(128,return_sequences=True,implementation=1))(embedding_layer_bs)
        lstm_bs = AttentionDecoder(64, 4)(lstm_bs)
        lstm_bs = Bidirectional(LSTM(64,return_sequences=True,implementation=1))(lstm_bs)
        lstm_bs = AttentionDecoder(32, 4)(lstm_bs)
        lstm_bs = Flatten()(lstm_bs)

        embedding_layer_sents = Embedding(
            word_number,
            embedding_dim,
            input_length=32,
            weights=[embedding_matrix],
            trainable=False
        )(input_sents)
        lstm_sents =  Bidirectional(LSTM(128,return_sequences=True,implementation=1))(embedding_layer_sents)
        lstm_sents = AttentionDecoder(64,4)(lstm_sents)
        lstm_sents = Bidirectional(LSTM(64,return_sequences=True,implementation=1))(lstm_sents)
        lstm_sents = AttentionDecoder(64, 4)(lstm_sents)
        lstm_sents = Flatten()(lstm_sents)



        lstm_all = concatenate([scene,lstm_bs,lstm_sents,before_char])
        output_2 = Dense(128,activation='relu')(lstm_all)
        output_2 = Dropout(0.1)(output_2)
        output_2 = Dense(32,activation='relu')(output_2)
        output_2 = Dropout(0.1)(output_2)
        output = Dense(outputdim, activation='softmax')(output_2)
        model = Model(inputs=[input_scene,input_before_sents,input_sents, input_before_char],outputs=output)


        return model

if __name__ == '__main__':
    embedding_dim = 62

    preprocessing = Preprocessing()
    characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict = preprocessing.load_dataset()
    _, id2char, _, id2word, char_number, word_number,embedding_matrix = preprocessing.encoding_reduction(characters, word_dict)

    X, Y, X_test_2, Y_test_2 = preprocessing.generate_X_Y_split_beforechar(characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid,
                                      para_number, word_dict)
    print('data loaded')
    X_test_1 = [np.array(X[0][10000:20000]), np.array(X[1][10000:20000]), np.array(X[2][10000:20000]),np.array(X[3][10000:20000])]
    Y_test_1 = np.array(Y[10000:20000])
    print(X[0].shape)
    print(X_test_2[0].shape,X_test_2[1].shape,X_test_2[2].shape)
    print(Y_test_2.shape)
    print('train data loaded')
    model_factory = Model_factory()
    model = model_factory.textRNN_attention(
        word_number=word_number,
        embedding_dim=embedding_dim,
        outputdim=char_number,
        embedding_matrix=embedding_matrix
    )

    #model.load_weights('lstm-0102-new-beforechar.h5')
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='model.png')
    for i in range(1):
        model.fit(X, Y, batch_size=128,epochs=5,validation_split=0.1,shuffle=True)
    model.save_weights('lstm-0114-new-beforechar.h5')

    Y_predict = model.predict(X_test_1)
    a,_ = Y_predict.shape
    print(X_test_1[1].shape, Y_test_1.shape)
    correct = 0
    for i in range(a):
        y_hat = Y_predict[i,:]
        char_id = np.argmax(y_hat)
        y = Y_test_1[i, :]
        true_char_id = np.argmax(y)
        if char_id == true_char_id:
            correct += 1


    accuracy_1 = correct / a
    print('Accuracy: ', accuracy_1)

    Y_predict = model.predict(X_test_2)
    a,_ = Y_predict.shape
    print(X_test_2[1].shape, Y_test_2.shape)
    correct = 0
    correct_top2 = 0
    for i in range(a):
        y_hat = Y_predict[i,:]
        char_id = np.argmax(y_hat)
        y = Y_test_2[i, :]
        true_char_id = np.argmax(y)
        char_id_2 = np.argsort(y_hat)
        char_id_2 = char_id_2[-2:]
        if char_id == true_char_id:
            correct += 1
        if true_char_id in char_id_2:
            correct_top2 += 1


    accuracy = correct / a
    accuracy_3 = (accuracy_1 + accuracy) / 2
    accuracy_2 = correct_top2 / a
    print('Accuracy: ', accuracy)
    print('Accuracy of top2: ', accuracy_2)

'''
preprocessing = Preprocessing()
characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict = preprocessing.load_dataset()
_, id2char, _, id2word, char_number, word_number = preprocessing.encoding_reduction_onlychar(characters, word_dict)

X, Y = preprocessing.generate_X_Y_recudechar(characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid,
                                  para_number, word_dict)
'''
