import const

import torch
import numpy as np

import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
#from keras import backend as K


class DQN():

    model_path = const.file_path_model

    def __init__(self, num_inputs, num_outputs, lr, b):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = lr
        self.batch_size = b
        #self.model = self._build_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(1024, input_dim=self.state_size, activation='elu'))
        model.add(Dense(512, activation='elu'))
        model.add(Dense(self.action_size, activation='linear'))
        loss_foo = 'mse' #self._huber_loss
        model.compile(loss=loss_foo,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def fit(self, X, y, batch_size=1, epochs=1, verbose=0):
        pass

    def predict(self, X, batch_size=1):
        return np.random.random(size=(batch_size, self.num_outputs))

    def get_weights(self):
        pass

    def set_weights(self, W):
        pass

    def save_weights(self, fp=None):
        pass

    def load_weights(self, fp=None):
        pass
