from keras.layers import Dense, Conv2D, Activation, Input, Flatten, AveragePooling2D, MaxPool2D, LeakyReLU, Dropout
from keras.models import Model
import numpy as np


class StandardConvNet(object):
    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=5, min_data_width=4,
                 hidden_activation="relu", output_activation="sigmoid",
                 pooling="mean", use_dropout=False, dropout_alpha=0.0,
                 optimizer="adam", loss="mse", leaky_alpha=0.1, batch_size=256, epochs=10, verbose=0):
        self.min_filters = min_filters
        self.filter_width = filter_width
        self.filter_growth_rate = filter_growth_rate
        self.min_data_width = min_data_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_dropout = use_dropout
        self.pooling = pooling
        self.dropout_alpha = dropout_alpha
        self.optimizer = optimizer
        self.loss = loss
        self.leaky_alpha = leaky_alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.verbose = verbose

    def build_network(self, input_shape, output_size):
        input_layer = Input(shape=input_shape, name="scn_input")
        num_conv_layers = int(np.log2(input_shape[1]) - np.log2(self.min_data_width))
        num_filters = self.min_filters
        scn_model = input_layer
        for c in range(num_conv_layers):
            scn_model = Conv2D(num_filters, (self.filter_width, self.filter_width),
                               padding="same", name="conv_{0:02d}".format(c))(scn_model)
            if self.hidden_activation == "leaky":
                scn_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(c))(scn_model)
            else:
                scn_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(c))(scn_model)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                scn_model = MaxPool2D(name="pooling_{0:02d}".format(c))(scn_model)
            else:
                scn_model = AveragePooling2D(name="pooling_{0:02d}".format(c))(scn_model)
        scn_model = Flatten(name="flatten")(scn_model)
        if self.use_dropout:
            scn_model = Dropout(self.dropout_alpha, name="dense_dropout")(scn_model)
        scn_model = Dense(output_size, name="dense_output")(scn_model)
        scn_model = Activation(self.output_activation, name="activation_output")(scn_model)
        scn_model_obj = Model(input_layer, scn_model)
        scn_model_obj.compile(self.optimizer, self.loss)
        self.model = scn_model_obj

    def fit(self, x, y):
        if len(x.shape) != 4:
            raise ValueError("Input data does not have dimensions (examples, y, x, predictor)")
        if len(y.shape) == 1:
            output_size = 1
        else:
            output_size = y.shape[1]
        self.build_network(x.shape[1:], output_size)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)

    def predict(self, x, y):
        return self.model.predict(x, y, batch_size=self.batch_size)


