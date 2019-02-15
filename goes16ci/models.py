from keras.layers import Dense, Conv2D, Activation, Input, Flatten, AveragePooling2D, MaxPool2D, LeakyReLU, Dropout
from keras.models import Model
import keras.backend as K
from keras.utils import multi_gpu_model
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
        self.model = scn_model_obj

    def compile_model(self):
        self.model.compile(self.optimizer, self.loss)

    @staticmethod
    def get_data_shapes(x, y):
        if len(x.shape) != 4:
            raise ValueError("Input data does not have dimensions (examples, y, x, predictor)")
        if len(y.shape) == 1:
            output_size = 1
        else:
            output_size = y.shape[1]
        return x.shape[1:], output_size

    def fit(self, x, y, val_x=None, val_y=None, build=True):
        if build:
            x_shape, y_size = self.get_data_shapes(x, y)
            self.build_network(x_shape, y_size)
            self.compile_model()
        if val_x is None:
            val_data = None
        else:
            val_data = (val_x, val_y)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       validation_data=val_data)

    def predict(self, x, y):
        return self.model.predict(x, y, batch_size=self.batch_size)


def train_conv_net_cpu(train_data, train_labels, val_data, val_labels,
                       conv_net_hyperparameters, num_processors, seed):
    np.random.seed(seed)
    K.tf.set_random_seed(seed)
    sess = K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=False, intra_op_parallelism_threads=1,
                                                inter_op_parallelism_threads=num_processors))
    K.set_session(sess)

    with K.tf.device("cpu:0"):
        scn = StandardConvNet(**conv_net_hyperparameters)
        scn.fit(train_data, train_labels, val_x=val_data, val_y=val_labels)
    sess.close()
    del sess
    return


def train_conv_net_gpu(train_data, train_labels, val_data, val_labels,
                       conv_net_hyperparameters, num_gpus, seed, cpu_relocation=True, cpu_merge=False):
    np.random.seed(seed)
    K.tf.set_random_seed(seed)
    config = K.tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    sess = K.tf.Session(config=config)
    K.set_session(sess)
    if num_gpus == 1:
        with K.tf.device("gpu:0"):
            scn = StandardConvNet(**conv_net_hyperparameters)
            scn.fit(train_data, train_labels, val_x=val_data, val_y=val_labels)
    elif num_gpus > 1:
        with K.tf.device("cpu:0"):
            scn = StandardConvNet(**conv_net_hyperparameters)
            x_shape, y_size = scn.get_data_shapes(train_data, train_labels)
            scn.build_network(x_shape, y_size)
        scn.model = multi_gpu_model(scn.model, gpus=num_gpus, cpu_merge=cpu_merge, cpu_relocation=cpu_relocation)
        scn.compile_model()
        scn.fit(train_data, train_labels, val_x=val_data, val_y=val_labels)
    sess.close()
    del sess
    return