import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Activation, Input, Flatten, AveragePooling2D, MaxPool2D, LeakyReLU, Dropout, Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, hinge_loss, mean_squared_error
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
from time import perf_counter
import logging
import csv
import optuna
from datetime import datetime
from sklearn.utils import class_weight
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
#from aimlutils.hyper_opt.utils import KerasPruningCallback


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        self.val_losses.append(logs.get("val_loss"))
        
class TimeHistory(Callback):
    def __init__(self):
        self.times = []
        self.epoch_time_start = 0
        super(TimeHistory, self).__init__()
        
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(perf_counter() - self.epoch_time_start)
        

class StandardConvNet(object):
    """
    Standard Convolutional Neural Network contains a series of convolution and pooling layers followed by one
    fully connected layer to a set of scalar outputs. The number of convolution filters is assumed to increase
    with depth.
    
    Attributes:
        min_filters (int): The number of convolution filters in the first layer. 
        filter_growth_rate (float): Multiplier on the number of convolution filters between layers.
        min_data_width (int): The minimum dimension of the input data after the final pooling layer. Constrains the number of 
            convolutional layers.
        hidden_activation (str): The nonlinear activation function applied after each convolutional layer. If "leaky", a leaky ReLU with
            alpha=0.1 is used.
        output_activation (str): The nonlinear activation function applied on the output layer.
        pooling (str): If mean, then :class:`keras.layers.AveragePooling2D` is used for pooling. If max, then :class:`keras.layers.MaxPool2D` is used.
        use_dropout (bool): If True, then a :class:`keras.layers.Dropout` layer is inserted between the final convolution block 
            and the output :class:`keras.laysers.Dense` layer.
        dropout_alpha (float): Dropout rate ranging from 0 to 1.
        data_format (str): "channels_last" or "channels_first"
        optimizer (str): if "adam" uses Adam, otherwise uses SGD.
        loss (str): one of the built in keras losses
        leaky_alpha (float): scaling factor for LeakyReLU activation
    """
    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=5, min_data_width=4,
                 hidden_activation="relu", output_activation="sigmoid",
                 pooling="mean", use_dropout=False, dropout_alpha=0.0,
                 data_format="channels_last", optimizer="adam", loss="mse", leaky_alpha=0.1, metrics=None, 
                 learning_rate=0.001, batch_size=1024, epochs=10, verbose=0, sgd_momentum=0.99):
        self.min_filters = min_filters
        self.filter_width = filter_width
        self.filter_growth_rate = filter_growth_rate
        self.min_data_width = min_data_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_dropout = use_dropout
        self.pooling = pooling
        self.dropout_alpha = dropout_alpha
        self.data_format = data_format
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.leaky_alpha = leaky_alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.sgd_momentum = sgd_momentum
        self.model = None
        self.parallel_model = None
        self.verbose = verbose
        self.time_history = TimeHistory()
        self.loss_history = LossHistory()
        self.early_stopping = EarlyStopping(monitor='val_loss')
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss')
        
    def build_network(self, input_shape, output_size, interval=1, trial=None):
        """
        Create a keras model with the hyperparameters specified in the constructor.
        Args:
            input_shape (tuple of shape [variable, y, x]): The shape of the input data
            output_size: Number of neurons in output layer.
        """
        #self.pruning_callback = KerasPruningCallback(trial,'val_loss',interval=interval)
        input_layer = Input(shape=input_shape, name="scn_input")
        num_conv_layers = int(np.log2(input_shape[1]) - np.log2(self.min_data_width))
        num_filters = self.min_filters
        scn_model = input_layer
        for c in range(num_conv_layers):
            scn_model = Conv2D(num_filters, (self.filter_width, self.filter_width),
                               data_format=self.data_format, padding="same", name="conv_{0:02d}".format(c))(scn_model)
            if self.hidden_activation == "leaky":
                scn_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(c))(scn_model)
            else:
                scn_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(c))(scn_model)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                scn_model = MaxPool2D(data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
            else:
                scn_model = AveragePooling2D(data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
        scn_model = Flatten(name="flatten")(scn_model)
        if self.use_dropout:
            scn_model = Dropout(self.dropout_alpha, name="dense_dropout")(scn_model)
        scn_model = Dense(output_size, name="dense_output")(scn_model)
        scn_model = Activation(self.output_activation, name="activation_output")(scn_model)
        self.model = Model(input_layer, scn_model)

    def compile_model(self):
        """
        Compile the model in tensorflow with the right optimizer and loss function.
        """
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        self.model.compile(opt, self.loss, metrics=self.metrics)

    def compile_parallel_model(self):
        if self.optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        self.parallel_model.compile(opt, self.loss, metrics=self.metrics)

    @staticmethod
    def get_data_shapes(x, y):
        """
        Extract the input and output data shapes in order to construct the neural network.
        """
        if len(x.shape) != 4:
            raise ValueError("Input data does not have dimensions (examples, y, x, predictor)")
        if len(y.shape) == 1:
            output_size = 1
        else:
            output_size = y.shape[1]
        return x.shape[1:], output_size

    def fit(self, x, y, val_x=None, val_y=None, build=True):
        """
        Train the neural network.
        """
        if build:
            x_shape, y_size = self.get_data_shapes(x, y)
            self.build_network(x_shape, y_size)
            self.compile_model()
        if val_x is None:
            val_data = None
        else:
            val_data = (val_x, val_y)

        class_weights = {0: 1.0, 1: 2.0}
        history = self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       validation_data=val_data, callbacks=[self.time_history, self.loss_history, self.early_stopping, self.reduce_lr],class_weight=class_weights)
        return history

    def predict(self, x):
        return self.model.predict(x, batch_size=self.batch_size)


class ResNet(StandardConvNet):
    """
    Extension of the :class:`goes16ci.models.StandardConvNet` to include Residual layers instead of single convolutional layers.
    The residual layers split the data signal off, apply normalization and convolutions to it, then adds it back on to the original field.
    """
    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=3, min_data_width=4,
                 hidden_activation="relu", output_activation="sigmoid", metrics=None,
                 pooling="mean", use_dropout=False, dropout_alpha=0.0, data_format="channels_last", learning_rate=0.001,
                 optimizer="adam", loss="mse", leaky_alpha=0.1, batch_size=1024, epochs=10, verbose=0):
        super().__init__(min_filters=min_filters, filter_growth_rate=filter_growth_rate, filter_width=filter_width,
                         min_data_width=min_data_width, hidden_activation=hidden_activation, data_format=data_format,
                         output_activation=output_activation, pooling=pooling, use_dropout=use_dropout,
                         dropout_alpha=dropout_alpha, optimizer=optimizer, loss=loss, metrics=metrics, leaky_alpha=leaky_alpha,
                         batch_size=batch_size, epochs=epochs, verbose=verbose, learning_rate=learning_rate)

    def residual_block(self, filters, in_layer, layer_number=0):
        """
        Generate a single residual block.
        """
        if self.data_format == "channels_first":
            norm_axis = 1
        else:
            norm_axis = -1
        if in_layer.shape[-1] != filters:
            x = Conv2D(filters, self.filter_width, data_format=self.data_format, padding="same")(in_layer)
        else:
            x = in_layer
        y = BatchNormalization(axis=norm_axis, name="bn_res_{0:02d}_a".format(layer_number))(x)
        if self.hidden_activation == "leaky":
            y = LeakyReLU(self.leaky_alpha, name="res_activation_{0:02d}_a".format(layer_number))(y)
        else:
            y = Activation(self.hidden_activation,
                           name="res_activation_{0:02d}_a".format(layer_number))(y)
        y = Conv2D(filters, self.filter_width, padding="same",
                   data_format=self.data_format, name="res_conv_{0:02d}_a".format(layer_number))(y)
        y = BatchNormalization(axis=norm_axis, name="bn_res_{0:02d}_b".format(layer_number))(y)
        if self.hidden_activation == "leaky":
            y = LeakyReLU(self.leaky_alpha, name="res_activation_{0:02d}_b".format(layer_number))(y)
        else:
            y = Activation(self.hidden_activation,
                           name="res_activation_{0:02d}_b".format(layer_number))(y)
        y = Conv2D(filters, self.filter_width, padding="same",
                   data_format=self.data_format, name="res_conv_{0:02d}_b".format(layer_number))(y)
        out = Add()([y, x])
        return out

    def build_network(self, input_shape, output_size, interval=1, trial=None):
        #self.pruning_callback = KerasPruningCallback(trial,'val_loss',interval=interval)
        input_layer = Input(shape=input_shape, name="scn_input")
        num_conv_layers = int(np.log2(input_shape[1]) - np.log2(self.min_data_width))
        num_filters = self.min_filters
        res_model = input_layer
        for c in range(num_conv_layers):
            res_model = self.residual_block(num_filters, res_model, c)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                res_model = MaxPool2D(data_format=self.data_format, name="pooling_{0:02d}".format(c))(res_model)
            else:
                res_model = AveragePooling2D(data_format=self.data_format, name="pooling_{0:02d}".format(c))(res_model)
        res_model = Flatten(name="flatten")(res_model)
        if self.use_dropout:
            res_model = Dropout(self.dropout_alpha, name="dense_dropout")(res_model)
        res_model = Dense(output_size, name="dense_output")(res_model)
        res_model = Activation(self.output_activation, name="activation_output")(res_model)
        self.model = Model(input_layer, res_model)


def train_conv_net_cpu(train_data, train_labels, val_data, val_labels,
                       conv_net_hyperparameters, num_processors, seed, dtype="float32", inter_op_threads=2,trial=None):
    """
    Train a convolutional neural network on the CPU.
    """
    np.random.seed(seed)
    if "get_visible_devices" in dir(tf.config.experimental):
        gpus = tf.config.experimental.get_visible_devices("GPU")
    else:
        gpus = tf.config.get_visible_devices("GPU")
    if len(gpus) > 0:
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, True)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_processors)
    if tf.__version__[0] == "1":
        tf.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)
    K.set_floatx(dtype)
    with tf.device("/CPU:0"):
        scn = ResNet(**conv_net_hyperparameters)
        history = scn.fit(train_data, train_labels, val_x=val_data, val_y=val_labels)
        #scn.fit(train_data, train_labels, val_x=val_data, val_y=val_labels)
        time_str = pd.Timestamp.now().strftime("%d/%m/%Y %I:%M:%S")
        with open('AUC_history.csv','w') as f:
            for key in history.history.keys():
                f.write("%s,%s\n"%(key,history.history[key]))
        epoch_times = scn.time_history.times
        batch_loss = np.array(scn.loss_history.losses).ravel().tolist()
        epoch_loss = np.array(scn.loss_history.val_losses).ravel().tolist()
    date_time = str(datetime.now())
    save_model(scn.model, 'goes16ci_model_cpu' + date_time + '.h5', save_format = 'h5')
    if epoch_loss[-1] == 0.0:
        print("History == 0", history)
        raise optuna.TrialPruned()
    return epoch_times, batch_loss, epoch_loss


def train_conv_net_gpu(train_data, train_labels, val_data, val_labels,
                       conv_net_hyperparameters, num_gpus, seed,
                       dtype="float32", scale_batch_size=1,trial=None):
    """
    Trains convolutional neural network on one or more GPUs.
    Args:
        train_data: array of training data inputs as a data cube
        train_labels: array of labels associated with each training example
        val_data: array of validation data inputs
        val_labels: array of validation data labels
        conv_net_hyperparameters: dictionary of configuration settings for conv net
        num_gpus: Maximum number of GPUs to test
        seed: Random seed for both numpy and tensorflow
        dtype: float datatype for neural nets
    """
    np.random.seed(seed)
    if "get_visible_devices" in dir(tf.config.experimental):
        gpus = tf.config.experimental.get_visible_devices("GPU")
    else:
        gpus = tf.config.get_visible_devices("GPU")
    if num_gpus <= len(gpus):
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, True)
        if tf.__version__[0] == "1":
            tf.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)
        K.set_floatx(dtype)
        if num_gpus == 1:
            with tf.device("/device:GPU:0"):
                scn = ResNet(**conv_net_hyperparameters)
                history = scn.fit(train_data, train_labels, val_x=val_data, val_y=val_labels)
                #scn.fit(train_data, train_labels, val_x=val_data, val_y=val_labels)
                time_str = pd.Timestamp.now().strftime("%d/%m/%Y %I:%M:%S")
                with open('AUC_history.csv','w') as f:
                    for key in history.history.keys():
                        f.write("%s,%s\n"%(key,history.history[key]))
                epoch_times = scn.time_history.times
                batch_loss = np.array(scn.loss_history.losses).ravel().tolist()
                epoch_loss = np.array(scn.loss_history.val_losses).ravel().tolist()
                logging.info(scn.model.summary())
                if epoch_loss[-1] == 0.0:
                    logging.warning(f"History == 0.0 {history.history}")
                    raise OSError("Died because val_loss was 0")
                if scn is not None:
                    save_model(scn.model, "goes16_resnet_gpus_{0:02d}.h5".format(num_gpus))
        elif num_gpus > 1: 
            gpu_devices = [gpu.name.replace("physical_", "") for gpu in gpus[:num_gpus]]
            print("GPU Devices", gpu_devices, num_gpus)
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
            with mirrored_strategy.scope():
                scn = ResNet(**conv_net_hyperparameters)
                if scale_batch_size > 0:
                    scn.batch_size *= num_gpus
                    scn.learning_rate *= num_gpus
                x_shape, y_size = scn.get_data_shapes(train_data, train_labels)
                scn.build_network(x_shape, y_size, trial)
                scn.compile_model()
                history = scn.fit(train_data, train_labels)
                #scn.fit(train_data, train_labels)
                time_str = pd.Timestamp.now().strftime("%d/%m/%Y %I:%M:%S")
                with open('AUC_history.csv','w') as f:
                    for key in history.history.keys():
                        f.write("%s,%s\n"%(key,history.history[key]))
                if scn is not None:
                    save_model(scn.model, "goes16_resnet_gpus_{0:02d}.h5".format(num_gpus), save_format="h5")
                logging.info(scn.model.summary())
                epoch_times = scn.time_history.times
                batch_loss = np.array(scn.loss_history.losses).ravel().tolist()
                epoch_loss = np.array(scn.loss_history.val_losses).ravel().tolist()
                if epoch_loss[-1] == 0.0:
                    logger.warning(f"History == 0.0 {history.history}")
                    raise OSError("Died because val_loss was 0")
    else:
        print("No GPUs available")
        epoch_times = [-1]
        batch_loss = [-1]
        epoch_loss = [-1]
    return epoch_times, batch_loss, epoch_loss


class MinMaxScaler2D(object):
    """
    Rescale input arrays of shape (examples, y, x, variable) to range from out_min to out_max.
    """
    def __init__(self, out_min=0, out_max=1, scale_values=None):
        self.out_min = out_min
        self.out_max = out_max
        self.out_range = out_max - out_min
        self.scale_values = scale_values

    def fit(self, x, y=None):
        """
        Calculate the values for the min/max transformation.
        """
        variables = np.arange(x.shape[-1])
        self.scale_values = pd.DataFrame(0.0, index=variables, columns=["min", "max"])
        for v in variables:
            self.scale_values.loc[v, "min"] = float(x[:, :, :, v].min())
            self.scale_values.loc[v, "max"] = float(x[:, :, :, v].max())
            self.scale_values.loc[v, "range"] = self.scale_values.loc[v, "max"] - self.scale_values.loc[v, "min"]

    def transform(self, x):
        """
        Apply the min/max scaling transformation.
        """
        if x.shape[-1] != self.scale_values.index.size:
            raise ValueError("Input x does not have the correct number of variables")
        x_new = np.zeros(x.shape, dtype=x.dtype)
        for v in self.scale_values.index:
            x_new[:, :, :, v] = (x[:, :, :, v] - self.scale_values.loc[v, "min"]) \
                / (self.scale_values.loc[v, "range"])
            if self.out_min != 0 or self.out_max != 1:
                x_new[:, :, :, v] = x_new[:, :, :, v] * self.out_range + self.out_min
        return x_new

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)
    
def torch_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)
        # get top maxk indicies that correspond to the most likely probability scores
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc.item())
        return list_topk_accs
    
class BinResNet(nn.Module):
    def __init__(self, fcl_layers = [], dr = 0.0, output_size = 1, resnet_model = 18, pretrained = True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.resnet_model = resnet_model 
        if self.resnet_model == 18:
            resnet = models.resnet18(pretrained=self.pretrained)
        elif self.resnet_model == 34:
            resnet = models.resnet34(pretrained=self.pretrained)
        elif self.resnet_model == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.resnet_model == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        elif self.resnet_model == 152:
            resnet = models.resnet152(pretrained=self.pretrained)
        resnet.conv1 = torch.nn.Conv1d(4, 64, (7, 7), (2, 2), (3, 3), bias=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet_output_dim = resnet.fc.in_features
        self.resnet = nn.Sequential(*modules)
        self.fcn = self.make_fcn(self.resnet_output_dim, output_size, fcl_layers, dr)
        
    def make_fcn(self, input_size, output_size, fcl_layers, dr):
        if len(fcl_layers) > 0:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, fcl_layers[0]),
                nn.BatchNorm1d(fcl_layers[0]),
                torch.nn.LeakyReLU()
            ]
            if len(fcl_layers) == 1:
                fcn.append(nn.Linear(fcl_layers[0], output_size))
            else:
                for i in range(len(fcl_layers)-1):
                    fcn += [
                        nn.Linear(fcl_layers[i], fcl_layers[i+1]),
                        nn.BatchNorm1d(fcl_layers[i+1]),
                        torch.nn.LeakyReLU(),
                        nn.Dropout(dr)
                    ]
                fcn.append(nn.Linear(fcl_layers[i+1], output_size))
        else:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, output_size)
            ]
        if output_size > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))
        return nn.Sequential(*fcn)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))
    
def bintrainloop(epochs, X_train, train_batch_size, batches_per_epoch, valid_batch_size, topk, x, y, model, train_criterion, test_criterion, patience, optimizer, lr_scheduler):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=train_batch_size, 
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=valid_batch_size,
        shuffle=False)
    #run training
    epoch_test_losses = []
    for epoch in range(epochs):

        ### Train the model 
        model.train()

        # Shuffle the data first
        batch_loss = []
        accuracy = {k: [] for k in topk}
        indices = list(range(X_train.shape[0]))
        random.shuffle(indices)

        # Now split into batches
        train_batches_per_epoch = int(X_train.shape[0] / train_batch_size) 
        train_batches_per_epoch = min(batches_per_epoch, train_batches_per_epoch)

        # custom tqdm so we can see the progress
        batch_group_generator = tqdm(
            enumerate(train_loader), 
            total=train_batches_per_epoch, 
            leave=True
        )

        for k, (x, y) in batch_group_generator:

            # Converting to torch tensors and moving to GPU
            inputs = x.float().to(device)
            lightning_counts = y.long().to(device)

            # Clear gradient
            optimizer.zero_grad()

            # get output from the model, given the inputs
            pred_lightning_counts = model(inputs)

            # get loss for the predicted output
            loss = train_criterion(pred_lightning_counts, lightning_counts.squeeze(-1))

            # compute the top-k accuracy
            acc = torch_accuracy(pred_lightning_counts.cpu(), lightning_counts.cpu(), topk = topk)
            for i,l in enumerate(topk):
                accuracy[l] += [acc[i]]

            # get gradients w.r.t to parameters
            loss.backward()
            batch_loss.append(loss.item())

            # update parameters
            optimizer.step()

            # update tqdm
            to_print = "Epoch {} train_loss: {:.4f}".format(epoch, np.mean(batch_loss))
            for l in sorted(accuracy.keys()):
                to_print += " top-{}_acc: {:.4f}".format(l,np.mean(accuracy[l]))
            #to_print += " top-2_acc: {:.4f}".format(np.mean(accuracy[2])
            #to_print += " top-3_acc: {:.4f}".format(np.mean(accuracy[3]))
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]['lr'])
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

            if k >= train_batches_per_epoch and k > 0:
                break

        torch.cuda.empty_cache()

        ### Test the model 
        model.eval()
        with torch.no_grad():

            batch_loss = []
            accuracy = {k: [] for k in topk}

            # custom tqdm so we can see the progress
            valid_batches_per_epoch = int(X_test.shape[0] / valid_batch_size) 
            batch_group_generator = tqdm(
                test_loader, 
                total=valid_batches_per_epoch, 
                leave=True
            )

            for (x, y) in batch_group_generator:
                # Converting to torch tensors and moving to GPU
                inputs = x.float().to(device)
                lightning_counts = y.long().to(device)
                # get output from the model, given the inputs
                pred_lightning_counts = model(inputs)
                # get loss for the predicted output
                loss = test_criterion(pred_lightning_counts, lightning_counts.squeeze(-1))
                batch_loss.append(loss.item())
                # compute the accuracy
                acc = torch_accuracy(pred_lightning_counts, lightning_counts, topk = topk)
                for i,k in enumerate(topk):
                    accuracy[k] += [acc[i]]
                # update tqdm
                to_print = "Epoch {} test_loss: {:.4f}".format(epoch, np.mean(batch_loss))
                for k in sorted(accuracy.keys()):
                    to_print += " top-{}_acc: {:.4f}".format(k,np.mean(accuracy[k]))
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

        test_loss = 1 - np.mean(accuracy[1])
        epoch_test_losses.append(test_loss)

        # Lower the learning rate if we are not improving
        lr_scheduler.step(test_loss)

        # Save the model if its the best so far.
        if test_loss == min(epoch_test_losses):
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss
            }
            torch.save(state_dict, "best.pt")

        # Stop training if we have not improved after X epochs
        best_epoch = [i for i,j in enumerate(epoch_test_losses) if j == min(epoch_test_losses)][-1]
        offset = epoch - best_epoch
        if offset >= patience:
            break
