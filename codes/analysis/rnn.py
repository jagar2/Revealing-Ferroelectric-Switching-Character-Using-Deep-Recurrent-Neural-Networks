"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""

import keras
from keras.models import Sequential, Input, Model
from keras.layers import (Dense, Conv1D, Convolution2D, GRU, LSTM, Recurrent, Bidirectional, TimeDistributed,
                          Dropout, Flatten, RepeatVector, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization)
from keras.optimizers import Adam
from keras.regularizers import l1
from ..util.file import *
import numpy as np
import datetime



def rnn(layer_type, size, encode_layers,
        decode_layers, embedding,
        steps, lr=3e-5, drop_frac=0.,
        bidirectional=True, l1_norm=1e-4,
        batch_norm=[False, False], **kwargs):
    """
    Function which builds the recurrent neural network autoencoder

    Parameters
    ----------
    layer : string; options: 'lstm','gru'
        selects the layer type
    size  : numpy, int
        sets the size of encoding and decoding layers in the network
    encode_layers  : numpy, int
        sets the number of encoding layers in the network
    decode_layers : numpy, int
        sets the number of decoding layers in the network
    embedding : numpy, int
        sets the size of the embedding layer
    steps : numpy, int
        length of the input time series
    lr : numpy, float
        sets the learning rate for the model
    drop_frac : numpy, float
        sets the dropout fraction
    bidirectional : numpy, bool
        selects if the model is linear or bidirectional
    l1_norm : numpy. float
        sets the lambda value of the l1 normalization. The larger the value the greater the
        sparsity. None can be passed to exclude the use or l1 normailzation.

    Returns
    -------
    model : Keras, object
        Keras tensorflow model
    """

    # Selects the type of RNN neurons to use
    if layer_type == 'lstm':
        layer = LSTM
    elif layer_type == 'gru':
        layer = GRU

    # defines the model
    model = Sequential()

    # selects if the model is bidirectional
    if bidirectional:
        wrapper = Bidirectional
        # builds the first layer

        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(encode_layers > 1)),
                          input_shape=(steps, 1)))
        add_dropout(model, drop_frac)
    else:
        def wrapper(x): return x
        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(encode_layers > 1),
                                input_shape=(steps, 1))))
        add_dropout(model, drop_frac)

    # builds the encoding layers
    for i in range(1, encode_layers):
        model.add(wrapper(layer(size, return_sequences=(i < encode_layers - 1))))
        add_dropout(model, drop_frac)

    # adds batch normalization prior to embedding layer
    if batch_norm[0]:
        model.add(BatchNormalization())

    # builds the embedding layer
    if l1_norm == None:
        # embedding layer without l1 regularization
        model.add(Dense(embedding, activation='relu', name='encoding'))
    else:
        # embedding layer with l1 regularization
        model.add(Dense(embedding, activation='relu',
                        name='encoding', activity_regularizer=l1(l1_norm)))

    # adds batch normalization after embedding layer
    if batch_norm[1]:
        model.add(BatchNormalization())

    # builds the repeat vector
    model.add(RepeatVector(steps))

    # builds the decoding layer
    for i in range(decode_layers):
        model.add(wrapper(layer(size, return_sequences=True)))
        add_dropout(model, drop_frac)

    # builds the time distributed layer to reconstruct the original input
    model.add(TimeDistributed(Dense(1, activation='linear')))

    # complies the model
    model.compile(Adam(lr), loss='mse')

    run_id = get_run_id(layer_type, size, encode_layers,
                        decode_layers, embedding,
                        lr, drop_frac, bidirectional, l1_norm,
                        batch_norm)

    # returns the model
    return model, run_id


def add_dropout(model, value):
    if value > 0:
        return model.add(Dropout(value))
    else:
        pass


def get_run_id(layer_type, size, encode_layers,
               decode_layers, embedding,
               lr, drop_frac,
               bidirectional, l1_norm,
               batch_norm, **kwargs):
    """
    Function which builds the run id

    Parameters
    ----------
    layer_type : string; options: 'lstm','gru'
        selects the layer type
    size  : numpy, int
        sets the size of encoding and decoding layers in the network
    encode_layers  : numpy, int
        sets the number of encoding layers in the network
    decode_layers : numpy, int
        sets the number of decoding layers in the network
    embedding : numpy, int
        sets the size of the embedding layer
    lr : numpy, float
        sets the learning rate for the model
    drop_frac : numpy, float
        sets the dropout fraction
    bidirectional : numpy, bool
        selects if the model is linear or bidirectional
    l1_norm : numpy. float
        sets the lambda value of the l1 normalization. The larger the value the greater the
        sparsity. None can be passed to exclude the use or l1 normailzation.

    Returns
    -------
    run : string
        string for the model
    """

    # builds the base of the model name
    run = (f"{layer_type}_size{size:03d}_enc{encode_layers}_emb{embedding}_dec{decode_layers}_lr{lr:1.0e}"
           f"_drop{int(100 * drop_frac)}").replace('e-', 'm')

    # adds optional information
    if Bidirectional:
        run = 'Bidirect_' + run
    if layer_type == 'conv':
        run += f'_k{kernel_size}'
    if np.any(batch_norm):

        if batch_norm[0]:
            ind = 'T'
        else:
            ind = 'F'

        if batch_norm[1]:
            ind1 = 'T'
        else:
            ind1 = 'F'

        run += f'_batchnorm_{ind}{ind1}'
    return run


def get_activations(model, X=[], i=[], mode='test'):
    """
    function to get the activations of a specific layer
    this function can take either a model and compute the activations or can load previously
    generated activations saved as an numpy array

    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm

    Returns
    -------
    activation : float
        array containing the output from layer i of the network
    """
    # if a string is passed loads the activations from a file
    if isinstance(model, str):
        activation = np.load(model)
        print(f'activations {model} loaded from saved file')
    else:
        # computes the output of the ith layer
        activation = get_ith_layer_output(model, np.atleast_3d(X), i, model)

    return activation


def get_ith_layer_output(model, X, i, mode='test'):
    """
    Computes the activations of a specific layer
    see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'


    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm
    Returns
    -------
    layer_output : float
        array containing the output from layer i of the network
    """
    # computes the output of the ith layer
    get_ith_layer = keras.backend.function(
        [model.layers[0].input, keras.backend.learning_phase()], [model.layers[i].output])
    layer_output = get_ith_layer([X, 0 if mode == 'test' else 1])[0]

    return layer_output


def train_model(run_id, model, data, data_val, folder,
                batch_size=1800, epochs=25000, seed=42):
    """
    Function which trains the model


    Parameters
    ----------
    run_id : string
        sets the id for the run
    model  : numpy array, float
        Input data
    data  : numpy, float
        training data
    data_val : numpy, float
        validation data
    folder : string, optional
        folder to save the training results
    batch_size : int, optional
        number of samples in the batch. This is limited by the GPU memory
    epochs : int, optional
        number of epochs to train for
    seed : int, optional
        sets a standard seed for reproducible training

    """
    # computes the current time to add to filename
    time = datetime.datetime.now()
    # fixes the seed for reproducible training
    np.random.seed(seed)

    # makes a folder to save the dara
    run_id = make_folder(folder + '/{0}_{1}_{2}_{3}h_{4}m'.format(time.month,
                                                                  time.day, time.year,
                                                                  time.hour, time.minute) + '_' + run_id)
    # saves the model prior to training
    model_name = run_id + 'start'
    keras.models.save_model(
        model, run_id + '/start_seed_{0:03d}.h5'.format(seed))

    # sets the file path
    filepath = run_id + '/weights.{epoch:06d}-{val_loss:.4f}.hdf5'

    # callback for saving checkpoints. Checkpoints are only saved when the model improves
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                 verbose=0, save_best_only=True,
                                                 save_weights_only=True, mode='min', period=1)

    # logs training data and the loss to a csv file
    logger = keras.callbacks.CSVLogger(
        run_id + '/log.csv', separator=',', append=True)

    # trains the model
    history = model.fit(np.atleast_3d(data), np.atleast_3d(data),
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(np.atleast_3d(
                            data_val), np.atleast_3d(data_val)),
                        callbacks=[checkpoint, logger])


def update_decoder(model, weights):
    """
    updates the decoder half of the model

    TODO update to make generalizable

    Parameters
    ----------
    model  : numpy array, float
        Input data
    weights  : string
        path to file where the weights are saved

    """
    # builds the resonance decoding model
    decode = Sequential()
    decode.add(BatchNormalization(input_shape=(16,)))
    decode.add(RepeatVector(96))
    decode.add(Bidirectional(
        LSTM(64, return_sequences=True), input_shape=(96, 1)))
    decode.add(Bidirectional(
        LSTM(64, return_sequences=True), input_shape=(96, 1)))
    decode.add(Bidirectional(
        LSTM(64, return_sequences=True), input_shape=(96, 1)))
    decode.add(Bidirectional(
        LSTM(64, return_sequences=True), input_shape=(96, 1)))
    decode.add(TimeDistributed(Dense(1, activation='linear')))

    decode.compile(Adam(3e-5), loss='mse')

    decode.load_weights(weights)

    # Sets the layers to match the training model
    model.layers[10].set_weights((decode.layers[0].get_weights()))
    model.layers[11].set_weights((decode.layers[1].get_weights()))
    model.layers[12].set_weights((decode.layers[2].get_weights()))
    model.layers[14].set_weights((decode.layers[3].get_weights()))
    model.layers[16].set_weights((decode.layers[4].get_weights()))
    model.layers[18].set_weights((decode.layers[5].get_weights()))
    model.layers[20].set_weights((decode.layers[6].get_weights()))

    return model, decode
