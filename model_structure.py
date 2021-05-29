import tensorflow as tf 
import os

from tensorflow.keras.layers import *
from attention_layer import M2O_Attention_Layer
from time2vector import Time2Vector

def attention(input_window_length):

    """Specifies the structure of a attention model using Keras' functional API.

    Returns:
    model (tensorflow.keras.Model): The uncompiled model.

    """
    K = tf.keras.backend
    window_size = input_window_length
    hidden_units = 512
    INPUT_DIM = 1

    # Encoder
    input_layer = Input(shape=(window_size,))
    reshape_layer = tf.keras.layers.Reshape((window_size, INPUT_DIM))(input_layer)
    enc_out, enc_state = GRU(hidden_units, return_sequences=True, return_state=True)(reshape_layer)
    #Dropout
    attention_vector = M2O_Attention_Layer()(enc_out)
    attention_vector = Flatten()(attention_vector)

    dense_out = Dense(1024,activation='relu')(attention_vector)
    out = Dense(1, activation='linear')(dense_out)
    model = tf.keras.Model(inputs=[input_layer], outputs=out)

    return model

def seq2point(input_window_length):
    """Specifies the structure of a seq2point model using Keras' functional API.

    Returns:
    model (tensorflow.keras.Model): The uncompiled seq2point model.

    """

    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(10, 1), strides=(1, 1), padding="same", activation="relu")(reshape_layer)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=30, kernel_size=(8, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=40, kernel_size=(6, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=50, kernel_size=(5, 1), strides=(1, 1), padding="same", activation="relu")(conv_layer_4)
    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
    label_layer = tf.keras.layers.Dense(1024, activation="relu")(flatten_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def time2vec_attention(input_window_length):
    window_size = input_window_length
    INPUT_DIM = 1
    # Encoder
    in_seq = tf.keras.Input(shape=(window_size, INPUT_DIM))
    x = Time2Vector(window_size)(in_seq)
    x = Concatenate(axis=-1)([in_seq, x])
    attention_vector = M2O_Attention_Layer()(x)
    attention_vector = Flatten()(attention_vector)
    
    dense_out = Dense(1024, activation='relu')(attention_vector)
    # dropout = Dropout(0.5)(dense_out)
    # out = Dense(INPUT_DIM, activation='linear')(dropout)
    out = Dense(1, activation='linear')(dense_out)
    model = tf.keras.Model(inputs=[in_seq], outputs=out)

    return model


def create_model(network_type,input_window_length):
    if network_type == 'seq2point':
        return seq2point(input_window_length)
    elif network_type == 'attention':
        return attention(input_window_length)
    elif network_type == 't2vattention':
        return time2vec_attention(input_window_length)



def save_model(model, network_type, algorithm, appliance, save_model_dir):

    """ Saves a model to a specified location. Models are named using a combination of their 
    target appliance, architecture, and pruning algorithm.

    Parameters:
    model (tensorflow.keras.Model): The Keras model to save.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """
    
    #model_path = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_path = save_model_dir

    if not os.path.exists (model_path):
        open((model_path), 'a').close()

    model.save(model_path)

def load_model(model, network_type, algorithm, appliance, saved_model_dir):

    """ Loads a model from a specified location.

    Parameters:
    model (tensorflow.keras.Model): The Keas model to which the loaded weights will be applied to.
    network_type (string): The architecture of the model ('', 'reduced', 'dropout', or 'reduced_dropout').
    algorithm (string): The pruning algorithm applied to the model.
    appliance (string): The appliance the model was trained with.

    """

    #model_name = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_name = saved_model_dir
    print("PATH NAME: ", model_name)

    if network_type == "t2vattention" or algorithm == "t2vattention" :
        model = tf.keras.models.load_model(model_name, custom_objects={'Time2Vector': Time2Vector})
    else:
        model = tf.keras.models.load_model(model_name)

    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model