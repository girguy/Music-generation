#!/usr/bin/env python
# coding: utf-8

from music21 import *
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras as keras
from keras.utils.np_utils import to_categorical
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Bidirectional, Flatten
from keras_self_attention import SeqSelfAttention
# pip install keras-self-attention
import pandas as pd


"""
This function extract the features from the MIDI files.

Input : Directory containing the midi files
outputs : numpy ndarray containing numpy arrays of the concatenated elements of the MIDI files.
          Elements are feature extracted from the MIDI files.
"""
def read_midi_dataset(file): 
    notes = list()
    for midi in glob.glob(file):
        notes_to_parse = None
        mu = converter.parse(midi)
        s2 = instrument.partitionByInstrument(mu)
        # parts[0] means we only takes into account piano
        notes_to_parse = s2.parts[0].recurse() 
        notes_song = list()
        for element in notes_to_parse:
            
            if isinstance(element, note.Note): # isinstance check if element is a note
                notes_song.append(str(element.pitch))

            elif isinstance(element, chord.Chord): # check if it is a chord
                notes_song.append('.'.join(str(n) for n in element.normalOrder))   
            
            elif isinstance(element, note.Rest):
                notes.append(str(element.name)  + " " + str(element.quarterLength))
            
        notes.append(notes_song)
    
    notes = np.array(notes)

    return notes


"""
This function transforms a numpy ndarray containaing arrays of elements of MIDI files into one list of
these elements. Example : [[a,b][c,d]] => [a,b,c,d]
"""
def from_ndarrays_to_list(data):
    return [element for elements_ in data for element in elements_] 


"""
This function shows an histogram of the notes and prints the total number of notes as well as the number
of unique notes.

Input : numpy ndarray containing numpy arrays of the concatenated elements of the
        MIDI files.
Output : No output. 
"""
def data_exploration(data, printt=False, show=False):
    elements_list = from_ndarrays_to_list(data)
    unique_elements = list(set(elements_list))
    frequence_of_elements = dict(Counter(elements_list))
    
    if printt is True:
        print("The number of notes in the dataset is {}.".format(len(elements_list)))
        print("The number of different notes in the dataset is {}.".format(len(unique_elements)))
     
    if show is True : # histogram of the notes
        plt.bar(list(frequence_of_elements.keys()), frequence_of_elements.values(), color='g')
        plt.show()

        
"""
This function deletes from the dataset elements that do not appear more than a particular frequency.
It is a filter.
Input : numpy ndarray containing numpy arrays of the concatenated elements of the MIDI files.
Output : List of list. Each list is a concatenation of all the elements of a MIDI file.
"""
def select_notes(data, frequency, printt=False):
    elements_list = from_ndarrays_to_list(data)
    frequence_of_notes = dict(Counter(elements_list))
    # unique_elements is the sorted set of unique elements of the set of MIDI files. The elements selected depends
    # on a particular frequency. Therefore, it is the total vacabulary of the dataset.
    unique_elements = sorted([elements_list for elements_list,
                              count in frequence_of_notes.items() if count>=frequency])

    if printt is True :
        print("The number of different notes that appear at least {} time is {}.".format(frequency,
                                                                                     len(unique_elements)))
    new_data = list()
    for elements_ in data:
        temp = list()
        for element in elements_:
            if element in unique_elements:
                temp.append(element)
        new_data.append(temp)
        
    return new_data


"""
This function creates the X and y matrices needed by the model.
We use a sliding window mechanism in order to create this dataset.
[a,b,c,d,e,f,g] becomes x1=[a,b,c], y1=[d] then x2=[b,c,d], y2=[e] etc.

Input : List of list. Each list is a concatenation of all the elements of a MIDI file.
Output : matrix X and vector y.
"""
def create_dataset(data, window): #time_step = window
    x = list()
    y = list()
    for elements_ in data:
        for i in range(len(elements_)-window):
            x.append(elements_[i:i + window])
            y.append(elements_[i + window])
    
    return np.array(x), np.array(y)


"""
This function makes the different matrices usable by an LSTM unit.
input : For X matrices : [nb_samples, window_size]
        For y matrices : [nb_samples, ]
output : For X matrices : [nb_samples, window_size, 1] # 1 because there is only one feature (element)
         For y matrices : [nb_samples, vocabulary_size] # One-hot encoding
"""
def reshape(X_train, X_test, y_train, y_test, size_vocab):
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes = size_vocab)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes = size_vocab)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))/float(size_vocab) # Normalization
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))/float(size_vocab) # Normalization
    
    return X_train, X_test, y_train, y_test


"""
Deep Neural network works better with numerical dataset. Each element is going to be replaced by a number.
Input : matrix X and vector y non usable by a Deep Neural network.
Output : X_train, y_train, X_test, y_test
"""
def dataset_for_NN(X, y, data, split_ratio):

    unique_data = list(sorted(set(from_ndarrays_to_list(data)))) 
    dict_vocabulary = dict((element, nb) for nb, element in enumerate(unique_data)) # from element to integer
    size_vocab = len(unique_data)

    X_dataset = list()
    y_dataset = list()
    
    for i in range(len(X)):
        temp_X = []
        for element in X[i]:
            temp_X.append(dict_vocabulary[element])
        X_dataset.append(temp_X)
        y_dataset.append(dict_vocabulary[y[i]])
    
    X_train, X_test, y_train, y_test = train_test_split(np.array(X_dataset), np.array(y_dataset),
                                                        test_size=split_ratio, random_state=0)
    
    X_train, X_test, y_train, y_test = reshape(X_train, X_test, y_train, y_test, size_vocab)
    
    return X_train, X_test, y_train, y_test, dict_vocabulary, size_vocab



def lstm_model_1(window_size, dropout_rate, size_vocab, size_lstm):
    model = Sequential()
    model.add(LSTM(size_lstm, input_shape=(window_size, 1), return_sequences=True)) # 512
    model.add(Dropout(dropout_rate))
    model.add(LSTM(size_lstm, return_sequences=True)) # 512
    model.add(Dropout(dropout_rate))
    model.add(LSTM(size_lstm)) # 512
    model.add(Dense(256))
    model.add(Dropout(dropout_rate))
    model.add(Dense(size_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model

def lstm_model_2(window_size, dropout_rate, size_vocab, size_lstm):
    model = Sequential()
    model.add(LSTM(size_lstm, input_shape=(window_size, 1), recurrent_dropout=0.3, return_sequences=True)) # 512
    model.add(LSTM(size_lstm, return_sequences=True, recurrent_dropout=0.3)) # 512
    model.add(LSTM(size_lstm)) # 512
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(size_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def lstm_attention_model(window_size, dropout_rate, size_vocab, size_lstm):
    model = Sequential()
    model.add(Bidirectional(LSTM(size_lstm, input_shape=(window_size, 1),return_sequences=True))) # 512
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    
    model.add(Bidirectional(LSTM(size_lstm, return_sequences=True))) # 512
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    
    model.add(Bidirectional(LSTM(size_lstm, return_sequences=True))) # 512
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dense(256))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten()) #Supposedly needed to fix stuff before dense layer
    model.add(Dense(size_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def save_on_disk(model, history, name):
    name_model = name+".h5"
    model.save(name_model)
    print(name_model + " model saved on disk")
    
    history_ = pd.DataFrame.from_dict(history.history, orient='index')
    name_history = name+"_history.csv"
    print(name_history + " history saved on disk")
    history_.to_csv(name_history)


# An history object is the output of the fit(), it keeps tracks of the value of
# [loss, val_loss, accuracy, val_accuray] for each epoch during the training of the model.
# Very important to plot the learning curve for the training (loss) and testing set (val_loss).
def fit_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, name, callbacks=False):
    if callbacks is False:
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                  batch_size=batch_size, verbose=1)
    else:
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')    
        callbacks_list = [checkpoint]
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                  batch_size=batch_size, callbacks=callbacks_list, verbose=1)
    
    # Save results on disk
    save_on_disk(model, history, name)
    
    return model

def get_model(which, window_size, dropout_rate, size_vocab, X_train, y_train, X_test, y_test,
             size_lstm, batch_size, epochs, name_param, callbacks):
    if which == 1:
        name = 'lstm_model_1'+name_param
        model = lstm_model_1(window_size, dropout_rate, size_vocab, size_lstm)
        model = fit_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, name, callbacks)

    elif which == 2:
        name = 'lstm_model_2'+name_param
        model = lstm_model_2(window_size, dropout_rate, size_vocab, size_lstm)
        model = fit_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, name, callbacks)

    elif which == 3:
        name = 'lstm_att_model'+name_param
        model = lstm_attention_model(window_size, dropout_rate, size_vocab, size_lstm)
        model = fit_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, name, callbacks)

    else:
        return print("No corresponding model")

    return model

def get_dataset(file, frequency, window_size):
    
    data_elements = read_midi_dataset(file)
    data_filtered = select_notes(data_elements, frequency=frequency)
    
    X, y = create_dataset(data_filtered, window_size)
    
    X_train, X_test, y_train, y_test, dict_vocabulary, size_vocab = dataset_for_NN(X, y, data_filtered,
                                                                                   split_ratio=0.2)
    
    return X_train, X_test, y_train, y_test, dict_vocabulary, size_vocab



if __name__ == "__main__":
    
    # PARAMETERS
    file = "/home/cj/Bureau/Master2/Q2/deep_learning/project/20_songs/*.mid"
    frequency = 0
    window_size = 32 # [32, 64, 100] => last to test
    dropout_rate = 0.3 # [0, 0.2, 0.4, 0.8] => second to test
    batch_size = 256  # [128, 256, 512, 1024] => third to test
    epochs = 1
    which = 1
    
    size_lstm = [64, 128, 256, 512] # first to test
    name_param = ['_size_64', '_size_128', '_size_256', '_size_512']
    
    X_train, X_test, y_train, y_test, dict_vocabulary, size_vocab = get_dataset(file, frequency, window_size)
    
    for i in range(len(size_lstm)):
        model = get_model(which, window_size, dropout_rate, size_vocab, X_train, y_train, X_test, y_test,
                         size_lstm[i], batch_size, epochs, name_param[i], callbacks=False)
    


