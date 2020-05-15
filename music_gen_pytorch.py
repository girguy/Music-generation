from music21 import *
import glob
from collections import Counter
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


"""
This function extract the features from the MIDI files.

Input : Directory containing the midi files
outputs : numpy ndarray containing numpy arrays of the concatenated elements of the MIDI files.
          Elements are feature extracted from the MIDI files.
"""
def read_midi_dataset(file):
    data = list()
    for midi in glob.glob(file):
        mu = converter.parse(midi)
        s2 = instrument.partitionByInstrument(mu)
        # parts[0] means we only takes into account piano
        note2parse = s2.parts[0].recurse() 
        temp = list()
        for note_ in note2parse:
            if isinstance(note_, note.Note): # isinstance check if element is a note
                temp.append(str(note_.pitch))
            elif isinstance(note_, chord.Chord): # check if it is a chord
                temp.append('.'.join(str(n) for n in note_.normalOrder))
                
        data.append(temp)
        
    data = np.array(data)

    return data


"""
This function transforms a numpy ndarray containaing arrays of elements of MIDI files into one list of
these elements. Example : [[a,b][c,d]] => [a,b,c,d]
"""
def from_ndarrays_to_list(data):
    return [note for notes_ in data for note in notes_] 


"""
This function deletes from the dataset elements that do not appear more than a particular frequency.
It is a filter.
Input : numpy ndarray containing numpy arrays of the concatenated elements of the MIDI files.
Output : List of list. Each list is a concatenation of all the elements of a MIDI file.
"""
def get_vocabulary(data):
    data_ = from_ndarrays_to_list(data)
    # frequence of notes
    freq = dict(Counter(data_))
    # unique_elements is the sorted set of unique elements of the set of MIDI files.
    # The elements selected depends on a particular frequency.
    # Therefore, it is the total vacabulary of the dataset.
    unique_note = sorted([note_ for note_, count in freq.items()])
    
    vocab_dict = {}
    for i in range(len(unique_note)): vocab_dict[unique_note[i]] = i
        
    return unique_note, vocab_dict


"""
"""
def note_to_vec(vocab, data, size_vocab, dim_input):
    
    embeds = nn.Embedding(size_vocab, dim_input)
    
    data_embed = list()
    for song in data:
        data_embed.append(np.array(
            [embeds(torch.tensor([vocab[note]], dtype=torch.long)).detach().numpy()[0] for note in song]))

    return data_embed


"""
"""
def note_to_ind(vocab, data):
    note2ind = {u:i for i, u in enumerate(vocab)}
    ind2note = np.array(vocab)
    
    dataInd = list()
    for song in data: dataInd.append(np.array([note2ind[note] for note in song]))
        
    ind2note = {}
    for note, ind in note2ind.items():
        ind2note[ind] = note
    
    return dataInd, note2ind, ind2note


"""
This function creates the X and y matrices needed by the model.
We use a sliding window mechanism in order to create this dataset.
[a,b,c,d,e,f,g] becomes x1=[a,b,c], y1=[d] then x2=[b,c,d], y2=[e] etc.

Input : List of list. Each list is a concatenation of all the elements of a MIDI file.
Output : matrix X and vector y.
"""
def training_target_samples(data_embed, dataInd, window_size, show_example=False): #time_step = window
    x = list()
    y = list()

    for i in range(len(data_embed)):
        for j in range(len(dataInd[i]) - window_size):
            x.append(data_embed[i][j : j + window_size])
            y.append(dataInd[i][j + window_size])
            
    if show_example is True:
        for i, (trainingInd, targetInd) in enumerate(zip(training[:5], target[:5])):
            print("Step {:4d}".format(i))
            print("  Input: {} ({:s})".format(trainingInd, repr(ind2note[trainingInd])))
            print("  expected output: {} ({:s})".format(targetInd, repr(ind2note[targetInd])))
    
    return np.array(x), np.array(y)


"""
"""
def split_reshape(X, y, split_ratio, size_vocab, dim_input, dim_output):

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y),
                                                        test_size=split_ratio, random_state=0)
        
    X_train, y_train = reshape_datasets(X_train, np.array(y_train), size_vocab, dim_input, dim_output)
    X_test, y_test = reshape_datasets(X_test, np.array(y_test), size_vocab, dim_input, dim_output)
    
    return X_train, X_test, y_train, y_test


"""
"""
def reshape_datasets(X, y, size_vocab, dim_input, dim_output):
    #y_train = np.eye(size_vocab)[y_train]
    #y_test = np.eye(size_vocab)[y_test]
    
    # batch_size , sequence_length , size_encoding = 1 (> 1 if one-hot encoding)
    
    nb_samples = X.shape[0]
    seq_length = X.shape[1]
    X = np.reshape(X, (nb_samples, seq_length, dim_input))/float(size_vocab) # Normalization
    y = np.reshape(y, (nb_samples, dim_output))


    return X, y


"""
"""
def ind_to_embedding(dataInd, dataEmbed):
    ind2embed = {}
    for i in range(len(dataInd)): # -1270
        ind2embed[dataInd[i]] = dataEmbed[i]
    
    return ind2embed


"""
"""
class lstm_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout):
        super(lstm_model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        self.lstm1 = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, n_layers)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fully_connected = nn.Linear(hidden_dim, output_size)
      
    
    def forward(self, x):
        
        batch_size = x.size(0)
        len_seq = x.size(1)
        x = x.to(dtype=torch.float64)
        hidden = self.init_hidden(batch_size)
        h_t = hidden[0].to(dtype=torch.float64)
        c_t = hidden[1].to(dtype=torch.float64)
                
        out, (h_t, c_t) = self.lstm1(x, (h_t, c_t))
        h_t = self.dropout(h_t)
        out, (h_t, c_t) = self.lstm2(h_t, (h_t, c_t)) 
        h_t = self.dropout(h_t)
        out, (h_t, c_t) = self.lstm3(h_t, (h_t, c_t)) 
     
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fully_connected(out)
 
        return out

    """
    This function can be used to extract the output of the last predictions of each batch
    """
    def last_output(batc_size, out):
        lengths = [len_seq for i in range(batch_size)]
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1)
        idx = idx.expand(len(lengths), out.size(2))
        time_dimension = 1 # because batch_first is True so the time step is dimension 1 !
        idx = idx.unsqueeze(time_dimension)
        
        if out.is_cuda:
            idx = idx.cuda(out.data.get_device())
        
        out = out.gather(time_dimension, idx).squeeze(time_dimension)
        
        return out
 
      
    """
    Generates the hidden state and the cell state used in a lstm layers
    """
    def init_hidden(self,  batch_size):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim))


"""
"""
def from_notes_to_MIDI(music_generated, name, offset):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in music_generated:
        
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                
                cn=int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        elif('rest' in pattern):
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano() #???
            output_notes.append(new_rest)
            
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5
    
    midi_stream = stream.Stream(output_notes)
    name_song = name+'.mid'
    midi_stream.write('midi', fp=name_song)
    print(name_song+" downloaded succesfully !")


"""
"""
def initialise_generation(dataInd, window_size, ind2note, ind2embed):
    song_ind = np.random.randint(0, len(dataInd))
    note_ind = np.random.randint(0+window_size, len(dataInd[song_ind])-window_size-1)
    init_sequence = dataInd[song_ind][note_ind:note_ind+window_size]
    
    music_generated = list()
    input_sequence = list()
    for ind in init_sequence:
        note = ind2note[ind]
        embed = ind2embed[ind]
        input_sequence.append(embed)
        music_generated.append(note)
        
        
    return music_generated, np.array(input_sequence)


"""
"""
def generate_note(input_seq):
    input_seq = np.reshape(input_seq, (1, input_seq.shape[0], dim_input))
    input_seq = torch.from_numpy(input_seq)
    input_seq.to(device)
    
    pred = model(input_seq)
    prob = nn.functional.softmax(pred[-1], dim=0).data 
    note_ind = torch.max(prob, dim=0)[1].item() # [1] take indice
    

    return note_ind


"""
"""
def music_generation(model, music_generated, input_sequence, ind2note, ind2embed, nb_steps, name_song):
    
    for i in range(nb_steps):
        pred_ind = generate_note(input_sequence)
        pred_note = ind2note[pred_ind]
        music_generated.append(pred_note)
        pred_embed = ind2embed[pred_ind]
        input_sequence = np.append(np.delete(input_sequence, 0, axis=0), [pred_embed], axis=0)

    offset = 0.5
    from_notes_to_MIDI(music_generated, name_song, offset)
    

"""
"""
def get_dataset(file, window_size, dim_input, dim_output, batch_size, split_ratio):
    data = read_midi_dataset(file)
    unique_note, vocab = get_vocabulary(data)
    size_vocab = len(unique_note)
    
    data_embed = note_to_vec(vocab, data, size_vocab, dim_input)
    dataInd, note2ind, ind2note = note_to_ind(unique_note, data)
    ind2embed = ind_to_embedding(dataInd[0], data_embed[0])
    
    X_dataset, y_dataset = training_target_samples(data_embed, dataInd, window_size)

    X_train, X_test, y_train, y_test = split_reshape(X_dataset, y_dataset, split_ratio,
                                                    size_vocab, dim_input, dim_output)
                                                
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=1, drop_last=True) #batch_size=1 for testing
    
    
    return train_loader, test_loader, dataInd, ind2note, ind2embed, size_vocab


def model_train(model, n_epochs, batch_size, train_loader, test_loader):
    
    # Parameters and Variables for the training
    print_every = batch_size
    valid_loss_min = np.Inf
    train_accuracy = list() # to return
    test_accuracy = list() # to return
    save = 0
    
    train_loss = list()
    val_loss = list()
    
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        true_pred_train = 0
        true_pred_test = 0
        
        epoch_loss = list()
        epoch_val_loss = list()
        for inputs, targets in train_loader:
            
            # Forward
            inputs, targets = inputs.to(device), targets.to(device)
            pred_train = model(inputs)
            
            # Compute Loss and backpropagation
            loss_train = criterion(pred_train, targets.view(-1).long())
            epoch_loss.append(loss_train.item())
            loss_train.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
            
            # Check if the right target has been predicted for the last input of the batch
            prob = nn.functional.softmax(pred_train[-1], dim=0).data 
            note_ind = torch.max(prob, dim=0)[1].item() # [1] take indice
        
            if note_ind == targets[-1].item(): # compare to the last target of the batch
                true_pred_train+=1
            
        # mean accuracy and loss accuracy of this training epoch
        train_loss.append(np.mean(epoch_loss))
        train_accuracy.append(true_pred_train/len(train_loader.dataset))
        
        model.eval()
        for inputs, target in test_loader:

            inputs, target = inputs.to(device), target.to(device)
            pred_test = model(inputs)
            
            loss_test = criterion(pred_test, target.view(-1).long())
            epoch_val_loss.append(loss_test.item())
            
            prob = nn.functional.softmax(pred_test[-1], dim=0).data 
            note_ind = torch.max(prob, dim=0)[1].item() # [1] take indice
            
            if note_ind == target[-1].item():
                true_pred_test+=1
        
        # mean accuracy and loss accuracy of this testing epoch
        val_loss.append(np.mean(epoch_val_loss))
        test_accuracy.append(true_pred_test/len(test_loader.dataset))
        
        model.train()
        print("Epoch: {}/{} => ".format(epoch, n_epochs),
            "Train Loss: {:.6f}, ".format(np.mean(epoch_loss)),
            "Val Loss: {:.6f}, ".format(np.mean(epoch_val_loss)),
            "Train accuracy: {:.6f},".format(true_pred_train/len(train_loader.dataset)),
            "Test accuracy: {:.6f}".format(true_pred_test/len(test_loader.dataset)))
        
        
        save+=1
        if save == 15:
            save = 0
            if np.mean(epoch_val_loss) <= valid_loss_min:
                valid_loss_min = np.mean(epoch_val_loss)
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.
                    format(valid_loss_min,np.mean(epoch_val_loss)))
    
    return model, train_accuracy, test_accuracy


if __name__ == "__main__":
    
    # PARAMETERS FOR THE DATASET.
    file = "/home/cj/Bureau/Master2/Q2/deep_learning/project/tf_dataset/*.mid"
    split_ratio = 0.90
    dim_output = 1 
    
    # HYPER_PARAMETERS FOR THE DATASET.
    window_size = 5
    batch_size = 3
    dim_input = 4
    
    train_loader, test_loader, dataInd, ind2note, ind2embed, size_vocab = get_dataset(file,
                                                                                       window_size,
                                                                                       dim_input,
                                                                                       dim_output,
                                                                                       batch_size,
                                                                                       split_ratio)
    
    # Look if there is a GPU available or not. Otherwise the CPU is used.
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")

    # PARAMETERS FOR THE MODEL USED.
    n_epochs = 1
    n_layers = 1
    
    # HYPER_PARAMETERS FOR THE MODEL USED.
    hidden_dim = 12 
    dropout = 0.3 
    lr = 0.01 # no tested
    
    # INITIALIZATION OF THE LSTM MODEL
    model = lstm_model(input_size=dim_input, output_size=size_vocab,
                       hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    model = model.to(dtype=torch.float64)
    model.to(device)
    
    # DEFINE LOSS AND OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # TRAIN THE MODEL
    model, train_accuracy, test_accuracy = model_train(model.train(), n_epochs, batch_size,
                                                       train_loader, test_loader)
    
    # GENERARTE MUSIC
    nb_steps = 50 # how many words are generated
    name_song = 'pytorch_song'
    music_generated, input_sequence = initialise_generation(dataInd, window_size, ind2note, ind2embed)
    music_generation(model, music_generated, input_sequence, ind2note, ind2embed, nb_steps, name_song)