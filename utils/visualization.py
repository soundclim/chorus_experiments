
import itertools
import random
import librosa
import librosa.display
import librosa as lr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from maad import sound
from IPython.display import Audio
from os import listdir
from os.path import isfile, join
from tqdm import tqdm


def plot_listen_examples(path_audio,
                         n_fft = 1024,
                         hop_length = 256, 
                         n_mels = 128,
                         f_min = 50,
                         f_max = 4000,
                         power=1.0
                        ):

    """
    # length of the FFT window
    n_fft = 1024
    # number of samples between successive frames.
    hop_length = 256 
    # number of Mel bands to generate
    n_mels = 128
    # frequency range? TALK WITH HERPETOLOGIST!!
    f_min = 50
    f_max = 4000
    # Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc.
    # power = 
    """
    
    if '.wav' in path_audio:
        y, sr = sound.load(path_audio)
        file = path_audio
    else:
        audio_files = [f for f in listdir(path_audio) if isfile(join(path_audio, f))]
        audio_files = [i for i in audio_files if '.wav' in i]
        file = random.choice(audio_files)
        y, sr = sound.load(join(path_audio, file))
        
    S = librosa.feature.melspectrogram(y=y,sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power,
                                       fmin=f_min, fmax=f_max)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S, x_axis='time',y_axis='mel',sr=sr, ax=ax,
                                   fmax=f_max)
    #fig.colorbar(img,ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram of '+path_audio.split('wav')[0])
    plt.show()
    display(Audio(data=y, rate=sr))
    
    return file


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    return cm

def plot_nn_history(history, epochs):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')