import os
import warnings
import itertools
import random
import librosa
import librosa.display
import librosa as lr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from maad import sound, util, features
from librosa import feature
from IPython.display import Audio
from os import listdir
from os.path import isfile, join


def plot_listen_examples(path_audio):
    
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
    
    
    if '.wav' in path_audio:
        print('wav')
        y, sr = sound.load(path_audio)
    else:
        audio_files = [f for f in listdir(path_audio) if isfile(join(path_audio, f))]
        audio_files = [i for i in audio_files if '.wav' in i]
        file = random.choice(audio_files)
        y, sr = sound.load(join(path_audio, file))
        
    S = librosa.feature.melspectrogram(y=y,sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=1.0,
                                       fmin=f_min, fmax=f_max)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S, x_axis='time',y_axis='mel',sr=sr, ax=ax,
                                   fmax=f_max)
    #fig.colorbar(img,ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram of '+file.split('wav')[0])
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


# https://github.com/juansulloa/chorus/blob/main/utils/segmentation.py

def roi2windowed(wl, roi):
    """
    Split a single region of interest (roi) into multiple regions of fixed size according
    to a window length. If window length (wl) is longer than the roi, the result is a single
    roi of length wl and centered in the middle of the roi. If the window length is 
    shorter than the, the roi is splitted into multiple regions. Regions must have at
    least 50% of overlap with the new window length. There is no overlap between windows.
    
    Parameters
    ----------
    wl : float
        Window length of the resulting regions of interest
    roi : pandas.core.frame.DataFrame
        Regions of interest with at least five columns, min_t, max_t, min_f, max_f, label.
    Returns
    -------
    roi_fmt : pandas.core.frame.DataFrame
        Formated regions of interest with fixed size.
    """
    roi_len = (roi.max_t - roi.min_t)
    
    if roi_len < wl:
        # region shorter than window length
        roi_median = roi.min_t + roi_len/2
        roi.loc['min_t'] = roi_median - wl/2
        roi.loc['max_t'] = roi_median + wl/2
        roi_fmt = roi.to_frame().T
    
    else:
        # region larger than window length
        # compute arrays. If more than 50% overlap remains, add a window
        roi_fmt = pd.DataFrame({'min_t': np.arange(roi.min_t, roi.max_t-wl+(wl/2), wl),
                                 'max_t': np.arange(roi.min_t+wl, roi.max_t+(wl/2), wl),
                                 'min_f': roi.min_f,
                                 'max_f': roi.max_f,
                                 'label': roi.label})
    return roi_fmt


def rois_windowed(wl, step, tlims=(0, 60), flims=(0, 22050), rois_annot=None, tn=None, fn=None):
    """
    Discretize audio signal into multiple segments and use manual annotations to label 
    rois.
    
    Parameters
    ----------
    wl : float
        Window length in seconds.
    step : float
        Step for winowed roi in seconds
    tlims : tuple, optional
        Temporal limits to create rois in seconds. The default is (0, 60).
    flims : tuple, optional
        Frequency limits to create rois in Hertz. The default is (0, 22050).
    rois_annot : pandas DataFrame
        regions of interest with annotations.
    Returns
    -------
    rois : pandas Dataframe
    """
    # init rois windowed
    rois = pd.DataFrame({'min_t': np.arange(tlims[0], tlims[1]-wl+1, step),
                         'max_t': np.arange(tlims[0]+wl, tlims[1]+1, step),
                         'min_f': flims[0],
                         'max_f': flims[1],
                         'label': '0'})
    
    rois = util.format_features(rois, tn, fn)
    
    # if no annotations provided, return windowed rois
    if rois_annot is None:
        return rois
    
    # if provided, add rois labels
    for idx, row in rois_annot.iterrows():
        if wl==step:
            idx_min_t = (rois.min_t - row.min_t).abs().idxmin()
            idx_max_t = (rois.max_t - row.max_t).abs().idxmin()
            rois.loc[idx_min_t:idx_max_t, 'label'] = row.label
        else:
            print('TODO: correct assignment of labels when wl < step')
            #idx_min_t = ((row.min_t - wl*0.5) - rois.min_t)<=0
            #idx_max_t = ((row.max_t + wl*0.5) - rois.max_t)>=0
            #rois.loc[idx_min_t & idx_max_t, 'label'] = row.label

    return rois

def find_file(filename, search_path):
    """
    File searching tool. Searches a file with filename recurively in a directory.
    
    Parameters
    ----------
    filename : str, optional
        Filename of the file that you want to search.
        
    search_path : str, optional
        Path to directory. The default is the current directory './'.
    Returns
    -------
    str
        Absolute path to file.
    """
    
    files_in_path = [f for f in listdir(search_path) if isfile(join(search_path, f))]
    filename_found = [file for file in files_in_path if filename in file]
    if len(filename_found)!=1:
        warnings.warn("No file found or more than one file has been found")
    return os.path.join(search_path, filename_found[0])

    
