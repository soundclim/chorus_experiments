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
from maad import sound
from librosa import feature, get_duration, load
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
 
# https://github.com/juansulloa/chorus/blob/main/utils/segmentation.py
    
def roi2windowed(wl, roi, fname, wav_path):
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
        roi.loc['min_t'] = roi.min_t #roi_median - wl/2
        roi.loc['max_t'] = roi.min_t + wl #roi_median + wl/2
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
        
def batch_format_rois(df, wl, wav_path):
    """ format rois to a fixed length window"""
    rois_fmt = pd.DataFrame()
    for idx, roi in df.iterrows():
        roi_fmt = roi2windowed(wl, roi, fname=roi.fname, wav_path=wav_path)
        roi_fmt['fname'] = roi.fname
        rois_fmt = pd.concat([rois_fmt,roi_fmt],ignore_index=True)

    rois_fmt.reset_index(inplace=True, drop=True)
    return rois_fmt

def batch_write_samples(rois, 
                        wav_path, 
                        target_sr,
                        path_save,
                        flims,
                        verbose=False):
    # TODO: Call raw recording 1 time
    # Multiprocessing
    for idx, roi in tqdm(rois.iterrows()):
        
        s, fs = sound.load(find_file(roi.fname, search_path=wav_path))
    
        # Preprocessing operations
        s_trim = sound.trim(s, target_sr, min_t=roi.min_t, max_t=roi.max_t, pad=True)
        s_resampled = sound.resample(s_trim, fs, target_sr)
        s_filtered = sound.select_bandwidth(s_resampled,target_sr,fcut=flims, forder=5, fname ='butter', ftype='bandpass')
        s_normalized = sound.normalize(s_filtered, max_amp=0.7)

        fname_save = os.path.join(path_save, roi.sample_name)
        sound.write(fname_save, target_sr, s_normalized, bit_depth=16)
        
"""        
def preprocessing_audio_file_todo(s,fs):
    
    # Preprocessing operations
    s_resampled = sound.resample(s, fs, target_sr)
    s_filtered = sound.select_bandwidth(s_resampled,target_sr,fcut=flims, forder=5, fname ='butter', ftype='bandpass')
    s_normalized = sound.normalize(s_filtered, max_amp=0.7)
    
    return s_normalized
            
def batch_write_samples_todo(rois, 
                        wav_path, 
                        target_sr,
                        path_save,
                        flims,
                        verbose=False):
    
    # TODO: Multiprocessing or avoid for loops
    raw_files = rois.fname.unique()   
    for raw_recording in raw_files:
        if verbose:
            print(find_file(roi.fname+'.wav', search_path=wav_path))
        s, fs = sound.load(find_file(raw_recording, search_path=wav_path))
        # https://joblib.readthedocs.io/en/latest/parallel.html
        s_normalized = preprocessing_audio_file(s,fs)
        
        rois_from_file = rois[rois.fname.isin([raw_recording])]
        for idx, roi in rois_from_file.iterrows():
            # could we use as last step trim?
            s_trim = sound.trim(s_normalized, target_sr, min_t=roi.min_t, max_t=roi.max_t, pad=True)
            fname_save = os.path.join(path_save, roi.sample_name)
            sound.write(fname_save, target_sr, s_normalized, bit_depth=16)
"""

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
        print('file:',filename)
        print('search_path:',search_path)
        warnings.warn("No file called found or more than one file has been found")
    return os.path.join(search_path, filename_found[0])

def readme_generator(df,name,sr,wl,flims):
    
    """
    Create a dataset from raw fixed time recordings and annotations from audacity 
    
    Parameters 
    ---------- 
    df: pandas.core.frame.DataFrame
        dataframe of the dataset
    wl : int
        Fixed window lenght to split the recording
    sr : int
        Sampling rate to convert the audio file
    flims : list
        tuple composed of (minimun_frequency, maximun_frequency) in Hz
    site_list : list
        Passive Acoustic Monitoring device
    name : str
        Path where the last folder is the place where preprocessed recordings and df_compiled is saved
        
    Returns 
    ------- 
    None but save a readme.txt file in name dir 
    
    """
    
    title = name.split('/')[-1]
    title_line = 'Chorus - ' + title
    samples = str(df.shape[0])
    line_samples = 'This dataset has '+samples+' samples annotated with presence-absence of '
    species = str(len(df['species'].unique()))
    line_species = species+' species (multilabel annotation).'
  
    line_site = 'It was built using soundscape recordings from passive acoustic monitoring in sites:' 
    site_list = df['site'].value_counts().to_string()
    site_list = '-'+site_list.replace('\n', '\n-')
    
    sampling_line = 'The recordings were preprocessed by resampling the audio to '+str(sr)+'Hz and formatted to 16 bit depth. '
    annotation_line = 'In addition, a clear begin and end of the vocalisation was annotated using Audacity. '
    preprocessing_line = 'The soundscape recordings were just trimmed to a fixed window length, but no filtering was applied.'
    frequency_line = 'The frequency limits are: '+str(flims)
    
    wl_line = 'The duration of each sample is '+str(wl)+' seconds. Number of fixed-window samples used:'
    species_list = df['species'].value_counts().to_string()
    species_list = '-'+species_list.replace('\n', '\n-')
    
    quality_line = 'The datasets has 3 quality labels:'
    quality_list = df['quality'].value_counts().to_string()
    quality_list = '-'+quality_list.replace('\n', '\n-')
    class_line = 'The final class are based on site, species and quality. We merge quality C and M:'
    class_list = df['class'].value_counts().to_string()
    class_list = '-'+class_list.replace('\n', '\n-')
    
    title_problems = '\nProbable issues and solutions'
    future_work_line1 = 'The construction was based on the annotations but not in a sliding windows approach. '
    future_work_line2 = 'This imply that some portions of audio could be repeated making emphasis in other sounds.'
    future_work_line = future_work_line1 + future_work_line2
    repo_link = 'https://github.com/jscanass/chorus_experiments'
    repo_line = 'See the repository for more information: '+repo_link
    dictionary_title =  ' \nDictionary of data \n-----------------------------\n'
    dictionary = '''-sample_name: unique identifier of each sample which follows SAMPLE_*id*_*site*_*label*_FOLD_*fold*.wav
                -fname: file of 60s extracted from a site and used for annotators
                -min_t: second where the annotation starts in fixed window length
                -max_t: second where the annotation ends in fixed window length
                -label: annotations with species and quality
                -species: species code
                -quality: quality code
                -site: identifier of recorder
                -date: date of recording
                -class: class of detection. This is the column of interest in the ML problem
                -fold: int number of fold. Fold 0 is a test set, you must not use it in training or cross-validation
                -subset: if the sample is from train or test. If test, the fold column must be 0
                - Binary columns of each class
                '''
    dictionary_line = dictionary_title + dictionary.replace('\n            ',' \n')
    lines = [title_line, 
             '-------------------------------\n',
             'Description',
             '-----------',
             line_samples + line_species,
             line_site,
             site_list,
             sampling_line,
             annotation_line,
             preprocessing_line,
             frequency_line,
             wl_line,
             species_list,
             quality_line,
             quality_list,
             class_line,
             class_list,
             title_problems,
            '-----------------------------',
             future_work_line,
             repo_line,
             dictionary_line
            ]
    with open(name+"/README.txt", "w") as file:
        file.write('\n'.join(lines))
        print('Readme saved on:', name)

    