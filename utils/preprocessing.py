
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