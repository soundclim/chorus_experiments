import pandas as pd
import numpy as np

from os import path
from tqdm import tqdm
from warnings import warn

from maad import sound
from librosa import feature


#def extract_features( ):
    
def spectrogram(df,
               path_audio,
               n_fft = 1024,
               hop_length = 256, 
               n_mels = 128,
               f_min = 50,
               f_max = 4000,
               power=1.0
               ):
    
    X_img = []  
    for idx_row, row in tqdm(df.iterrows()):
        full_path_audio = path.join(path_audio, row.sample_name)
        s, fs = sound.load(full_path_audio)
        S = feature.melspectrogram(y=s,sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power, fmin = f_min, fmax=f_max)
        X_img.append(S)
    print(len(X_img))
    print(X_img[0].shape)
    try:
        X_img = np.asarray(X_img)
        X_img = np.reshape(X_img, (X_img.shape[0],X_img.shape[1],X_img.shape[2],1))
        
        empty_img = [index for index,img in enumerate(X_img) if np.isnan(img).any()]
        if len(empty_img)>1:
            warn('Empty image!')
        else:
            print('Execution done. Final shape:',X_img.shape)
            return X_img

    except:
        warn('Check raw data!')
        return X_img
    
