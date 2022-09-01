import numpy as np
import pandas as pd

from itertools import groupby
from operator import itemgetter

from sklearn.model_selection import train_test_split


def get_absence_slots_from_presence_rois(df, 
                                        wl, 
                                        max_t):
    
    """
    
    Compute the complement of ROIs in a dataframe. It is assumed that complemement means 'ABSENCE' class
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe where each row correspond to one sample of the dataset. Each row is a ROI that
        comes from audacity annotations. The columns are 'min_t', 'max_t', 'fname', and 'label' 
    wl : int
        Window length. In general is the same as the window lenght used in df
    max_t : int
        Total length of raw recordings. It assumes that all recordings have the same duration
    Returns
    -------
    df_absence_slots : pandas.core.frame.DataFrame
        Dataframe composed of the recordings complement of df. The label is 'ABSENCE'
        
    """
    
    df['segment_label'] = df.apply(lambda x: list(range(int(x['min_t']),int(x['max_t']+1))),axis=1)
    df = df.groupby(['fname'])['segment_label'].apply(sum).to_frame().reset_index()
    df['absence_space'] = df['segment_label'].apply(lambda x: sorted(set(range(max_t+1))-set(x)))
    
    absence_slots = []
    for idx, x in df.iterrows():
        sub_segments = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(x['absence_space']),
                                                                        lambda i_x: i_x[0] - i_x[1])]
        absence_slots.extend([[x['fname'],'ABSENCE',min(i),max(i)] for i in sub_segments if len(i) >= wl])
        
    df_absence_slots = pd.DataFrame(absence_slots,columns=['fname','label','min_t','max_t'])
    # check if we could delete next two lines, In which cases it would be useful?
    df_absence_slots['min_f'] = np.nan
    df_absence_slots['max_f'] = np.nan
    
    return df_absence_slots

def stratified_split_train_test(df, 
                                x_name, 
                                y_name):
    
    """
    
    Split dataset using stratified sampling and add a column called 'dataset' 
    where specify if sample is in train or test subset.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        pandas Dataframe where each row correspond to one sample of the dataset
    x_name : str
        Name in df of x values or id
    y_name : str
        Name in df of y values or annotation. This value is used to stratify 
    Returns
    -------
    df_compiled : pandas.core.frame.DataFrame
        Formated regions of interest with fixed size.
        
    """

    X = df[x_name]
    y = df[y_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    df_train = df.loc[X_train.index,:]
    df_train['dataset'] = 'train'
    df_test = df.loc[X_test.index,:]
    df_test['dataset'] = 'test'

    df_compiled = pd.concat([df_train, df_test], axis=0)
    df_compiled.sort_values('sample_name', inplace=True, ignore_index=True)

    return df_compiled