import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join
from itertools import groupby
from operator import itemgetter

from maad.util import read_audacity_annot
from sklearn.model_selection import train_test_split

from utils import batch_format_rois
from utils import batch_write_samples

def load_annotations(path_annot, 
                     verbose=False):
    
    """
    Load all audacity annotations on a folder
    
    Parameters
    ----------
    path_annot : str
        Path where annotations are located
    ...
    Returns
    -------
    df_all_annotations : pandas.core.frame.DataFrame
        Dataframe composed of the annotations from audacity
    """
    
    annotation_files = [f for f in listdir(path_annot) if isfile(join(path_annot, f))]
    """
    It is clean and useful use this part?
    if verbose:
        print('Number of files:',len(annotation_files))
        files = [i.split('.')[-1] for i in annotation_files]
        print('Fortmats:',set(files))
        print()
        print('Frequency of files:',pd.Series(files).value_counts())
        files_names = [i.split('.')[0] for i in annotation_files]
        print()
        print('Unique names:',len(files_names))
    """
    fnames_list = []
    df_all_annotations = pd.DataFrame()
    for file in annotation_files:
        # It is clean and useful use this part?
        #y, sr = sound.load(recordings_folder+file.split('.')[0]+'.wav')
        #duration = round(get_duration(y=y, sr=sr))
        #if sr != 22050:
        #    print(sr, file)
        #if duration != 60:
        #    print(duration, file)
        #Sxx_power, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=1024//2)
        #Sxx_db = power2dB(Sxx_power) + 96 # why 96?
        df_annotation_file = read_audacity_annot(path_annot+file) 
        #df_rois = format_features(df_rois, tn, fn) # neccesary????????????
        fnames_list.extend([file.split('.')[0]]*df_annotation_file.shape[0])
        df_all_annotations = df_all_annotations.append(df_annotation_file,ignore_index=True) 
    
    df_all_annotations.insert(loc=0, column='fname', value=fnames_list)
    df_all_annotations['min_t'] = np.floor(df_all_annotations['min_t'])
    df_all_annotations['max_t'] = np.ceil(df_all_annotations['max_t'])
    df_all_annotations = df_all_annotations.sort_values(by=['fname','min_t','max_t'],ignore_index=True)
    
    # This part could be included in the dashboard or other part!
    #df_all_annotations[['site','date']] = df_all_annotations['fname'].str.split('_',1)
    #df_all_annotations['date'] = df_all_annotations['date'].str.split('_').apply(lambda x: x[0]+x[1])
    #df_all_annotations['date'] = pd.to_datetime(df_all_annotations['date'])
    
    #df_all_annotations[['label','quality']] = df_all_annotations_['label'].str.split('_',expand=True)
    #df_all_annotations['label'] = df_all_annotations['label'].replace({'BPAFAB':'BOAFAB','PHUCUV':'PHYCUV'})
    #df_all_annotations['quality'] = df_all_annotations['quality'].replace({'FAR':'F','MED':'M','CLR':'C'})
    #df_all_annotations['label_duration'] = df_all_annotations['max_t'] - df_all_annotations['min_t']

    return df_all_annotations


def get_absence_slots_from_presence_rois(df, 
                                        wl, 
                                        max_duration):
    
    """
    Compute the complement of ROIs in a dataframe. It is assumed that complemement means 'ABSENCE' class
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe where each row correspond to one sample of the dataset. Each row is a ROI that
        comes from audacity annotations. The columns are 'min_t', 'max_t', 'fname', and 'label' 
    wl : int
        Window length. In general is the same as the window lenght used in df
    max_duration : int
        Total length of raw recordings. It assumes that all recordings have the same duration
    Returns
    -------
    df_absence_slots : pandas.core.frame.DataFrame
        Dataframe composed of the recordings complement of df. The label is 'ABSENCE'
    """
    
    df['segment_label'] = df.apply(lambda x: list(range(int(x['min_t']),int(x['max_t']+1))),axis=1)
    df = df.groupby(['fname'])['segment_label'].apply(sum).to_frame().reset_index()
    df['absence_space'] = df['segment_label'].apply(lambda x: sorted(set(range(max_duration+1))-set(x)))
    
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


def get_available_files(wav_path,
                       report=False):
    
    """
    Compute the complement of ROIs in a dataframe. It is assumed that complemement means 'ABSENCE' class
    
    Parameters
    ----------
     wav_path : str
        Path of folder that contains recording files. We expect .wav format 
     report : bool
         Print report of coherence between planilha and recordings
     Returns
    -------
    df_absence_rois_from_planilha : pandas.core.frame.DataFrame
        Dataframe composed of the recordings complement of df. The label is 'ABSENCE'   
    """
    df_planilha = pd.read_excel('data/Planilha_INCT_Anderson_Selvino.xlsx',
                                engine='openpyxl')
    
    files_in_planilha = set(df_planilha['gravacao_id'])
    files_in_folder = set([f.split('.wav')[0] for f in listdir(wav_path) if isfile(join(wav_path, f))])
    file_in_both = files_in_planilha & files_in_folder
    
    #if report:
        # df_planilha['absence'].value_counts()
        # check if gravacao in planilha not annotated
        # gravacao_in_planilha - 
        # check if annotated file not in planilha
        # - gravacao_in_planilha
        # check if some recording not identified in planilha
        # recordings-(absence_in_planilha | gravacao_in_planilha)
        # check if some recording not identified in planilha
        # (absence_in_planilha | gravacao_in_planilha)-recordings
    
    return list(file_in_both)


def get_absence_slots_from_planilha(n_sample,
                                    wav_path, # call function or use output as parameter?
                                    wl,
                                    max_duration,
                                    site,
                                    labels_cols):
    
    """    
    Compute the complement of ROIs in a dataframe. It is assumed that complemement means 'ABSENCE' class
    
    Parameters
    ----------
    n_sample : int
        gasas
     wav_path : str
        Path of folder that contains recording files. We expect .wav format 
    wl : int
        Window length. In general is the same as the window lenght used in df
    max_duration : int
        Total length of raw recordings. It assumes that all recordings have the same duration
    site : list
        Total length of raw recordings. It assumes that all recordings have the same duration
    labels_cols : list
        Daticos
    Returns
    -------
    df_absence_rois_from_planilha : pandas.core.frame.DataFrame
        Dataframe composed of the recordings complement of df. The label is 'ABSENCE'   
    """
    # Add new species with a standarized name
    dictionary_of_species = {'Boa_fab':'BOAFAB', 
                             'Phy_cuv':'PHYCUV'}
    # Dont use quality annotation, just the species
    labels_cols = [i.split('_',1)[0] for i in labels_cols]
    available_recordings = get_available_files(wav_path)
    
    df_planilha = pd.read_excel('data/Planilha_INCT_Anderson_Selvino.xlsx',
                                engine='openpyxl')
    df_planilha = df_planilha.rename(columns=dictionary_of_species)
    df_planilha = df_planilha[df_planilha['gravador'].isin(site)]
    df_planilha = df_planilha[df_planilha['gravacao_id'].isin(available_recordings)]
    df_planilha = df_planilha[['gravacao_id']+labels_cols]
    df_planilha['label'] = df_planilha[labels_cols].sum(axis=1).apply(lambda x: 'ABSENCE' if x ==0 else 'PRESENCE')
    
    df_planilha_absence = df_planilha[df_planilha['label']=='ABSENCE']
    df_planilha_absence = df_planilha_absence[['gravacao_id','label']]
    df_planilha_absence = df_planilha_absence.rename(columns={'gravacao_id':'fname'})
    df_planilha_absence['min_t'] = 0
    df_planilha_absence['max_t'] = max_duration
    # check if we could delete next two lines, In which cases it would be useful?
    df_planilha_absence['min_f'] = np.nan
    df_planilha_absence['max_f'] = np.nan
    
    df_absence_rois_from_planilha = batch_format_rois(df_planilha_absence, wl=wl)
   
    return df_absence_rois_from_planilha.sample(n=n_sample)#, ignore_index=True)

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
        Formated regions of interest with fixed size
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