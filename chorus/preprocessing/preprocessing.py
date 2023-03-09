
import numpy as np
import pandas as pd

from itertools import groupby
from operator import itemgetter
from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import make_archive
from warnings import warn

from maad.util import read_audacity_annot
from librosa import get_duration, load

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

from utils import batch_format_rois
from utils import batch_write_samples
from utils import readme_generator

def load_annotations(path_annot):
    
    """
    Load all audacity annotations on a folder
    
    Parameters
    ----------
    path_annot : str
        Path where annotations are located
    Returns
    -------
    df_all_annotations : pandas.core.frame.DataFrame
        Dataframe composed of the annotations from audacity
    """
    
    annotation_files = [f for f in listdir(path_annot) if isfile(join(path_annot, f))]
    
    fnames_list = []
    df_all_annotations = pd.DataFrame()
    # TODO: avoid for loops, list comprenhension and multiprocessing
    for file in annotation_files:
        df_annotation_file = read_audacity_annot(path_annot+file) 
        fnames_list.extend([file.split('.')[0]]*df_annotation_file.shape[0])
        df_all_annotations = pd.concat([df_all_annotations, df_annotation_file],ignore_index=True)
    
    df_all_annotations.insert(loc=0, column='fname', value=fnames_list)
    df_all_annotations['min_t'] = np.floor(df_all_annotations['min_t'])
    df_all_annotations['max_t'] = np.ceil(df_all_annotations['max_t'])

    df_all_annotations = df_all_annotations.sort_values(by=['fname','min_t','max_t'],ignore_index=True)
    
    return df_all_annotations

def get_absence_slots_from_presence_rois(df, 
                                        wl, 
                                          ):
    
    """
    Compute the complement of ROIs in a dataframe. It is assumed that complemement means 'ABSENCE' class
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe where each row correspond to one sample of the dataset. Each row is a ROI that
        comes from audacity annotations. The columns are 'min_t', 'max_t', 'fname', and 'label' 
    wl : int
        Window length. In general is the same as the window lenght used in df
    Returns
    -------
    df_absence_slots : pandas.core.frame.DataFrame
        Dataframe composed of the recordings complement of df. The label is 'ABSENCE'
    """

    df['segment_label'] = df.apply(lambda x: list(range(int(x['min_t']),int(x['max_t']+1))),axis=1)
    df_segment = df.groupby(['fname'])['segment_label'].apply(sum).to_frame().reset_index()
    df_max = df.groupby(['fname'])['max_t'].max().to_frame() # if this is the max of the annotation, we should change for get_duration 
    df = pd.merge(df_segment, df_max, on='fname', how='left')
    df['absence_space'] = df.apply(lambda x: sorted(set(range(int(x['max_t'])+1))-set(x['segment_label'])),axis=1)

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


def get_available_files(site,
                        wav_path,
                       report=False):
    
    """
    Compute the complement of ROIs in a dataframe. It is assumed that complemement means 'ABSENCE' class
    
    Parameters
    ----------
     wav_path : str
        Path of folder that contains recording files. We expect .wav format 
     report : bool
         Print reportmake_archive of coherence between planilha and recordings
     Returns
    -------
    df_absence_rois_from_planilha : pandas.core.frame.DataFrame
        Dataframe composed of the recordings complement of df. The label is 'ABSENCE'   
    """
    planilha_path = 'data/raw/site_name/Planilha.xlsx'
    planilha_path = planilha_path.replace('site_name',site)
    
    df_planilha = pd.read_excel(planilha_path,
                                engine='openpyxl')
    
    files_in_planilha = set(df_planilha['gravacao_id'])
    files_in_folder = set([f.split('.wav')[0] for f in listdir(wav_path) if isfile(join(wav_path, f))])
    
    files_in_both = list(files_in_planilha & files_in_folder)
    files_in_both = [f for f in files_in_both if isinstance(f, str)]
    # TODO: Fix corrupted files
        
    """
    corrupted_files =  ['INCT20955_20200220_033000',
                         'INCT20955_20200314_050000',
                         'INCT20955_20200315_011500',
                         'INCT20955_20200325_221500',
                         'INCT20955_20200326_011500',
                         'INCT20955_20200326_050000',
                         'INCT20955_20200326_180000',
                         'INCT20955_20200331_041500',
                         'INCT20955_20200331_201500',
                         'INCT20955_20200331_221500',
                         'INCT20955_20200407_020000',
                         'INCT20955_20200407_200000',
                         'INCT20955_20200407_230000',
                         'INCT20955_20200411_001500',
                         'INCT20955_20200411_031500',
                         'INCT20955_20200411_191500',
                         'INCT20955_20200411_221500',
                         'INCT20955_20200417_174500',
                         'INCT20955_20200420_193000']
     """
        
    files_in_both, max_duration = map(list,zip(*[(file,get_duration(load(join(wav_path, 
                                    file)+'.wav')[0],22050)) for file in files_in_both]))
    files_in_both, max_duration = map(list,zip(*[(files_in_both[index],
                                    max_duration[index]) for index, time in enumerate(max_duration) if time>57]))
        
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
    
    return files_in_both, max_duration


def get_absence_slots_from_planilha(n_sample,
                                    wav_path, # call function or use output as parameter?
                                    wl,
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
                             'Phy_cuv':'PHYCUV',
                             'Boa_alb':'BOAALB',
                             'Boa_lun':'BOALUN',
                             'Den_cru':'DENCRU',
                             'Phy_mar':'PHYMAR',
                             'Pit_azu':'PITAZU',}
    others = {'Boa_bisc':'	',
                             'Sci_pere':' '	,
                             'Spe_surd':' ',	
                             'Den_Nadh':' ',	
                             'Lep_latr':' ',	
                             'Rhi_ict':' ',	
                             'Den_min':' ',	
                             'Boa_pra':' ',	
                             'Ela_bic':' ',	
                             'Boa_lep':' ',	
                             'Phy_nan':' ',	
                             'Lep_pla':' ',	
                             'Pro_boi':' ',	
                             'Ade_sp1':' ',	
                             'Fri_mit':' ',	
                             'Ade_sp2':' ',	
                             'Rhi_abe':' ',	
                             'Lit_cat':' ',	
                             'Apl_per':' ',	
                             'Pse_car':' ',	
                             'Sci_gra':' ',	
                             'Phy_gra':' ',	
                             'Olo_ber':' ',}

    # Dont use quality annotation, just the species
    labels_cols = list(set([i.split('_',1)[0] for i in labels_cols]))
    available_recordings, max_duration = get_available_files(site=site,wav_path=wav_path)
    
    d_max_duration = {'fname': available_recordings, 'max_t': max_duration}
    df_max_duration = pd.DataFrame(data=d_max_duration)
    
    planilha_path = 'data/raw/site_name/Planilha.xlsx'
    planilha_path = planilha_path.replace('site_name',site)
    
    df_planilha = pd.read_excel(planilha_path,
                                engine='openpyxl')
    df_planilha = df_planilha.rename(columns=dictionary_of_species)
    df_planilha = df_planilha[df_planilha['gravador'].isin([site])]
    df_planilha = df_planilha[df_planilha['gravacao_id'].isin(available_recordings)].drop_duplicates(subset=['gravacao_id'])
    labels_cols_in_site = list(set(df_planilha.columns)& set(dictionary_of_species.values()))
    
    df_planilha = df_planilha[['gravacao_id']+labels_cols_in_site]
    df_planilha['label'] = df_planilha[labels_cols_in_site].sum(axis=1).apply(lambda x: 'ABSENCE' if x ==0 else 'PRESENCE')
    
    df_planilha_absence = df_planilha[df_planilha['label']=='ABSENCE']
    df_planilha_absence = df_planilha_absence[['gravacao_id','label']]
    df_planilha_absence = df_planilha_absence.rename(columns={'gravacao_id':'fname'})
    df_planilha_absence['min_t'] = 0
    df_planilha_absence = pd.merge(df_planilha_absence,df_max_duration,on='fname', how='left')
    # check if we could delete next two lines, In which cases it would be useful?
    df_planilha_absence['min_f'] = np.nan
    df_planilha_absence['max_f'] = np.nan
    
    df_absence_rois_from_planilha = batch_format_rois(df_planilha_absence, wl=wl,wav_path=wav_path)
    
    print(df_absence_rois_from_planilha.shape)
    
    if n_sample>df_absence_rois_from_planilha.shape[0]:
        message = 'We want ' +str(n_sample)+ 'samples but only have'+ str(df_absence_rois_from_planilha.shape[0])
        warn(message)
        return df_absence_rois_from_planilha
    else:
        return df_absence_rois_from_planilha.sample(n=n_sample)#, ignore_index=True)

def stratified_split_train_test(df, 
                                x_name, 
                                y_name,
                                test_size=0.2):
    
    """
    WARNING: PRONE TO DATA LEAKAGE WITH RECORDINGS
    Split dataset using stratified sampling and add a column called 'subset' 
    where specify if sample is in train or test subset.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        pandas Dataframe where each row correspond to one sample of the dataset
    x_name : str
        Name in columns to have a unique identifier in df
    y_name : str
        Name in columns of labels in df. This column is used to stratify
    test_size : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
        If int, represents the absolute number of test samples. If None, the value is set to the complement of the train 
        size. Sames as scikit-learn
    Returns
    -------
    df_compiled : pandas.core.frame.DataFrame
        Formated regions of interest with fixed size
    """
    from sklearn.model_selection import train_test_split

    
    X = df[x_name]
    y = df[y_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    df_train = df.loc[X_train.index,:]
    df_train['subset'] = 'train'
    df_test = df.loc[X_test.index,:]
    df_test['subset'] = 'test'

    df_compiled = pd.concat([df_train, df_test], axis=0)
    df_compiled.sort_values('sample_name', inplace=True, ignore_index=True)

    return df_compiled



def assign_cross_validations_folds(df,
                                 x_name, 
                                 y_name,
                                 column_group_name,
                                 n_folds=5):
    
    """
    Split dataset using StratifiedGroupKFold. Add a column called 'subset' 
    where specify if sample is in train or test subset. Add one column called
    'fold' where 0 fold is the test set.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        pandas Dataframe where each row correspond to one sample of the dataset
    x_name : str
        Name in columns to have a unique identifier in df
    y_name : str
        Name in columns of labels in df. This column is used to stratify
    column_group_name : str
        Name of column to group in df.
    n_folds : int, defaul t=5
        Number of folds. Must be at least 2.
    Returns
    -------
    df : pandas.core.frame.DataFrame
        pandas Dataframe with columns of fold and subset
                                   
    """

    X = df[x_name]
    y = df[y_name]
    
    groups = df[column_group_name]
    sgkf = StratifiedGroupKFold(n_splits=n_folds+1)
    
    for folder_number, split_inds in enumerate(sgkf.split(X, y, groups=groups)):
        test_inds = split_inds[-1]
        df.loc[test_inds, 'fold'] = [folder_number]*len(test_inds)
    
    df['subset'] = df['fold'].apply(lambda x: 'test' if x==0 else 'train')

    return df


        
def build_dataset(wl, 
                 target_sr, 
                 flims, 
                 site_list,
                 path_save, 
                 labels_cols,
                 prefix='SAMPLE_',
                 ):
    """
    #SHOULD WE USE THIS PART AS A CLASS?
    Create a dataset from raw fixed time recordings and annotations from audacity 
    
    Parameters 
    ---------- 
    wl : int
        Fixed window lenght to split the recording
    target_sr : int
        Sampling rate to convert the audio file
    flims : list
        tuple composed of (minimun_frequency, maximun_frequency) in Hz
    site_list : list
        Passive Acoustic Monitoring device
    path_save : str
        Path where the last folder is the place where preprocessed recordings and df_compiled is saved
    labels_cols : list
        list with str labels used in Audacity annotations
    prefix : str
        name of preprocessed recordings
        
    Returns 
    ------- 
    df_dataset_cv : pandas.core.frame.DataFrame
        dataframe composed of each sample in the dataset, labels 
    """
    df_all = []
    for site in site_list:
        annotation_path = 'data/raw/site_name/annotations/'
        annotation_path = annotation_path.replace('site_name',site)
        wav_path = 'data/raw/site_name/recordings/'
        wav_path = wav_path.replace('site_name',site)
        df_all_annotations = load_annotations(path_annot=annotation_path)
            
    
        df_all_annotations = df_all_annotations[df_all_annotations['label'].isin(labels_cols)]

        df_presence_rois = batch_format_rois(df=df_all_annotations, 
                                             wl=wl,
                                             wav_path=wav_path)

        df_absence_slots = get_absence_slots_from_presence_rois(df=df_presence_rois, 
                                                               wl=wl)

        df_absence_in_presence_files = batch_format_rois(df=df_absence_slots, 
                                                         wl=wl,
                                                        wav_path=wav_path)

        presence_samples = df_presence_rois.shape[0]
        absence_samples_in_presence_files = df_absence_in_presence_files.shape[0]
        absence_samples_in_absence_files = presence_samples - absence_samples_in_presence_files
        
        print(presence_samples, absence_samples_in_presence_files, absence_samples_in_absence_files)

        if absence_samples_in_absence_files>0:

            df_dataset_presence = pd.concat([df_presence_rois,df_absence_in_presence_files])

            df_absence_rois_from_planilha = get_absence_slots_from_planilha(n_sample=absence_samples_in_absence_files,
                                                              wav_path=wav_path, 
                                                              wl=wl,
                                                              site=site,
                                                              labels_cols=labels_cols)

            df_dataset = pd.concat([df_dataset_presence, df_absence_rois_from_planilha],
                               ignore_index=True)
        else:
            df_absence_in_presence_files = df_absence_in_presence_files.sample(n=presence_samples)

            df_dataset = pd.concat([df_presence_rois,df_absence_in_presence_files],
                                  ignore_index=True)
        
        df_all.append(df_dataset)

            
    df_dataset_concat = pd.concat(df_all, ignore_index=True)
    
    df_dataset_concat[['species','quality']] = df_dataset_concat['label'].str.split('_',1,expand=True)
    df_dataset_concat[['site','date']] = df_dataset_concat['fname'].str.split('_',1,expand=True)
    df_dataset_concat['label'] = df_dataset_concat['label'].str.replace('_C$', '_M',regex=True)
    df_dataset_concat['class'] = df_dataset_concat['site'] + '_' + df_dataset_concat['label']
    df_dataset_concat['class_'] = df_dataset_concat['class']
    df_dataset_concat['date'] = df_dataset_concat['date'].str.split('_').apply(lambda x: x[0]+x[1])
    df_dataset_concat['date'] = pd.to_datetime(df_dataset_concat['date'])
    
    dataset_size = df_dataset_concat.shape[0]
    exponent_of_10 = int(np.ceil(np.log10(dataset_size)))
    sample_names = prefix + df_dataset_concat.index.astype(str).str.zfill(exponent_of_10)  + '_' + df_dataset_concat['class'].values + '.wav'
    
    df_dataset_concat.insert(loc=0, 
                          column='sample_name', 
                          value=sample_names)
    
    df_dataset_cv = assign_cross_validations_folds(df=df_dataset_concat, 
                                                 x_name='sample_name', 
                                                 y_name='class',
                                                 column_group_name='fname')

    df_dataset_cv['dummy'] = 1 
    columns_name = ['sample_name','fname','min_t','max_t','label',
                    'species','quality','site','date','class','fold','subset']
    df_dataset_cv = df_dataset_cv.pivot(index=columns_name, 
                                                columns='class_', 
                                                values='dummy').fillna(0).reset_index()
    df_dataset_cv = df_dataset_cv.rename_axis(None, axis=1)    

    complete_name_function = lambda x: x['sample_name'].split('.wav')[0] + '_FOLD_'+str(int(x['fold']))+'.wav'
    df_dataset_cv['sample_name'] = df_dataset_cv.apply(complete_name_function,axis=1)
    
    columns_as_int = ['min_t','max_t','fold'] + list(df_dataset_cv.columns[12:])
    df_dataset_cv[columns_as_int] = df_dataset_cv[columns_as_int].astype(int)
        
    print('Finalized df construction')
    
    path_save_audio = join(path_save, 'audio')
 
    if not exists(path_save_audio):
        makedirs(path_save_audio)
        print('Folder created:', path_save_audio)
        
    readme_generator(df_dataset_cv,name=path_save,sr=target_sr,wl=wl,flims=flims)
      
    for site in site_list:
        
        print('Saving files for site:',site)
        wav_path = 'data/raw/site_name/recordings/'
        wav_path = wav_path.replace('site_name',site)
    
        batch_write_samples(df_dataset_cv[df_dataset_cv['site']==site], 
                        wav_path=wav_path, 
                        target_sr=target_sr,
                        path_save=path_save_audio,
                        flims=flims, 
                        verbose=True)
    print('Last checking')
    samples_len = len([f for f in listdir(path_save_audio) if isfile(join(path_save_audio, f))])
    df_len = df_dataset_cv.shape[0]

    if samples_len==df_len:
        df_dataset_cv.to_csv(path_save+'/df_train_test_files.csv', index=False)
        print("Datased created in:", path_save) 
        # TODO: save parameters   
                
        make_archive(path_save,
                    'zip',
                    path_save.rsplit('/',1)[0],
                    path_save.split('/')[-1])
        return df_dataset_cv   
    else:
        warn("Different size between recordings extracted and samples!!")
        print('dataset samples:',df_len)
        print('audio samples:',samples_len)
        return df_dataset_cv 

