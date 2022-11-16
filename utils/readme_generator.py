import pandas as pd

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