import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_train_test_split(df, 
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