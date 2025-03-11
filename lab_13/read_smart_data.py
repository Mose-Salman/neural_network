import pandas as pd
import numpy as np
 
def read_SMARD_data(path, remove_bad_columns=False):
    """ Read SMARD data .csv file to a pandas Dataframe

        Input:
        path: file path to SMARD data .csv file

        Returns:
        pandas Dataframe with SMARD data
    """
    # save data from .csv file to dataframe
    df = pd.read_csv(path, delimiter=';', thousands='.', decimal=',', parse_dates=[[0,1]], dayfirst="True")

    # rename columns
    df = df.rename(
        columns={
        'Datum_Anfang': "Date",
        'Gesamt (Netzlast) [MWh] Originalauflösungen': 'Total Load [MWh]',
        'Residuallast [MWh] Originalauflösungen': 'Residual Load [MWh]',
        'Pumpspeicher [MWh] Originalauflösungen' : 'Energy from Pumped Storage [MWh]'
        }
    )
    if remove_bad_columns==True:
        # remove columns from Dataframe
        df = df.drop(['Residual Load [MWh]', 'Energy from Pumped Storage [MWh]'], axis="columns")
        df.pop('Ende')
    return df
 
def split_dataset(dataframe, train_start, train_end, validation_start, validation_end):
    train_set = dataframe[(dataframe['Date'] >= train_start) & (dataframe['Date'] <= train_end)]
    validation_set = dataframe[(dataframe['Date'] >= validation_start) & (dataframe['Date'] <= validation_end)]
    
    # Drop 'Date' column if it exists
    if 'Date' in train_set.columns:
        train_set = train_set.drop(['Date'], axis=1)
    if 'Date' in validation_set.columns:
        validation_set = validation_set.drop(['Date'], axis=1)
    
    # Drop 'Ende' column if it exists
    if 'Ende' in train_set.columns:
        train_set = train_set.drop(['Ende'], axis=1)
    if 'Ende' in validation_set.columns:
        validation_set = validation_set.drop(['Ende'], axis=1)
    
    train_set = train_set.to_numpy()
    validation_set = validation_set.to_numpy()
    
    return train_set, validation_set
 
def standardize_data(train_set, validation_set):
    mean = np.mean(train_set, axis=0)
    std = np.std(train_set, axis=0)
    std[std == 0] = 1
    
    train_set_standardized = (train_set - mean) / std
    validation_set_standardized = (validation_set - mean) / std
    
    return train_set_standardized, validation_set_standardized


def destandardize_data(train_set, validation_set):
    mean = np.mean(train_set, axis=0)
    std = np.std(train_set, axis=0)
    std[std == 0] = 1
    
    train_set_standardized = (train_set - mean) / std
    validation_set_standardized = (validation_set - mean) / std
    
    return train_set_standardized, validation_set_standardized
 
def create_data_windows(dataset, input_timesteps, output_timesteps):
    input_data = []
    label_data = []
    
    for i in range(len(dataset) - input_timesteps - output_timesteps + 1):
        input_data.append(dataset[i:(i + input_timesteps)])
        label_data.append(dataset[(i + input_timesteps):(i + input_timesteps + output_timesteps)])
    
    input_data = np.array(input_data)
    label_data = np.array(label_data)
    
    return input_data, label_data
 