import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

def extractData(df):
    # Correct the column names
    for key in df.keys():
        chars = ' ():%°'
        old_column_name = key
        if any((c in chars) for c in old_column_name):
            new_column_name = old_column_name.replace(" ", "_")
            new_column_name = new_column_name.replace("°", "deg")
            new_column_name = new_column_name.replace("(", "_")
            new_column_name = new_column_name.replace(")", "")
            new_column_name = new_column_name.replace(":", "")
            new_column_name = new_column_name.replace("%", "")
            df[new_column_name] = df[old_column_name]

            df = df.drop(labels=[old_column_name], axis=1)

    # Drop duplicate entries
    df = df.drop_duplicates()

    # Calculate magnitude of acceleration
    df['total acceleration'] = np.sqrt(df['Acceleration_X_g']**2 + df['Acceleration_Y_g']**2 + df['Acceleration_Z_g']**2)

    # Create a new DataFrame with intended columns
    new_df = df[['Week', 'Label', 'Game_Type', 'total acceleration']].copy()

    # Convert ChipTime to datetime and get UTC timestamp
    # new_df['ChipTime_UTC'] = pd.to_datetime(new_df['SysTime'], format='%m/%d/%Y %H:%M')

    # Sort the DataFrame by ChipTime_UTC
    # new_df.sort_values(by='ChipTime_UTC', inplace=True) 

    # Reset the index of the DataFrame
    new_df.reset_index(drop=True, inplace=True)

    # Drop the original ChipTime column
    # new_df.drop(columns='SysTime', inplace=True)

    # new_df['ChipTime_UTC'] = pd.to_datetime(new_df['ChipTime_UTC'])
    
    # new_df['ChipTime_UTC'] = new_df['ChipTime_UTC'].dt.strftime('%d %b')

    return new_df
