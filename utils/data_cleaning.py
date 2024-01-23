import numpy as np

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

    if 'Game_Type' in df.columns:
        # Create a new DataFrame with intended columns
        new_df = df[['Week', 'Game_Type', 'total acceleration']].copy()
    else:
        df['Game_Type'] = 'Exercise'
        new_df = df[['Week', 'Game_Type', 'total acceleration']].copy()

    # Reset the index of the DataFrame
    new_df.reset_index(drop=True, inplace=True)

    return new_df
