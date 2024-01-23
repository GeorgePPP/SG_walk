import pandas as pd
from sklearn.preprocessing import StandardScaler

def removeLowVar(df):
    # Define the threshold for low variability
    threshold = 0.1
    # Initialize lists to store low-variability columns and their variances
    low_variability_columns = []
    variances = []
    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        # Calculate the variance of the column
        variance = df[column].var()
        # Check if the variance is below the threshold
        if variance < threshold:
            low_variability_columns.append(column)
            variances.append(variance)
    return df.drop(df[low_variability_columns], axis=1)
    
def oneHotEncode(df):
    colsToEncode = []
    for col in df.columns:
        if len(df[col].unique()) <= 10 and len(df[col].unique()) > 2: # Exclude continuous variables and binary variables
            colsToEncode.append(col)

    return pd.get_dummies(df, columns=colsToEncode, drop_first=True)

def upSample(df):
    if 'flourishing' in df.columns:
        total = len(df['flourishing'])
        # Calculate the class frequencies
        class_frequencies = df['flourishing'].value_counts(normalize=True).to_dict()
        if any([ratio for ratio in class_frequencies.values() if ratio < 0.5 or ratio >= 0.6]):
            # Find the class with the largest ratio
            maxRatioClass = max(class_frequencies, key=class_frequencies.get)
            otherClass = [cls for cls in class_frequencies.keys() if cls != maxRatioClass]
            maxRatioClassRecord = int(class_frequencies[maxRatioClass] * total)

            for cls in otherClass:
                # Create a sliced DataFrame for the class with the largest ratio
                sliced_df = df[df['flourishing'] == cls]
                thisCLassRecord = len(sliced_df)
                numOfSample = maxRatioClassRecord - thisCLassRecord
                df = pd.concat([df, sliced_df.sample(n=numOfSample, replace=True)], ignore_index=True)
        return df

    else:
        print("The 'flourishing' column does not exist in the DataFrame.")
        return df  # Return the original DataFrame and frequencies

def scaleContVar(df):
    colsToScale = []
    for col in df.columns:
        if len(df[col].unique()) > 2: # Exclude binary variables
            colsToScale.append(col)

    scaler = StandardScaler()
    df[colsToScale] = scaler.fit_transform(df[colsToScale])

    return df

# This function filters out participants that have same wellbeing status for 4 weeks
def filterSameParticipants(df):
    participants = []
    participants_ls = list(df['Participant ID'].unique())

    for participant in participants_ls:
        participants.append(participant)
        flourishing = list(df[df['Participant ID'] == participant]['flourishing'])
        moderate = list(df[df['Participant ID'] == participant]['moderate'])
        if all(ele == flourishing[0] for ele in flourishing) and all(ele == moderate[0] for ele in moderate):
            participants_ls.remove(participant)

    new_df = df[df['Participant ID'].isin(participants_ls)]
    return new_df