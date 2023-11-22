import metrics
import pandas as pd
from tqdm import tqdm
import numpy as np

def flourishing(row):
    emotion = row[["Q1", "Q2", "Q3", "Q4"]]
    psy_soc = row[["Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"]]
    count_emo = 0
    count_psy_soc = 0
    for col in emotion:
        if col >= 5:
            count_emo += 1
    for col in psy_soc:
        if col >= 5:
            count_psy_soc += 1
    if count_emo >= 1 and count_psy_soc >= 6:
        return 1
    return 0

def languishing(row):
    emotion = row[["Q1", "Q2", "Q3", "Q4"]] # emotion score test
    psy_soc = row[["Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"]] # social and psychological score test
    count_emo = 0
    count_psy_soc = 0
    for col in emotion:
        if col == 1 or col == 2:
            count_emo += 1
    for col in psy_soc:
        if col == 1 or col == 2:
            count_psy_soc += 1
    if count_emo >= 1 and count_psy_soc >= 6:
        return 1
    return 0


def aggregateAll(result):   
    metrics = []
    # Loop through the nested dictionaries (if any) and top-level values
    for child in result.values():
        if isinstance(child, dict):
            for value in child.values():
                metrics.append(value)
        else:
            metrics.append(child)

    # Convert the list to a NumPy array
    data_array = np.array(metrics)

    # Calculate the mean of the entire flattened list
    aggregated_value = np.mean(data_array)

    # Print the aggregated value
    return aggregated_value

def main():
    # Participants whose accelerations data are available
    participant_acc_ls = list(range(1, 56))

    # These two csv files contain all participants' information
    survey_data = pd.read_csv(r'C:\Users\User\Desktop\SG_walk\Mental Health data (Including condition 3&4).csv', encoding='unicode-escape')
    demographic_data = pd.read_csv(r"C:\Users\User\Desktop\SG_walk\Demographic.csv", encoding='unicode-escape')

    # Filter the DataFrame based on 'Dropout condition' being equal to 0
    filtered_survey = survey_data[survey_data['Dropout condition'] == 0]

    # Get unique participant IDs from the filtered DataFrame
    participant_survey_ls = list(filtered_survey['Participant ID'].unique())
    participant_demo_ls = list(demographic_data['id'].unique())
    participant_ls = list(set(participant_acc_ls) & set(participant_demo_ls) & set(participant_survey_ls))

    # Initialize empty list for dataframe
    # Motion data
    participant_ids = []  
    max_acc = []
    calories_burnt = []
    action_count = []
    weeks = []
    games = []

    # Survey data
    emotional = []
    psychological = []
    social = []
    flourish = []
    languish = []
    moder = []

    # Demographic data
    alone = []
    collab= []
    peer = []
    coach = []
    gender = []
    age = []
    income = []
    edu = []
    selfses = []

    for participant in (participant_ls):
        print("Retrieving data for participant {}".format(participant))

        # Get motion metrics for this participant
        motion_file = r'C:\Users\User\Desktop\SG_walk\Participant wise acceleromter data\Participant ' + str(participant) + r'.csv'
        participant_motion = metrics.metrics(motion_file) # A dictionary
        action_result = participant_motion.getActionCount()
        calorie_result = participant_motion.getCalorieCount()
        max_acc_result = participant_motion.getMaxAcceleration()

        # Extract information from the motion metrics
        for game in participant_motion.games:
            demographic_row = demographic_data[(demographic_data['id'] == participant)].iloc[0]
            for week in participant_motion.weeks:
                # Get int value for 'week' column
                if not isinstance(week, int):
                    int_week = (int([*week][-1]))
                else:
                    int_week = week
                # Slice data for this 'participant' and 'week'
                mask = (survey_data['Participant ID'] == participant) & (survey_data['Week'] == int_week)
                if not survey_data.loc[mask].empty:
                    survey_row = survey_data.loc[mask].iloc[0]
                else:
                    continue

                # Create columns for the dataframe
                participant_ids.append(participant)
                games.append(game)
                weeks.append(int_week)
                action_count.append(int((action_result[game][week])))
                calories_burnt.append(round((calorie_result[game][week]), 2))
                max_acc.append(round((max_acc_result[game][week]), 2))

                # Append demographic data from the stored row
                alone.append(demographic_row['Game'])
                collab.append(demographic_row['Collaboration'])
                peer.append(demographic_row['Peer'])
                coach.append(demographic_row['HealthCoach'])
                gender.append(demographic_row['Gender'])
                age.append(demographic_row['Age'])
                income.append(demographic_row['Income'])
                edu.append(demographic_row['Education'])
                selfses.append(demographic_row['SelfSES'])

                # Append survey data from the stored row
                # (1 = elderly-health coach; 2 = elderly-elderly; 3 = elderly single; 4 = elderly exercise)
                emotional.append(int(survey_row['Emotional']))
                psychological.append(int(survey_row['Psychological']))
                social.append(int(survey_row['Social']))
                flourish.append(flourishing(survey_row))
                languish.append(languishing(survey_row))
                if flourishing(survey_row) == 0 and languishing(survey_row) == 0: 
                    moder.append(1)
                else:
                    moder.append(0)
        print("Done retrieving data for participant {}".format(participant))
                
    # Create a DataFrame from the extracted data
    motion_df = pd.DataFrame({
        "Participant ID": participant_ids,
        "Game Type": games,
        "Week": weeks,
        "Calories Burnt": calories_burnt,
        "Action Count": action_count,
        "Max Acceleration": max_acc,
        "Alone": alone,
        "Collaboration": collab,
        "Peer": peer,
        "HealthCoach": coach,
        "Gender": gender,
        "Age": age,
        "Income": income,
        "Education": edu,
        "SelfSES": selfses,
        "Emotional": emotional,
        "Psychological": psychological,
        "Social": social,
        "languishing": languish,
        "flourishing": flourish,
        "moderate": moder,
    })

    motion_df.to_csv(r'C:\Users\User\Desktop\SG_walk\cache_data\data.csv')
    print(motion_df)

if __name__ == "__main__":
    main()
 
# If aggregation is needed
# agg_dict_survey = {
# 'Companion': 'first',
# 'Q1': 'mean',
# 'Q2': 'mean',
# 'Q3': 'mean',
# 'Q4': 'mean',
# 'Emotional': 'mean',
# 'Q5': 'mean',
# 'Q6': 'mean',
# 'Q7': 'mean',
# 'Q8': 'mean',
# 'Q9': 'mean',
# 'Q10': 'mean',
# 'Psychological': 'mean',
# 'Q11': 'mean',
# 'Q12': 'mean',
# 'Q13': 'mean',
# 'Q14': 'mean',
# 'Q15': 'mean',
# 'Social': 'mean',
# 'Dropout condition': 'first'
# }

# survey_data = survey_data.groupby(['Participant ID']).agg(agg_dict_survey).reset_index()
# print("Survey data has been aggregated into participant-specific data")
