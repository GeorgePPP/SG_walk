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

def moderate(row):
    if row['flourishing'] == 0 and row['languishing'] == 0:
        return 1
    return 0

def main():
    # Specify participants playing game or exercise
    participant_game_ls = list(range(1, 31)) + [31, 32, 33, 34, 35, 36, 37, 38, 40, 42, 47, 54]
    participant_game_ls.remove(19)
    participant_game_ls.remove(24)
    participant_game_ls.remove(27)
    participant_game_ls.remove(8)
    participant_game_ls.remove(35)
    print(participant_game_ls)

    # pariticpant_exer_ls = list(range(1, 56)) - participant_game_ls

    # Read motion, survey, demographic files
    survey_data = pd.read_csv(r'C:\Users\User\Desktop\SG_walk\Mental Health data (Including condition 3&4).csv', encoding='unicode-escape')
    demographic_data = pd.read_csv(r"C:\Users\User\Desktop\SG_walk\Demographic.csv", encoding='unicode-escape')

    # Initialize empty list for dataframe
    participant_ids = []  
    weeks = []
    games = []
    max_acc = []
    calories_burnt = []
    action_count = []

    # Survey data
    companion = []
    emotional = []
    psychological = []
    social = [] 

    # Demographic data
    grouping = []
    gender = []
    age = []
    income = []
    edu = []
    selfses = []
    
    print("Retrieving data for paticipant: ")
    for participant in tqdm(participant_game_ls):
        print("{} \n".format(participant))
        # Loop through the data dictionary and extract the information
        # Get data for motion metrics
        motion_file = r'C:\Users\User\Desktop\SG_walk\Participant wise acceleromter data\Participant ' + str(participant) + r'.csv'
        participant_motion = metrics.metrics(motion_file) # A dictionary
        action_result = participant_motion.getActionCount()
        calorie_result = participant_motion.getCalorieCount()
        max_acc_result = participant_motion.getMaxAcceleration()
        demographic_row = demographic_data[demographic_data['id'] == participant].iloc[0]
        for game, game_data in action_result.items():
            for week, result in game_data.items():
                print("Participant: {}; Week: {}".format(participant, week))
                if not isinstance(week, np.int64) and not isinstance(week, int):
                    int_week = (int([*week][-1]))
                else:
                    int_week = week
                string_week = week
                survey_row = survey_data[(survey_data['Participant ID'] == participant) & (survey_data['Week'] == int_week)].iloc[0]
                weeks.append(int_week)
                games.append(game)
                calories_burnt.append(calorie_result[game][string_week])
                action_count.append(action_result[game][string_week])
                max_acc.append(max_acc_result[game][string_week])
                participant_ids.append(participant)

                # Append demographic data from the stored row
                grouping.append(demographic_row['Grouping'])
                gender.append(demographic_row['Gender'])
                age.append(demographic_row['Age'])
                income.append(demographic_row['Income'])
                edu.append(demographic_row['Education'])
                selfses.append(demographic_row['SelfSES'])

                # Append survey data from the stored row
                    # (1 = elderly-health coach; 2 = elderly-elderly; 3 = elderly single; 4 = elderly exercise)
                companion.append(survey_row['Companion'])
                emotional.append(survey_row['Emotional'])
                psychological.append(survey_row['Psychological'])
                social.append(survey_row['Social'])

                
    # Create a DataFrame from the extracted data
    motion_df = pd.DataFrame({
        "Participant ID": participant_ids,
        "Week": weeks,
        "Game": games,
        "Calories Burnt": calories_burnt,
        "Action Count": action_count,
        "Max Acceleration": max_acc,
        "Grouping": grouping,
        "Gender": gender,
        "Age": age,
        "Income": income,
        "Education": edu,
        "SelfSES": selfses,
        "Companion": companion,
        "Emotional": emotional,
        "Psychological": psychological,
        "Social": social
    })

    motion_df.to_csv(r'C:\Users\User\Desktop\SG_walk\cache_data\all_participant.csv')

if __name__ == "__main__":
    main()
