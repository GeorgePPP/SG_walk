import metrics
import pandas as pd

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
    pariticpant_exer_ls = list(range(1, 56)) - participant_game_ls

    # Read motion, survey, demographic files
    motion_file = r'C:\Users\User\Desktop\SG_walk\Participant wise accelerometer data\Participant ' + str(participant_game_ls[0]) + r'.csv'
    survey_data = pd.read_csv(r'Mental Health data (Including condition 3&4).csv', encoding='unicode-escape')
    demographic_data = pd.read_csv(r"C:\Users\User\Desktop\SG_walk\Demographic.csv", encoding='unicode-escape')
    participant_motion = metrics.metrics(motion_file) # A dictionary

    # Get data for motion metrics
    action_result = participant_motion.getActionCount()
    calorie_result = participant_motion.getCalorieCount()
    max_acc_result = participant_motion.getMaxAcceleration()

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
    languishing = []
    flourishing = []
    moderate = []

    # Demographic data
    grouping = []
    gender = []
    age = []
    income = []
    edu = []
    selfses = []
    

    for participant in participant_game_ls:
        # Loop through the data dictionary and extract the information
        demographic_row = demographic_data[demographic_data['ID'] == participant].iloc[0]
        for game, game_data in action_result.items():
            for week, result in game_data.items():
                survey_row = survey_data[(survey_data['Participant ID'] == participant) & (survey_data['Week'] == week)].iloc[0]
                if isinstance(week, int):
                    weeks.append(week)
                else:
                    weeks.append(int([*week][-1]))
                games.append(game)
                calories_burnt.append(calorie_result[game][week])
                action_count.append(action_result[game][weeks])
                max_acc.append(max_acc_result[game][week])
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

                #TODO: Add logic for languishing, flourishing, moderate row



                



                


                
    

    # Create a DataFrame from the extracted data
    motion_df = pd.DataFrame({
        "Participant ID": participant_ids,
        "Week": weeks,
        "Game": games,
        "Calories Burnt": calories_burnt,
        "Action Count": action_count
    })

    # Extract and prepare data from the survey file
    survey_df = participant_survey.copy()
    merged_df = motion_df.merge(survey_df, on=['Participant ID', 'Week'], how='left')

    merged_df.to_csv('Participant_'+str(participant_no)+'_data.csv')
    
if __name__ == "__main__":
    main()
