import metrics
import pandas as pd

def main():
    participant_no = 30
    motion_file = r'OneDrive_2023-08-30\Partcipants wise conslidated\Participant ' + str(participant_no) + r'.csv'
    survey_file = r'Mental Health.csv'
    participant_motion = metrics.metrics(motion_file)
    participant_survey = pd.read_csv(survey_file, encoding='unicode-escape')
    action_result = participant_motion.getActionCount()
    calorie_result = participant_motion.getCalorieCount()
    # print(result)
    # Initialize empty lists to store data
    participant_ids = []
    weeks = []
    games = []
    calories_burnt = []
    action_count = []
    emotional = []
    psychological = []
    social = []

    # Loop through the data dictionary and extract the information
    for game, game_data in action_result.items():
        for week, action_count_value in game_data.items():
            participant_ids.append(participant_no)
            weeks.append(int([*week][-1]))
            games.append(game)
            calories_burnt.append(calorie_result[game][week])
            action_count.append(action_count_value)

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
