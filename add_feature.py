import metrics
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.DataFrame()
consolidated_df = pd.read_csv(r"C:\Users\User\Desktop\SG_walk\Motion and wellbeing data.csv", encoding='UTF-8')

for participant in tqdm(range(1, 31)):
    max_accs = []
    participants = []
    weeks = []
    games = []
    motion_file = r'OneDrive_2023-08-30\Partcipants wise conslidated\Participant ' + str(participant) + r'.csv'
    motion = metrics.metrics(motion_file)
    max_acc_dict = motion.getMaxAcceleration()

    for game, game_data in max_acc_dict.items():
        for week, value in game_data.items():
            participants.append(participant)
            weeks.append(week)
            games.append(game)
            max_accs.append(value)

    # Create a DataFrame from the extracted data
    this_df = pd.DataFrame({
        "Participant ID": participants,
        "Week": weeks,
        "Game": games,
        "Max Acceleration": max_accs
    })

    df = pd.concat([df, this_df], axis=0)

survey_df = consolidated_df.copy()
merged_df = df.merge(survey_df, on=['Participant ID', 'Week', 'Game'], how='left')
merged_df.to_csv('Motion_wellbeing_consolidated.csv')




