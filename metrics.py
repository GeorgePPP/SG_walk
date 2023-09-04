import json
import numpy as np
import pandas as pd  
import data_cleaning

class metrics:
    def __init__(self, fp) -> None:
        self.games = ['Arctic Punch', 'Fruit Ninja', 'Piano Step']
        self.labels = ['left hand', 'right hand', 'waist', 'left leg', 'right leg']
        self.df = data_cleaning.extractData(pd.read_csv(fp, encoding='unicode_escape'))

    def getJson(self):
        output = {}
        output['Action Count'] = self.getActionCount()
        output['Calories Burnt'] = self.getCalorieCount()
        return json.dumps(output, indent=4)

    # This algorithm returns numbers of peak identified
    def getPeakNumber(self, arr):
        threshold = 4 # experimentally good threshold
        y = np.array(arr)
        std = np.std(y)
        mean = np.mean(y)
        deviations = np.abs(y - mean)
        peak_count = np.sum(deviations > (threshold * std))
        return peak_count

    def getActionCount(self):
        # Create a nested dictionary to store the peak counts
        action_count_dict = {}

        # Process each game
        for game in self.games:
            action_count_dict[game] = {}

            for label in self.labels:
                action_count_dict[game][label] = {}

                # Filter the data based on game & label 
                input_signal_acceleration = self.df[(self.df['Game_Type'] == game) &
                                    (self.df['Label'] == label)]['total acceleration']
                
                action_count_dict[game][label] = int(self.getPeakNumber(input_signal_acceleration))
        return action_count_dict

    def getCalorieCount(self):
        calorie_count_dict = {}

        # Process each game
        for game in self.games:
            calorie_count_dict[game] = {}
            input_signal_acceleration = self.df[(self.df['Game_Type'] == game)]['total acceleration']
            if len(input_signal_acceleration) == 0:
                calorie_count_dict[game]= 0
            else:
                weight = 65 # Mean weight of participants
                frequency = 50
                duration = (len(input_signal_acceleration)/frequency)/3600 # Activity duration of this label in each day in hour 
                RMS = np.sqrt(np.mean((input_signal_acceleration*9.8)**2)) # RMS in m/s2 
                MET = 1.8*RMS - 15
                kcal_per_hour = 1.05*MET*weight*duration
                total_kcal = kcal_per_hour * duration
                calorie_count_dict[game]= round(total_kcal, 2)
        return calorie_count_dict