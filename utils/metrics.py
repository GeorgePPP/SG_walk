import json
import numpy as np
import pandas as pd  
import data_cleaning

class metrics:
    def __init__(self, fp) -> None:
        self.df = data_cleaning.extractData(pd.read_csv(fp, encoding='unicode_escape'))
        original_list = list(self.df['Week'].unique())
        # Check if any element is of type np.int64
        if any(type(ele) == np.int64 for ele in original_list):
            self.weeks = [ele.item() for ele in original_list if type(ele) == np.int64]
        else:
            self.weeks = original_list
        if 'Game_Type' in self.df.columns:
            self.games = list(self.df['Game_Type'].unique())

    def getJson(self):
        output = {}
        output['Action Count'] = self.getActionCount()
        output['Calories Burnt'] = self.getCalorieCount()
        output['Max Acceleration'] = self.getMaxAcceleration()
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
    
    def getMaxAcceleration(self):
        # Create a nested dictionary to store the peak counts
        max_acc_dict = {}

        if 'Game_Type' in self.df.columns:
            # Process each game
            for game in self.games:
                max_acc_dict[game] = {}

                for week in self.weeks:
                    max_acc_dict[game][week] = {}
                    # Filter the data based on game, label, week 
                    input_signal_acceleration = self.df[(self.df['Game_Type'] == game) &
                                        (self.df['Week'] == week)]['total acceleration']
                    
                    if len(input_signal_acceleration) == 0:
                        max_acc_dict[game][week] = 0
                    else:
                        max_acc_dict[game][week] = round(np.max(input_signal_acceleration), 2)
        else:
            for week in self.weeks:
                    max_acc_dict[week] = {}
                    # Filter the data based on game, label, week 
                    input_signal_acceleration = self.df[
                                        (self.df['Week'] == week)]['total acceleration']

                    if len(input_signal_acceleration) == 0:
                        max_acc_dict[week] = 0
                    else:    
                        max_acc_dict[week] = round(np.max(input_signal_acceleration), 2)
                    
        return max_acc_dict

    def getActionCount(self):
        # Create a nested dictionary to store the peak counts
        action_count_dict = {}

        if 'Game_Type' in self.df.columns:
            # Process each game
            for game in self.games:
                action_count_dict[game] = {}

                for week in self.weeks:
                    action_count_dict[game][week] = {}
                    # Filter the data based on game, label, week 
                    input_signal_acceleration = self.df[(self.df['Game_Type'] == game) &
                                        (self.df['Week'] == week)]['total acceleration']
                
                    action_count_dict[game][week] = int(self.getPeakNumber(input_signal_acceleration))
        else:
            for week in self.weeks:
                action_count_dict[week] = {}
                # Filter the data based on game, label, week 
                input_signal_acceleration = self.df[
                                    (self.df['Week'] == week)]['total acceleration']
                
                action_count_dict[week] = int(self.getPeakNumber(input_signal_acceleration))

        return action_count_dict

    def getCalorieCount(self):
        calorie_count_dict = {}

        if 'Game_Type' in self.df.columns:
            # Process each game
            for game in self.games:
                calorie_count_dict[game] = {}
                for week in self.weeks:
                    calorie_count_dict[game][week] = {}
                    input_signal_acceleration = self.df[(self.df['Game_Type'] == game) & 
                                                        (self.df['Week'] == week)]['total acceleration']
                    if len(input_signal_acceleration) == 0:
                        calorie_count_dict[game][week]= 0
                    else:
                        weight = 65 # Mean weight of participants
                        frequency = 50
                        duration = (len(input_signal_acceleration)/frequency)/3600 # Activity duration of this label in each day in hour 
                        RMS = np.sqrt(np.mean((input_signal_acceleration*9.8)**2)) # RMS in m/s2 
                        MET = 1.8*RMS - 15
                        kcal_per_hour = 1.05*MET*weight*duration
                        total_kcal = kcal_per_hour * duration
                        calorie_count_dict[game][week]= round(total_kcal, 2)

        else:
            for week in self.weeks:
                calorie_count_dict[week] = {}
                input_signal_acceleration = self.df[
                                                    (self.df['Week'] == week)]['total acceleration']
                if len(input_signal_acceleration) == 0:
                    calorie_count_dict[week]= 0
                else:
                    weight = 65 # Mean weight of participants
                    frequency = 50
                    duration = (len(input_signal_acceleration)/frequency)/3600 # Activity duration of this label in each day in hour 
                    RMS = np.sqrt(np.mean((input_signal_acceleration*9.8)**2)) # RMS in m/s2 
                    MET = 1.8*RMS - 15
                    kcal_per_hour = 1.05*MET*weight*duration
                    total_kcal = kcal_per_hour * duration
                    calorie_count_dict[week]= round(total_kcal, 2)

        return calorie_count_dict
