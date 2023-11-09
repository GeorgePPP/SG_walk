import os
import pandas as pd

# Specify the directory containing the XLSX files
dir = r'C:\Users\User\Desktop\SG_walk\Participant wise accelerometer data (not consolidated)\Exercise consolidated'

# Iterate through each file in the directory
for filename in os.listdir(dir):
    if filename.endswith('.xlsx'):
        # Construct the full file paths
        fp = os.path.join(dir, filename)
    
        print("Converting {}".format(fp))

        # Read the XLSX file using pandas
        df = pd.read_excel(fp)

        csv_filename = filename.replace(".xlsx", ".csv")
        csv_fp = os.path.join(dir, csv_filename)

        # Save the DataFrame as a CSV file
        df.to_csv(csv_fp, index=False, encoding='utf-8', sep=',')

        print("Converted {}".format(fp))

print("Conversion  complete.")
