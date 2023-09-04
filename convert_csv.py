import os
import pandas as pd

# Specify the directory containing the XLSX files
xlsx_directory = 'OneDrive_2023-08-30\Partcipants wise conslidated'

# Iterate through each file in the directory
for filename in os.listdir(xlsx_directory):
    if filename.endswith('.xlsx'):
        # Construct the full file paths
        xlsx_file_path = os.path.join(xlsx_directory, filename)
        csv_file_path = os.path.splitext(xlsx_file_path)[0] + '.csv'
        
        # Read the XLSX file using pandas
        df = pd.read_excel(xlsx_file_path, engine='openpyxl')
        
        # Save the DataFrame as a CSV file
        df.to_csv(csv_file_path, index=False, encoding='utf-8', sep=',')

print("Conversion complete.")
