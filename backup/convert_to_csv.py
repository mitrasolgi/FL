import os
import pandas as pd

folder_in = "data/behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_xlsx"
folder_out = "data/behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_csv"
os.makedirs(folder_out, exist_ok=True)

for file in os.listdir(folder_in):
    if file.endswith(".xlsx"):
        full_path = os.path.join(folder_in, file)
        df = pd.read_excel(full_path, engine='openpyxl')
        csv_filename = os.path.splitext(file)[0] + ".csv"
        csv_path = os.path.join(folder_out, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Converted {file} to {csv_filename}")
