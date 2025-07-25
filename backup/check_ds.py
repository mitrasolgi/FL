import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you are loading data into a Pandas DataFrame
df = pd.read_csv("data/edge_iiot_data.csv", low_memory=False)  # Adjust with actual path to your dataset

attack_column = 'Attack_label'  # Use 'Attack_label' or 'Attack_type' depending on your dataset

if attack_column in df.columns:
    print(f"Column '{attack_column}' exists in the dataset.")
    
    # Check for missing values in the 'Attack_label' column
    missing_values = df[attack_column].isnull().sum()
    print(f"Missing values in '{attack_column}': {missing_values}")
    
    # Check distribution of the 'Attack_label' column
    print("\nDistribution of 'Attack_label':")
    distribution = df[attack_column].value_counts()
    print(distribution)

else:
    print(f"Column '{attack_column}' does not exist in the dataset.")