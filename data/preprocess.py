import pandas as pd
from sklearn.utils import shuffle

# Load the full dataset
df = pd.read_csv('fl.csv', low_memory=False)

# Display the dataset before preprocessing
print(f"Dataset before preprocessing: {df.shape[0]} rows and {df.shape[1]} columns.")
print(df['Attack_type'].value_counts())

# Define the relevant attack types (based on the distribution you provided)
relevant_attacks = [
    'Normal', 'DDoS_UDP', 'DDoS_ICMP', 'SQL_injection', 'Password',
    'Vulnerability_scanner', 'DDoS_TCP', 'DDoS_HTTP', 'Uploading',
    'Backdoor', 'Port_Scanning', 'XSS', 'Ransomware', 'MITM', 'Fingerprinting'
]

# Keep only rows with relevant attack types
df = df[df['Attack_type'].isin(relevant_attacks)]

# Optionally, filter on device categories (if available)
# If a 'Device_Type' or similar column exists:
# relevant_devices = ['Camera', 'Sensor', 'Gateway']
# df = df[df['Device_Type'].isin(relevant_devices)]

# Drop unnecessary columns (replace `...` with actual column names you want to drop)
drop_columns = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
    "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"
]
df.drop(columns=drop_columns, errors='ignore', inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Encode the target label (Attack_type) into numerical values
attack_mapping = {
    'Normal': 0, 'DDoS_UDP': 1, 'DDoS_ICMP': 2, 'SQL_injection': 3, 'Password': 4,
    'Vulnerability_scanner': 5, 'DDoS_TCP': 6, 'DDoS_HTTP': 7, 'Uploading': 8,
    'Backdoor': 9, 'Port_Scanning': 10, 'XSS': 11, 'Ransomware': 12, 'MITM': 13, 'Fingerprinting': 14
}

df['Attack_type'] = df['Attack_type'].map(attack_mapping)

# Shuffle the data to mix the samples
df = shuffle(df)

# Save the preprocessed data to a new CSV file
df.to_csv('edge_iiot_data.csv', index=False)

# Display a quick summary of the dataset after preprocessing
print(f"Dataset after preprocessing: {df.shape[0]} rows and {df.shape[1]} columns.")
print(df['Attack_type'].value_counts())
