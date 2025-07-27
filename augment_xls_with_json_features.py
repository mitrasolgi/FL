import os
import json
import pandas as pd
import numpy as np
from dateutil import parser

# ========== CONFIGURATION ==========
json_dir = "data/behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_json"
xls_dir = "data/behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_xlsx"
output_dir = "data/behaviour_biometrics_dataset/feature_kmt_dataset/custom_feature_kmt_xlsx"
# ===================================

def parse_time(t):
    try:
        return parser.parse(t).timestamp()
    except Exception:
        return None

def extract_features_from_events(events):
    """Extract features from a list of key events"""
    hold_times = []
    flight_times = []
    key_down_times = {}

    for ev in events:
        key = ev.get("Key")
        event_type = ev.get("Event", "").lower()
        timestamp = ev.get("Epoch", None)
        if not timestamp:
            timestamp = parse_time(ev.get("Timestamp"))
        else:
            try:
                timestamp = float(timestamp)
            except:
                timestamp = parse_time(ev.get("Timestamp"))

        if timestamp is None:
            continue

        if event_type in ["keydown", "down", "pressed"]:
            key_down_times[key] = timestamp
        elif event_type in ["keyup", "up", "released"]:
            if key in key_down_times:
                hold_time = timestamp - key_down_times[key]
                if 0 < hold_time < 5:
                    hold_times.append(hold_time)
                key_down_times.pop(key, None)

    # Flight times (between consecutive key presses)
    down_timestamps = []
    for ev in events:
        if ev.get("Event", "").lower() in ["keydown", "down", "pressed"]:
            timestamp = ev.get("Epoch", None)
            if not timestamp:
                timestamp = parse_time(ev.get("Timestamp"))
            else:
                try:
                    timestamp = float(timestamp)
                except:
                    timestamp = parse_time(ev.get("Timestamp"))
            if timestamp is not None:
                down_timestamps.append(timestamp)
    
    down_timestamps.sort()
    for i in range(1, len(down_timestamps)):
        delta = down_timestamps[i] - down_timestamps[i - 1]
        if 0 < delta < 5:
            flight_times.append(delta)

    return {
        "hold_mean": np.mean(hold_times) if hold_times else 0,
        "hold_std": np.std(hold_times) if hold_times else 0,
        "flight_mean": np.mean(flight_times) if flight_times else 0,
        "flight_std": np.std(flight_times) if flight_times else 0
    }

def segment_events_by_time(events, num_segments):
    """Segment events into equal time windows"""
    if not events:
        return [[] for _ in range(num_segments)]
    
    # Get timestamps
    timestamps = []
    for ev in events:
        timestamp = ev.get("Epoch", None)
        if not timestamp:
            timestamp = parse_time(ev.get("Timestamp"))
        else:
            try:
                timestamp = float(timestamp)
            except:
                timestamp = parse_time(ev.get("Timestamp"))
        if timestamp is not None:
            timestamps.append(timestamp)
    
    if not timestamps:
        return [[] for _ in range(num_segments)]
    
    min_time = min(timestamps)
    max_time = max(timestamps)
    time_span = max_time - min_time
    segment_duration = time_span / num_segments
    
    segments = [[] for _ in range(num_segments)]
    
    for ev in events:
        timestamp = ev.get("Epoch", None)
        if not timestamp:
            timestamp = parse_time(ev.get("Timestamp"))
        else:
            try:
                timestamp = float(timestamp)
            except:
                timestamp = parse_time(ev.get("Timestamp"))
        
        if timestamp is not None:
            segment_idx = min(int((timestamp - min_time) / segment_duration), num_segments - 1)
            segments[segment_idx].append(ev)
    
    return segments

def segment_events_by_count(events, num_segments):
    """Segment events into roughly equal counts"""
    if not events:
        return [[] for _ in range(num_segments)]
    
    events_per_segment = len(events) // num_segments
    remainder = len(events) % num_segments
    
    segments = []
    start_idx = 0
    
    for i in range(num_segments):
        # Add one extra event to the first 'remainder' segments
        segment_size = events_per_segment + (1 if i < remainder else 0)
        end_idx = start_idx + segment_size
        segments.append(events[start_idx:end_idx])
        start_idx = end_idx
    
    return segments

def extract_segmented_features_from_json(file_path, num_segments):
    with open(file_path, "r") as f:
        raw_data = json.load(f)

    key_events = raw_data.get("true_data", {}).get("test_1", {}).get("key_events", [])
    
    # Try time-based segmentation first, fall back to count-based
    segments = segment_events_by_time(key_events, num_segments)
    
    # If time-based segmentation fails or gives empty segments, use count-based
    if not any(segments) or any(len(seg) == 0 for seg in segments):
        segments = segment_events_by_count(key_events, num_segments)
    
    features_list = []
    for i, segment_events in enumerate(segments):
        features = extract_features_from_events(segment_events)
        features_list.append(features)
        print(f"  Segment {i+1}: {len(segment_events)} events, hold_mean={features['hold_mean']:.6f}")
    
    return features_list

def load_xls_and_append_segmented_features(xls_path, features_list):
    df = pd.read_excel(xls_path)
    
    # Ensure we have the right number of feature sets
    if len(features_list) != len(df):
        print(f"  ⚠️ Mismatch: {len(features_list)} feature segments vs {len(df)} Excel rows")
        # Pad or truncate as needed
        while len(features_list) < len(df):
            features_list.append({"hold_mean": 0, "hold_std": 0, "flight_mean": 0, "flight_std": 0})
        features_list = features_list[:len(df)]
    
    # Add features row by row
    for feature_name in ["hold_mean", "hold_std", "flight_mean", "flight_std"]:
        df[feature_name] = [features[feature_name] for features in features_list]
    
    return df

def main():
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(json_dir):
        if filename.endswith(".json") and filename.startswith("raw_kmt_user_"):
            user_id = filename.replace("raw_kmt_user_", "").replace(".json", "").zfill(4)
            json_path = os.path.join(json_dir, filename)
            xls_filename = f"feature_kmt_user_{user_id}.xlsx"
            xls_path = os.path.join(xls_dir, xls_filename)

            if not os.path.exists(xls_path):
                print(f"⚠️ XLS file not found for user {user_id}")
                continue

            # Load Excel to get number of rows
            df_temp = pd.read_excel(xls_path)
            num_segments = len(df_temp)
            
            print(f"Processing user {user_id}: {num_segments} segments")
            features_list = extract_segmented_features_from_json(json_path, num_segments)
            
            df = load_xls_and_append_segmented_features(xls_path, features_list)
            output_path = os.path.join(output_dir, f"user_{user_id}_extended.xlsx")
            df.to_excel(output_path, index=False)
            print(f"✅ Processed and saved: {output_path}")

if __name__ == "__main__":
    main()