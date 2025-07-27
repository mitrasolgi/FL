
import json
from collections import Counter

json_path = "data/behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_json/raw_kmt_user_0001.json"

with open(json_path, "r") as f:
    raw_data = json.load(f)

key_events = []
for session in raw_data.get("true_data", {}).values():
    key_events.extend(session.get("key_events", []))

print(f"Total key_events: {len(key_events)}")

event_types = [e.get("Event", "").lower() for e in key_events]
print("Event types count:", Counter(event_types))

keys = [e.get("Key") for e in key_events]
print(f"Keys present: {len([k for k in keys if k is not None])} / {len(keys)} total events")

# Print first 5 key_events for manual inspection
print("\nFirst 5 key events:")
for ev in key_events[:5]:
    print(ev)
