import time

from Real_Time_IDS.capture.packet_capture import start_capture
from Real_Time_IDS.flow.flow_manager import update_flow
from Real_Time_IDS.flow.flow_table import flows
from Real_Time_IDS.features.feature_extractor import extract_features
#from Real_Time_IDS.features.feature_extractor import extract_features


capture = start_capture(interface="Wi-Fi")

print("Real-Time Feature Extraction (10 sec)...\n")

start_time = time.time()
DURATION = 10  # seconds

for packet in capture.sniff_continuously():

    try:
        update_flow(packet)

        # Stop after 10 seconds
        if time.time() - start_time > DURATION:
            break

    except:
        pass


# Print all flows after stopping
print("\n FINAL FLOWS:\n")

for key, flow in flows.items():
    features = extract_features(flow)

    print("FLOW:", key)
    print("FEATURES:", features)
    print("-" * 50)