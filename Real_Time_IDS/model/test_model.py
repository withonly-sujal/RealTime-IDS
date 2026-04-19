import time

from Real_Time_IDS.capture.packet_capture import start_capture
from Real_Time_IDS.flow.flow_manager import update_flow
from Real_Time_IDS.flow.flow_table import flows
from Real_Time_IDS.features.feature_extractor import extract_features
from Real_Time_IDS.features.feature_mapper import prepare_features
from Real_Time_IDS.model.model_runner import ModelRunner

# Load features
from Real_Time_IDS.utils.config import SELECTED_FEATURES_PATH
from Real_Time_IDS.utils.config import PROCESSED_FEATURES_PATH

with open(PROCESSED_FEATURES_PATH) as f:
    feature_order = [line.strip() for line in f]

# Load model
from Real_Time_IDS.utils.config import STACK_MODEL_PATH

model = ModelRunner(STACK_MODEL_PATH)

capture = start_capture(interface="Wi-Fi")

print("IDS with Model Running (10 sec)...\n")

start_time = time.time()

for packet in capture.sniff_continuously():
    try:
        update_flow(packet)

        if time.time() - start_time > 10:
            break

    except:
        pass


print("\nPREDICTIONS:\n")

for key, flow in flows.items():
    features = extract_features(flow)

    df = prepare_features(features, feature_order)

    pred, prob = model.predict(df)

    if pred[0] == 1:
        print(f"ATTACK DETECTED: {key} | Prob: {prob[0]:.4f}")
    else:
        print(f"NORMAL: {key}")