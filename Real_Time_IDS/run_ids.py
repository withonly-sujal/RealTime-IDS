import time

from Real_Time_IDS.capture.packet_capture import start_capture
from Real_Time_IDS.flow.flow_manager import update_flow, get_expired_flows
from Real_Time_IDS.features.feature_extractor import extract_features
from Real_Time_IDS.features.feature_mapper import prepare_features
from Real_Time_IDS.model.model_runner import ModelRunner
from Real_Time_IDS.utils.config import STACK_MODEL_PATH


# Load model
model = ModelRunner(STACK_MODEL_PATH)

print("Real-Time IDS Started...\n")

capture = start_capture(interface="Wi-Fi")

FLOW_TIMEOUT = 10

try:
    for packet in capture.sniff_continuously():

        try:
            # Step 1: Update flow
            update_flow(packet)

            # Step 2: Get expired flows
            expired_flows = get_expired_flows(timeout=FLOW_TIMEOUT)

            # Step 3: Process each expired flow
            for key, flow in expired_flows:

                features = extract_features(flow)

                # Convert to DataFrame (model handles missing features)
                import pandas as pd
                df = pd.DataFrame([features])

                pred, prob = model.predict(df)

                if pred[0] == 1:
                    print(f"🚨 ATTACK DETECTED: {key} | Prob: {prob[0]:.4f}")
                else:
                    print(f"✅ NORMAL: {key}")

        except Exception as e:
            print("Packet Error:", e)

except KeyboardInterrupt:
    print("\n IDS Stopped by user")