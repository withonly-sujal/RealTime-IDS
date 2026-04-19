from Real_Time_IDS.flow.flow_manager import update_flow, print_flows
from Real_Time_IDS.capture.packet_capture import start_capture

capture = start_capture(interface="Wi-Fi")

print("Testing Flow Builder...\n")

for packet in capture.sniff_continuously(packet_count=20):
    try:
        update_flow(packet)
    except:
        pass
    
print_flows()