from Real_Time_IDS.capture.packet_capture import start_capture

capture = start_capture(interface="Wi-Fi")

print("Starting packet capture...\n")

for packet in capture.sniff_continuously(packet_count=10):
    try:
        print(f"{packet.ip.src} → {packet.ip.dst}")
    except:
        pass