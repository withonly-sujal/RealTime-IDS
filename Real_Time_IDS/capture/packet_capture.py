import pyshark

def start_capture(interface="Wi-Fi"):
    capture = pyshark.LiveCapture(
        interface=interface,
        tshark_path=r"D:\wireshark\tshark.exe",
        use_json=True,
        bpf_filter='ip'
    )
    return capture