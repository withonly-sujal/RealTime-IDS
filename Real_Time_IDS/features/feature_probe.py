import pyshark
import time

INTERFACE = "Wi-Fi"   # change if needed
NUM_PACKETS = 20


def safe_get(attr, default=None):
    try:
        return attr
    except:
        return default


def inspect_packet(packet):
    print("\n" + "=" * 60)

    # --- BASIC INFO ---
    try:
        proto = packet.transport_layer
    except:
        proto = "UNKNOWN"

    print(f"Protocol: {proto}")

    # --- LAYERS ---
    print("Layers:", packet.layers)

    # --- IP INFO ---
    if hasattr(packet, "ip"):
        print("IP SRC:", packet.ip.src)
        print("IP DST:", packet.ip.dst)
        print("TTL:", safe_get(packet.ip.ttl))

    # --- PORT INFO ---
    if proto in ["TCP", "UDP"]:
        try:
            print("SRC PORT:", packet[proto].srcport)
            print("DST PORT:", packet[proto].dstport)
        except:
            print("Port info not available")

    # --- TCP DETAILS ---
    if proto == "TCP":
        try:
            print("SEQ:", safe_get(packet.tcp.seq))
            print("WINDOW:", safe_get(packet.tcp.window_size))
            print("FLAGS:", safe_get(packet.tcp.flags))
        except:
            print("TCP details missing")

    # --- HTTP ---
    if hasattr(packet, "http"):
        print("HTTP detected!")

    # --- LENGTH ---
    print("Packet Length:", safe_get(packet.length))

    print("=" * 60)


def feature_availability(packet):
    print("\nFEATURE AVAILABILITY CHECK")

    available = {}
    derived = {}
    missing = {}

    # --- DIRECT FEATURES ---
    available["proto"] = hasattr(packet, "transport_layer")
    available["ttl"] = hasattr(packet, "ip") and hasattr(packet.ip, "ttl")
    available["packet_length"] = hasattr(packet, "length")

    # --- PORT ---
    try:
        proto = packet.transport_layer
        available["ports"] = proto in ["TCP", "UDP"]
    except:
        available["ports"] = False

    # --- TCP ---
    available["tcp_seq"] = hasattr(packet, "tcp") and hasattr(packet.tcp, "seq")
    available["tcp_window"] = hasattr(packet, "tcp") and hasattr(packet.tcp, "window_size")

    # --- HTTP ---
    available["http"] = hasattr(packet, "http")

    # --- DERIVED FEATURES ---
    derived["dur"] = "needs flow tracking"
    derived["rate"] = "needs duration + packets"
    derived["sinpkt"] = "needs timestamps"
    derived["sjit"] = "needs multiple packets"
    derived["ct_features"] = "needs flow history"

    # --- MISSING FEATURES ---
    missing["tcprtt"] = "needs full handshake"
    missing["synack"] = "needs precise timing"
    missing["ackdat"] = "needs handshake tracking"
    missing["sloss"] = "not reliable (approx only)"
    missing["dloss"] = "not reliable (approx only)"

    print("\nDIRECTLY AVAILABLE:")
    for k, v in available.items():
        if v:
            print(f"{k}")

    print("\nDERIVABLE (you must compute):")
    for k, v in derived.items():
        print(f"{k} → {v}")

    print("\n NOT RELIABLE:")
    for k, v in missing.items():
        print(f"{k} → {v}")


def main():
    print("Starting Feature Probe...\n")

    capture = pyshark.LiveCapture(
        interface=INTERFACE,
        tshark_path=r"D:\wireshark\tshark.exe"
    )

    count = 0

    for packet in capture.sniff_continuously():
        try:
            inspect_packet(packet)
            feature_availability(packet)

            count += 1
            if count >= NUM_PACKETS:
                break

        except Exception as e:
            print("Error:", e)

    print("\nProbe Complete")


if __name__ == "__main__":
    main()