import time
from collections import defaultdict, deque

TIME_WINDOW = 60  # seconds

# Store recent flows
flow_history = deque()

def add_flow(flow):
    now = time.time()

    flow_entry = {
        "time": now,
        "src_ip": flow["src_ip"],
        "dst_ip": flow["dst_ip"],
        "src_port": None,
        "dst_port": None,
    }

    flow_history.append(flow_entry)

    # Remove old flows
    while flow_history and (now - flow_history[0]["time"] > TIME_WINDOW):
        flow_history.popleft()


def compute_ct_features(flow):
    now = time.time()

    src_ip = flow["src_ip"]
    dst_ip = flow["dst_ip"]

    ct_src_ltm = 0
    ct_dst_ltm = 0
    ct_src_dport_ltm = 0
    ct_dst_sport_ltm = 0

    for f in flow_history:
        if now - f["time"] > TIME_WINDOW:
            continue

        if f["src_ip"] == src_ip:
            ct_src_ltm += 1

        if f["dst_ip"] == dst_ip:
            ct_dst_ltm += 1

        if f["src_ip"] == src_ip and f["dst_ip"] == dst_ip:
            ct_src_dport_ltm += 1

        if f["dst_ip"] == dst_ip and f["src_ip"] == src_ip:
            ct_dst_sport_ltm += 1

    return {
        "ct_src_ltm": ct_src_ltm,
        "ct_dst_ltm": ct_dst_ltm,
        "ct_src_dport_ltm": ct_src_dport_ltm,
        "ct_dst_sport_ltm": ct_dst_sport_ltm,
    }