import numpy as np
from Real_Time_IDS.features.ct_features import add_flow, compute_ct_features


def avg_interarrival(times):
    if len(times) < 2:
        return 0
    diffs = np.diff(times)
    return float(np.mean(diffs))


def jitter(times):
    if len(times) < 2:
        return 0
    diffs = np.diff(times)
    return float(np.std(diffs))


def extract_features(flow):

    duration = flow["last_seen"] - flow["start_time"]

    spkts = flow["fwd_packets"]
    dpkts = flow["bwd_packets"]

    sbytes = flow["fwd_bytes"]
    dbytes = flow["bwd_bytes"]

    total_packets = spkts + dpkts

    rate = total_packets / duration if duration > 0 else 0

    mean_pkt = np.mean(flow["packet_lengths"]) if flow["packet_lengths"] else 0

    # Phase 1
    sinpkt = avg_interarrival(flow["fwd_timestamps"])
    dinpkt = avg_interarrival(flow["bwd_timestamps"])

    sjit = jitter(flow["fwd_timestamps"])
    djit = jitter(flow["bwd_timestamps"])

    sload = sbytes / duration if duration > 0 else 0
    dload = dbytes / duration if duration > 0 else 0

    sttl = float(np.mean(flow["sttl"])) if flow["sttl"] else 0
    dttl = float(np.mean(flow["dttl"])) if flow["dttl"] else 0

    # Phase 2 (TCP)
    synack = 0
    ackdat = 0
    tcprtt = 0

    if flow.get("syn_time") and flow.get("synack_time"):
        synack = flow["synack_time"] - flow["syn_time"]

    if flow.get("synack_time") and flow.get("ack_time"):
        ackdat = flow["ack_time"] - flow["synack_time"]

    if flow.get("syn_time") and flow.get("ack_time"):
        tcprtt = flow["ack_time"] - flow["syn_time"]

    # Phase 3 (CT features)
    add_flow(flow)
    ct_features = compute_ct_features(flow)

    # Protocol encoding
    proto_map = {"TCP": 1, "UDP": 2}
    proto = proto_map.get(flow.get("proto"), 0)

    # TCP stats
    swin = np.mean(flow["swin"]) if flow["swin"] else 0
    dwin = np.mean(flow["dwin"]) if flow["dwin"] else 0

    stcpb = np.mean(flow["stcpb"]) if flow["stcpb"] else 0
    dtcpb = np.mean(flow["dtcpb"]) if flow["dtcpb"] else 0

    sloss = flow.get("sloss", 0)
    dloss = flow.get("dloss", 0)

    # State (simplified)
    state = 1 if flow["syn_time"] and flow["ack_time"] else 0

    return {
        "dur": duration,
        "spkts": spkts,
        "dpkts": dpkts,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "rate": rate,
        "smean": float(mean_pkt),
        "dmean": float(mean_pkt),

        # Phase 1
        "sinpkt": sinpkt,
        "dinpkt": dinpkt,
        "sjit": sjit,
        "djit": djit,
        "sload": sload,
        "dload": dload,
        "sttl": sttl,
        "dttl": dttl,

        # Phase 2
        "synack": synack,
        "ackdat": ackdat,
        "tcprtt": tcprtt,

        # Phase 3
        "ct_dst_ltm": ct_features["ct_dst_ltm"],
        "ct_src_dport_ltm": ct_features["ct_src_dport_ltm"],
        "ct_dst_sport_ltm": ct_features["ct_dst_sport_ltm"],
        "ct_src_ltm": ct_features["ct_src_ltm"],

        # Application layer
        "ct_ftp_cmd": flow.get("ftp_cmd_count", 0),
        "ct_flw_http_mthd": flow.get("http_method_count", 0),

        # Remaining (still dummy)
        "proto": proto,
        "service": 0,
        "state": state,
        "sloss": sloss,
        "dloss": dloss,
        "swin": swin,
        "stcpb": stcpb,
        "dtcpb": dtcpb,
        "dwin": dwin,
        "trans_depth": 0,
        "response_body_len": 0,
        "is_ftp_login": 0,
        "is_sm_ips_ports": 0,
    }