import time

flows = {}

def create_flow(packet):
    return {
        "start_time": time.time(),
        "last_seen": time.time(),

        "fwd_packets": 0,
        "bwd_packets": 0,

        "fwd_bytes": 0,
        "bwd_bytes": 0,

        "packet_lengths": [],
        "timestamps": [],

        "src_ip": packet.ip.src,
        "dst_ip": packet.ip.dst,

        # Phase 1
        "fwd_timestamps": [],
        "bwd_timestamps": [],

        "sttl": [],
        "dttl": [],

        # Phase 2 (TCP)
        "syn_time": None,
        "synack_time": None,
        "ack_time": None,

        # Phase 3 (Application layer)
        "ftp_cmd_count": 0,
        "http_method_count": 0,
        
        # Protocol / TCP stats
        "proto": None,
        "swin": [],
        "dwin": [],
        "stcpb": [],
        "dtcpb": [],
        
        # Loss tracking
        "prev_seq_fwd": None,
        "prev_seq_bwd": None,
        "sloss": 0,
        "dloss": 0,
    }