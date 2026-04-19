from .flow_table import flows, create_flow
import time


def get_flow_key(packet):
    try:
        src_ip = packet.ip.src
        dst_ip = packet.ip.dst
        proto = packet.transport_layer

        if proto not in ["TCP", "UDP"]:
            return None

        src_port = packet[proto].srcport
        dst_port = packet[proto].dstport

        if (src_ip, src_port) < (dst_ip, dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto)
        else:
            return (dst_ip, src_ip, dst_port, src_port, proto)

    except:
        return None


def update_flow(packet):
    key = get_flow_key(packet)
    if key is None:
        return

    if key not in flows:
        flows[key] = create_flow(packet)

    flow = flows[key]

    src_ip = packet.ip.src
    pkt_len = int(packet.length)
    now = time.time()

    flow["last_seen"] = now

    # Direction logic
    if src_ip == flow["src_ip"]:
        flow["fwd_packets"] += 1
        flow["fwd_bytes"] += pkt_len
        flow["fwd_timestamps"].append(now)

        if hasattr(packet.ip, "ttl"):
            flow["sttl"].append(int(packet.ip.ttl))

    else:
        flow["bwd_packets"] += 1
        flow["bwd_bytes"] += pkt_len
        flow["bwd_timestamps"].append(now)

        if hasattr(packet.ip, "ttl"):
            flow["dttl"].append(int(packet.ip.ttl))

    flow["packet_lengths"].append(pkt_len)
    flow["timestamps"].append(now)
    
    flow["proto"] = packet.transport_layer
    if packet.transport_layer == "TCP":
        try:
            seq = int(packet.tcp.seq)
            win = int(packet.tcp.window_size)
    
            if src_ip == flow["src_ip"]:
                flow["swin"].append(win)
                flow["stcpb"].append(seq)
    
                if flow["prev_seq_fwd"] is not None and seq < flow["prev_seq_fwd"]:
                    flow["sloss"] += 1
    
                flow["prev_seq_fwd"] = seq
    
            else:
                flow["dwin"].append(win)
                flow["dtcpb"].append(seq)
    
                if flow["prev_seq_bwd"] is not None and seq < flow["prev_seq_bwd"]:
                    flow["dloss"] += 1
    
                flow["prev_seq_bwd"] = seq
    
        except:
            pass
        

    if packet.transport_layer == "TCP":
        try:
            flags = packet.tcp.flags

            if flags == "0x0002":  # SYN
                flow["syn_time"] = now

            elif flags == "0x0012":  # SYN-ACK
                flow["synack_time"] = now

            elif flags == "0x0010":  # ACK
                if flow["synack_time"] is not None:
                    flow["ack_time"] = now

        except:
            pass

    try:
        if packet.transport_layer == "TCP":
            src_port = packet.tcp.srcport
            dst_port = packet.tcp.dstport

            # FTP
            if src_port == "21" or dst_port == "21":
                flow["ftp_cmd_count"] += 1

        # HTTP (only visible if not encrypted)
        if hasattr(packet, "http"):
            flow["http_method_count"] += 1

    except:
        pass


def get_expired_flows(timeout=30):
    now = time.time()
    expired = []

    for key, flow in list(flows.items()):
        if now - flow["last_seen"] > timeout:
            expired.append((key, flows.pop(key)))

    return expired


def print_flows():
    print("\n Current Flows:")
    for key, flow in flows.items():
        print(key, "| Packets:", flow["fwd_packets"] + flow["bwd_packets"])