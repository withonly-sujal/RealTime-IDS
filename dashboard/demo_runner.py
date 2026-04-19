"""
Demo Runner — generates simulated IDS events for testing the dashboard
without requiring Wireshark/tshark or a live network.

Usage: python dashboard/run.py --demo
"""

import time
import random
import threading


FAKE_IPS = [
    "192.168.1.100", "192.168.1.101", "192.168.1.102", "192.168.1.103",
    "10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.50",
    "172.16.0.10", "172.16.0.20",
    "203.0.113.5", "198.51.100.12",
]

FAKE_PORTS = ["80", "443", "8080", "22", "3306", "5432", "21", "25", "53", "8443", "3389", "445"]

PROTOCOLS = ["TCP", "UDP"]


class DemoRunner:
    """
    Generates fake IDS prediction events to test the dashboard UI.
    Mimics the same event format as IDSRunner.
    """
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.running = False
        self.thread = None

        # Counters
        self.total_packets = 0
        self.attacks_detected = 0
        self.normal_flows = 0
        self.start_time = None

    def start(self):
        """Start generating demo events in a background thread."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[DEMO] Demo mode started — generating simulated events")

    def stop(self):
        """Stop the demo runner."""
        self.running = False
        print("[DEMO] Stopping...")

    def _run(self):
        """Generate fake events continuously."""
        self.event_bus.emit({
            "type": "status",
            "data": {"status": "running", "message": "IDS Started (Demo Mode)"}
        })

        # Simulate an initial burst of normal traffic
        time.sleep(1)

        while self.running:
            # Simulate packet processing (random batch)
            self.total_packets += random.randint(10, 80)

            # Generate 1-3 flow predictions per cycle
            num_predictions = random.randint(1, 3)

            for _ in range(num_predictions):
                self._generate_prediction()

            self._emit_stats()

            # Wait between cycles (simulates flow timeout expiry)
            time.sleep(random.uniform(1.0, 3.0))

        self.event_bus.emit({
            "type": "status",
            "data": {"status": "stopped", "message": "IDS Stopped (Demo Mode)"}
        })

    def _generate_prediction(self):
        """Generate a single fake flow prediction."""
        src_ip = random.choice(FAKE_IPS)
        dst_ip = random.choice([ip for ip in FAKE_IPS if ip != src_ip])
        src_port = random.choice(FAKE_PORTS)
        dst_port = random.choice(FAKE_PORTS)
        proto = random.choices(PROTOCOLS, weights=[0.75, 0.25])[0]

        # ~18% attack rate, with realistic probability distributions
        is_attack = random.random() < 0.18

        if is_attack:
            probability = round(random.uniform(0.70, 0.99), 4)
            self.attacks_detected += 1
        else:
            # Most normal traffic has very low probability
            if random.random() < 0.7:
                probability = round(random.uniform(0.01, 0.20), 4)
            else:
                probability = round(random.uniform(0.20, 0.55), 4)
            self.normal_flows += 1

        # Generate realistic-looking features
        duration = round(random.uniform(0.05, 45.0), 4)
        fwd_pkts = random.randint(1, 300)
        bwd_pkts = random.randint(0, 200)

        features = {
            "dur": duration,
            "spkts": fwd_pkts,
            "dpkts": bwd_pkts,
            "sbytes": fwd_pkts * random.randint(40, 1500),
            "dbytes": bwd_pkts * random.randint(40, 1500),
            "rate": round((fwd_pkts + bwd_pkts) / max(duration, 0.001), 2),
            "sttl": random.choice([64, 128, 255]),
            "dttl": random.choice([64, 128, 255]),
            "sload": round(random.uniform(0, 2000000), 2),
            "dload": round(random.uniform(0, 2000000), 2),
            "sinpkt": round(random.uniform(0, 500), 4),
            "dinpkt": round(random.uniform(0, 500), 4),
            "sjit": round(random.uniform(0, 100), 4),
            "djit": round(random.uniform(0, 100), 4),
            "smean": round(random.uniform(40, 1500), 2),
            "dmean": round(random.uniform(40, 1500), 2),
            "synack": round(random.uniform(0, 0.5), 4),
            "ackdat": round(random.uniform(0, 0.5), 4),
            "tcprtt": round(random.uniform(0, 1.0), 4),
            "ct_dst_ltm": random.randint(0, 20),
            "ct_src_dport_ltm": random.randint(0, 15),
            "ct_dst_sport_ltm": random.randint(0, 15),
            "ct_src_ltm": random.randint(0, 20),
            "proto": 1 if proto == "TCP" else 2,
            "state": random.choice([0, 1]),
            "sloss": random.randint(0, 5),
            "dloss": random.randint(0, 5),
        }

        self.event_bus.emit({
            "type": "prediction",
            "data": {
                "flow_key": f"('{src_ip}', '{dst_ip}', '{src_port}', '{dst_port}', '{proto}')",
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": proto,
                "prediction": 1 if is_attack else 0,
                "probability": probability,
                "features": features,
                "timestamp": time.time()
            }
        })

    def _emit_stats(self):
        total = self.attacks_detected + self.normal_flows
        adr = (self.attacks_detected / total * 100) if total > 0 else 0

        self.event_bus.emit({
            "type": "stats",
            "data": {
                "total_packets": self.total_packets,
                "attacks_detected": self.attacks_detected,
                "normal_flows": self.normal_flows,
                "total_flows": total,
                "active_flows": random.randint(3, 30),
                "adr": round(adr, 1),
                "uptime": round(time.time() - self.start_time, 0) if self.start_time else 0
            }
        })
