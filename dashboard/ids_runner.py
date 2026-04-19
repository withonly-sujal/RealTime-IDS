import time
import threading
import pandas as pd

from Real_Time_IDS.capture.packet_capture import start_capture
from Real_Time_IDS.flow.flow_manager import update_flow, get_expired_flows
from Real_Time_IDS.flow.flow_table import flows
from Real_Time_IDS.features.feature_extractor import extract_features
from Real_Time_IDS.model.model_runner import ModelRunner
from Real_Time_IDS.utils.config import STACK_MODEL_PATH


class IDSRunner:
    def __init__(self, event_bus, interface="Wi-Fi", flow_timeout=10):
        self.event_bus = event_bus
        self.interface = interface
        self.flow_timeout = flow_timeout
        self.running = False
        self.thread = None

        self.total_packets = 0
        self.attacks_detected = 0
        self.normal_flows = 0
        self.start_time = None

    def start(self):
        """Start the IDS in a background thread."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[IDS] Started on interface: {self.interface}")

    def stop(self):
        """Stop the IDS."""
        self.running = False
        print("[IDS] Stopping...")

    def _run(self):
        """Main IDS loop (runs in background thread)."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            print("[IDS] Loading model...")
            model = ModelRunner(STACK_MODEL_PATH)
            print("[IDS] Model loaded successfully")

            print(f"[IDS] Starting packet capture on '{self.interface}'...")
            capture = start_capture(interface=self.interface)

            self.event_bus.emit({
                "type": "status",
                "data": {"status": "running", "message": "Real-Time IDS Started"}
            })

            last_stats_time = time.time()

            for packet in capture.sniff_continuously():
                if not self.running:
                    break

                try:
                    update_flow(packet)
                    self.total_packets += 1

                    now = time.time()
                    if now - last_stats_time >= 2:
                        self._emit_stats()
                        last_stats_time = now

                    expired_flows = get_expired_flows(timeout=self.flow_timeout)

                    for key, flow in expired_flows:
                        features = extract_features(flow)
                        df = pd.DataFrame([features])
                        pred, prob = model.predict(df)

                        is_attack = pred[0] == 1

                        if is_attack:
                            self.attacks_detected += 1
                        else:
                            self.normal_flows += 1

                        self.event_bus.emit({
                            "type": "prediction",
                            "data": {
                                "flow_key": str(key),
                                "src_ip": key[0],
                                "dst_ip": key[1],
                                "src_port": str(key[2]),
                                "dst_port": str(key[3]),
                                "protocol": key[4],
                                "prediction": int(pred[0]),
                                "probability": round(float(prob[0]), 4),
                                "features": {
                                    k: round(float(v), 4) if isinstance(v, (int, float)) else str(v)
                                    for k, v in features.items()
                                },
                                "timestamp": time.time()
                            }
                        })

                        self._emit_stats()

                except Exception as e:
                    self.event_bus.emit({
                        "type": "error",
                        "data": {"message": f"Packet processing error: {str(e)}"}
                    })

        except Exception as e:
            print(f"[IDS] Fatal error: {e}")
            self.event_bus.emit({
                "type": "status",
                "data": {"status": "error", "message": f"IDS Error: {str(e)}"}
            })

        finally:
            self.event_bus.emit({
                "type": "status",
                "data": {"status": "stopped", "message": "IDS Stopped"}
            })
            print("[IDS] Stopped")

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
                "active_flows": len(flows),
                "adr": round(adr, 1),
                "uptime": round(time.time() - self.start_time, 0) if self.start_time else 0
            }
        })
