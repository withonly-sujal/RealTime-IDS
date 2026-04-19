"""
Dashboard Entry Point — starts the IDS + Dashboard server.

Usage:
    python dashboard/run.py                    # Real IDS mode (requires Wireshark)
    python dashboard/run.py --demo             # Demo mode (simulated events)
    python dashboard/run.py --port 9000        # Custom port
    python dashboard/run.py --interface Ethernet  # Custom network interface

Run from the project root directory (e.g., E:\\RealTime-IDS).
"""

import sys
import asyncio
import argparse
from pathlib import Path
from contextlib import asynccontextmanager


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def check_dependencies():
    missing = []

    try:
        import fastapi
    except ImportError:
        missing.append("fastapi")

    try:
        import uvicorn
    except ImportError:
        missing.append("uvicorn")

    try:
        import websockets
    except ImportError:
        missing.append("websockets")

    if missing:
        print(f"\n[ERROR] Missing dependencies: {', '.join(missing)}")
        print(f"        Install with:  pip install {' '.join(missing)}")
        sys.exit(1)



_runner = None
_event_bus = None


def main():
    global _runner, _event_bus

    parser = argparse.ArgumentParser(
        description="Real-Time IDS Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dashboard/run.py --demo        Test dashboard with simulated data
  python dashboard/run.py               Run with live packet capture
  python dashboard/run.py --port 9000   Use a custom port
        """
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with simulated IDS events (no Wireshark needed)"
    )
    parser.add_argument(
        "--interface", type=str, default="Wi-Fi",
        help="Network interface for packet capture (default: Wi-Fi)"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Dashboard server port (default: 8765)"
    )
    parser.add_argument(
        "--flow-timeout", type=int, default=10,
        help="Flow expiry timeout in seconds (default: 10)"
    )

    args = parser.parse_args()

    check_dependencies()

    import uvicorn
    from dashboard.server import create_app, event_bus

    _event_bus = event_bus

    # Choose runner based on mode
    if args.demo:
        from dashboard.demo_runner import DemoRunner
        _runner = DemoRunner(event_bus)
    else:
        from dashboard.ids_runner import IDSRunner
        _runner = IDSRunner(
            event_bus,
            interface=args.interface,
            flow_timeout=args.flow_timeout
        )

    app = create_app(lifespan=_lifespan)

    mode = "DEMO MODE" if args.demo else f"LIVE MODE (interface: {args.interface})"
    print("")
    print("=" * 56)
    print("   [*] Real-Time IDS Dashboard")
    print(f"   Mode: {mode}")
    print(f"   Dashboard: http://localhost:{args.port}")
    print("=" * 56)
    print("")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning"
    )


@asynccontextmanager
async def _lifespan(app):
    _event_bus.set_loop(asyncio.get_event_loop())
    _runner.start()
    yield
    _runner.stop()


if __name__ == "__main__":
    main()
