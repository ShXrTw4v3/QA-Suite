#!/usr/bin/env python3
"""
hospital_server.py — Micro-QA + TRS-398 Unified Hospital Network Server
=============================================================================
Deploys the combined Micro-QA Quality Assurance & IAEA TRS-398 Reference
Dosimetry platform on the local hospital network.

Binds to 0.0.0.0 so that any authorized desktop, laptop, or tablet on the
hospital intranet / private IP subnet can access the application via a standard
web browser without requiring any software installation on client machines.
"""

import os
import sys
import socket
import subprocess
import argparse
import webbrowser
import threading
import time

DEFAULT_PORT = 8501
STANDALONE_PORT = 8000

def get_all_lan_ips():
    """Returns a list of all private local network IPv4 addresses for this machine."""
    ips = set()
    
    # Method 1: UDP routing socket (fastest & most accurate for active default gateway)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        if not ip.startswith("127."):
            ips.add(ip)
        s.close()
    except Exception:
        pass

    # Method 2: Hostname address resolution
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None):
            ip = item[4][0]
            if ":" not in ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    # Method 3: Socket connect to external probe
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        if not ip.startswith("127."):
            ips.add(ip)
        s.close()
    except Exception:
        pass

    if not ips:
        ips.add("127.0.0.1")
        
    return sorted(list(ips))


def ensure_config_toml(port: int):
    """Ensures .streamlit/config.toml is configured for local hospital network deployment."""
    dot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit")
    os.makedirs(dot_dir, exist_ok=True)
    config_path = os.path.join(dot_dir, "config.toml")
    
    config_content = f"""[server]
address = "0.0.0.0"
port = {port}
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#0284c7"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f1f5f9"
textColor = "#0f172a"
font = "sans serif"
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)


def print_banner(lan_ips, port):
    """Prints a clear, clinical server startup banner."""
    primary_ip = lan_ips[0] if lan_ips else "127.0.0.1"
    
    print("\n" + "=" * 82)
    print("   🏥  MICRO-QA + IAEA TRS-398 CLINICAL REFERENCE DOSIMETRY SERVER")
    print("       Local Hospital Network / Intranet Deployment (Private IP)")
    print("=" * 82)
    print(f"\n Server is actively listening on all network interfaces (0.0.0.0:{port})")
    print("\n ACCESS URLS FOR CLINIC STAFF:")
    print(f"   ► On this Host Workstation:     http://localhost:{port}")
    for ip in lan_ips:
        print(f"   ► Local Hospital Network URL:   http://{ip}:{port}  <-- Share this with colleagues")
    
    print("\n MODULES AVAILABLE ON THIS SERVER:")
    print("   [1] 🏥 Today's QA Dashboard          [6] 📊 Trajectory Log Analyzer (.bin)")
    print("   [2] 📋 QA Data Entry (Pass/Tol/Act)  [7] 🔲 Picket Fence MLC QA (.dcm)")
    print("   [3] 📈 Trending & Control Charts     [8] 🎯 Winston-Lutz Isocenter QA (.zip)")
    print("   [4] ☢️ TRS-398 Absolute Dosimetry     [9] 🔬 CBCT CatPhan 3D QA (.zip)")
    print("   [5] 🌐 Embedded Interactive Suite   [10] 🔧 Linac Service & Downtime Logs")

    print("\n NETWORK & FIREWALL GUIDELINES:")
    print("   • Windows Defender Firewall: Open inbound TCP Port " + str(port))
    print(f"     cmd: netsh advfirewall firewall add rule name=\"MicroQA\" dir=in action=allow protocol=TCP localport={port}")
    print("   • macOS Firewall: System Settings > Network > Firewall > Allow incoming connections")
    print("   • Linux (Ubuntu/Debian): sudo ufw allow " + str(port) + "/tcp")
    print("\n Press Ctrl+C at any time to gracefully stop the hospital server.")
    print("=" * 82 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Micro-QA & TRS-398 on the local hospital private network."
    )
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT,
                        help=f"TCP port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--browser", "-b", action="store_true",
                        help="Automatically open the application in local default browser")
    parser.add_argument("--standalone", "-s", action="store_true",
                        help=f"Also start the lightweight standalone web server on port {STANDALONE_PORT}")

    args = parser.parse_args()

    # Move to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Set Matplotlib directory to a writable directory to suppress warnings
    mpl_dir = os.path.join(script_dir, ".mpl_cache")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir

    # Detect Python runtime
    venv_python = os.path.join(script_dir, "venv", "bin", "python3")

    # Windows fallback
    if sys.platform == "win32":
        venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")

    # Verify runtime (use `python -m streamlit` so relocated venvs work cleanly)
    if os.path.exists(venv_python):
        streamlit_cmd = [venv_python, "-m", "streamlit"]
    else:
        streamlit_cmd = [sys.executable, "-m", "streamlit"]

    # Ensure Streamlit config file
    ensure_config_toml(args.port)

    # Detect network IPs
    lan_ips = get_all_lan_ips()
    print_banner(lan_ips, args.port)

    # Optional standalone server in background thread
    if args.standalone:
        def run_standalone():
            import http.server
            import socketserver
            class Handler(http.server.SimpleHTTPRequestHandler):
                def log_message(self, format, *args): pass
            with socketserver.TCPServer(("0.0.0.0", STANDALONE_PORT), Handler) as httpd:
                httpd.serve_forever()
        t = threading.Thread(target=run_standalone, daemon=True)
        t.start()
        print(f" [✓] Standalone HTML5 visual suite running on: http://{lan_ips[0]}:{STANDALONE_PORT}/index.html\n")

    # Optional auto-open browser
    if args.browser:
        def open_b():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=open_b, daemon=True).start()

    # Launch Streamlit server
    cmd = streamlit_cmd + [
        "run",
        "app.py",
        "--server.address", "0.0.0.0",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down Micro-QA hospital server. Data safely preserved in qa_data.db.")
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
