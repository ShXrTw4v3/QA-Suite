#!/usr/bin/env python3
"""
Local HTTP Server for IAEA TRS-398 Reference Dosimetry Web Application
Auto-detects open port and launches the system browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import threading
import time

DEFAULT_PORT = 8000
MAX_PORT_ATTEMPTS = 20

class CustomHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable caching for instant updates
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

def find_available_port(start_port: int) -> int:
    import socket
    for port in range(start_port, start_port + MAX_PORT_ATTEMPTS):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise OSError("No open ports found in range.")

def start_server(port: int = DEFAULT_PORT, auto_open: bool = True):
    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)

    actual_port = find_available_port(port)
    url = f"http://localhost:{actual_port}/index.html"

    print("=" * 70)
    print("  IAEA TRS-398 ABSOLUTE REFERENCE DOSIMETRY APP")
    print("  High-Energy Photon Beams (6 MV, PTW Farmer, SSD 100, z=10 cm)")
    print("=" * 70)
    print(f"\nServing application from: {web_dir}")
    print(f"URL: {url}\n")
    print("Press Ctrl+C to terminate the server.\n")

    if auto_open:
        def open_browser():
            time.sleep(0.6)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    with socketserver.TCPServer(("", actual_port), CustomHTTPHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server. Clinical measurements saved.")
            httpd.server_close()

if __name__ == "__main__":
    port_arg = DEFAULT_PORT
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port_arg = int(sys.argv[1])
    start_server(port=port_arg, auto_open=True)
