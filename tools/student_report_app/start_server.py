"""Production launcher for Student Report Viewer."""

import logging
import os
import socket
import sys

from waitress import serve

from src.main import app


if __name__ == "__main__":
    host = os.getenv("STUDENT_REPORT_HOST", "127.0.0.1")
    port = int(os.getenv("STUDENT_REPORT_PORT", "5000"))
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    # Fail with a useful message instead of letting Waitress emit an opaque
    # socket error when an older copy of the app is already using the port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(
                f"\nPort {port} is already in use. Another Student Report Viewer "
                "may still be running.\n"
                "Close its command window, or run start_server_5001.bat to use "
                "http://localhost:5001 instead.\n",
                file=sys.stderr,
            )
            raise SystemExit(1)

    logging.getLogger(__name__).info(
        "Student Report Viewer is available at http://%s:%s", host, port
    )
    serve(app, host=host, port=port, threads=4)
