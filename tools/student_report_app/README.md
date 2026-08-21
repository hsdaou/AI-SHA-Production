# Student Report Viewer

A small local web application for looking up student averages, subject marks,
missed exams, and infractions by student computer number.

## Setup and start

1. Double-click `setup.bat` once to create the virtual environment and install
   the two required packages.
2. Double-click `start_server.bat`.
3. Open <http://localhost:5000>.

If an older copy is already using port 5000, double-click
`start_server_5001.bat`. It starts the enhanced app at
<http://localhost:5001> and opens that address automatically.

Keep the command window open while using the application. Closing that window
stops the web server, even if the browser page remains open.

On the original Windows computer, the launchers connect to the authoritative
`C:\StudentReportApp\Houssam Report.csv`. The CSV contains private student data
and is deliberately excluded from source control. On another computer, set
`STUDENT_REPORT_DATA_FILE` to the protected report location. The application
automatically reloads the data after the file changes; a restart is not required.

## Configuration

The safe default is to listen only on this computer. Environment variables can
change the defaults:

- `STUDENT_REPORT_DATA_FILE`: absolute path to a different CSV report.
- `STUDENT_REPORT_HOST`: listening address. Use `0.0.0.0` only when the app must
  be reachable from the school network and the computer/network are trusted.
- `STUDENT_REPORT_PORT`: listening port (default `5000`).
- `LOG_LEVEL`: logging level (default `INFO`).

Never commit the report CSV, `.student_report.env`, SMTP credentials, or robot
key. The deployed Linux service reads those values from a mode-`0600`
environment file outside the repository.

The health endpoint at `/health` reports whether the CSV loaded successfully,
the number of indexed students, and non-fatal schema warnings.

## Tests

With the virtual environment active, run:

```text
python -m unittest discover -s tests -v
```
