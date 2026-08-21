import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import main


HEADERS = [
    "Student Number",
    "Student Name",
    "Grade",
    "Section",
    "AMS Average T3",
    "Periodic Average T3",
    "Missed AMS Exasms",
    "Frenh Average",
]


class StudentReportAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.report = Path(self.temp_dir.name) / "report.csv"
        with self.report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerow(
                {
                    "Student Number": "00123",
                    "Student Name": "Test Student",
                    "Grade": "9",
                    "Section": "A",
                    "AMS Average T3": "88",
                    "Periodic Average T3": "84",
                    "Missed AMS Exasms": "2",
                    "Frenh Average": "91",
                }
            )
        self.data_patch = patch.object(main, "DATA_FILE", self.report)
        self.data_patch.start()
        main._cache.update(mtime_ns=None, students=None, warnings=[])
        main.app.config.update(TESTING=True)
        self.client = main.app.test_client()

    def tearDown(self):
        self.data_patch.stop()
        self.temp_dir.cleanup()

    def test_index_is_self_contained(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Student Report Viewer", response.data)
        self.assertNotIn(b"cdn.jsdelivr.net", response.data)

    def test_search_preserves_leading_zeroes_and_aliases(self):
        response = self.client.post("/search", data={"student_id": "00123"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["student_name"], "Test Student")
        self.assertEqual(payload["missed_ams"], "2")
        self.assertIn({"name": "French", "mark": "91"}, payload["subjects"])

    def test_search_validation_and_not_found_statuses(self):
        self.assertEqual(self.client.post("/search", data={"student_id": ""}).status_code, 400)
        self.assertEqual(
            self.client.post("/search", data={"student_id": "does not exist"}).status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/search", data={"student_id": "99999"}).status_code,
            404,
        )

    def test_health_reports_loaded_record_count(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["student_count"], 1)

    def test_robot_report_fails_closed_without_key(self):
        response = self.client.get("/api/robot/student-report?student_id=00123")
        self.assertEqual(response.status_code, 401)

    def test_robot_report_never_returns_student_data(self):
        with patch.dict("os.environ", {"STUDENT_REPORT_ROBOT_KEY": "secret"}), patch.object(
            main, "_email_student_report", return_value=(True, "sent")
        ):
            response = self.client.get(
                "/api/robot/student-report?student_id=00123",
                headers={"x-robot-key": "secret"},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["emailed"])
        self.assertNotIn("student_name", payload)
        self.assertNotIn("subjects", payload)


if __name__ == "__main__":
    unittest.main()
