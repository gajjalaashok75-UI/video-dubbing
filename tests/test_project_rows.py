"""
The project list must show a project that has not finished yet.

`GET /videos/` used to skip every job whose status was not `done`, so the
moment you started a render the project vanished from the list — and when the
runtime died there was nothing left saying it had ever been started. These
tests pin the rows the endpoint builds.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import project


def rows(jobs):
    return project.video_rows(jobs)


class VideoRowsTest(unittest.TestCase):

    def test_a_project_that_is_still_rendering_is_listed(self):
        jobs = {
            "abc": {
                "status": "processing", "step": "lip_syncing",
                "project_id": "abc", "title": "demo",
                "target_language": "te", "started_at": 1_700_000_000.0,
            },
        }
        out = rows(jobs)
        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row["id"], "abc")
        self.assertEqual(row["job_id"], "abc")
        self.assertEqual(row["processing_status"], "processing")
        self.assertTrue(row["openable"])
        self.assertEqual(row["outputs"], 0)
        self.assertEqual(row["title"], "demo")
        self.assertIsNotNone(row["created_at"])

    def test_a_queued_project_is_listed_too(self):
        out = rows({"abc": {"status": "queued", "project_id": "abc",
                            "started_at": 1_700_000_000.0}})
        self.assertEqual([r["processing_status"] for r in out], ["queued"])

    def test_one_row_per_project_however_many_outputs(self):
        jobs = {
            "one": {"status": "done", "project_id": "one", "title": "demo",
                    "saved_at": 1_700_000_100.0},
            "two": {"status": "done", "project_id": "one", "title": "demo",
                    "saved_at": 1_700_000_200.0},
            "three": {"status": "done", "project_id": "one", "title": "demo",
                      "saved_at": 1_700_000_300.0},
        }
        out = rows(jobs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["outputs"], 3)
        self.assertEqual(out[0]["job_id"], "one")     # the one that founded it

    def test_a_running_job_is_what_the_row_is_about(self):
        jobs = {
            "done1": {"status": "done", "project_id": "p", "title": "demo",
                      "saved_at": 1_700_000_100.0, "url": "/jobs/done1/video"},
            "live": {"status": "processing", "project_id": "p", "title": "demo",
                     "started_at": 1_700_000_900.0},
        }
        out = rows(jobs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["processing_status"], "processing")
        self.assertEqual(out[0]["job_id"], "live")
        self.assertEqual(out[0]["outputs"], 1)        # the finished one still counts
        self.assertIsNone(out[0]["dubbed_url"])       # it has produced nothing yet

    def test_a_failed_run_carries_its_reason(self):
        jobs = {"bad": {"status": "failed", "project_id": "bad",
                        "error": "Ran out of memory while syncing lips.",
                        "started_at": 1_700_000_000.0}}
        out = rows(jobs)
        self.assertEqual(out[0]["processing_status"], "failed")
        self.assertEqual(out[0]["error_message"],
                         "Ran out of memory while syncing lips.")

    def test_a_cancelled_run_is_not_a_project(self):
        self.assertEqual(rows({"gone": {"status": "cancelled"}}), [])

    def test_newest_first(self):
        jobs = {
            "old": {"status": "done", "project_id": "old",
                    "saved_at": 1_700_000_000.0},
            "new": {"status": "done", "project_id": "new",
                    "saved_at": 1_700_090_000.0},
            "live": {"status": "processing", "project_id": "live",
                     "started_at": 1_700_100_000.0},
        }
        self.assertEqual([r["id"] for r in rows(jobs)], ["live", "new", "old"])

    def test_rows_are_json_safe(self):
        import json
        out = rows({"abc": {"status": "done", "project_id": "abc",
                            "saved_at": 1_700_000_000.0}})
        json.dumps(out)                               # the endpoint serialises these


if __name__ == "__main__":
    unittest.main()
