"""
The per-step stage log: one record per stage, readable after the run.

The point of these is that a wrong dub can be diagnosed from the log alone,
so the tests assert the properties a reader depends on — a record per stage,
in order, parseable, and never able to fail a render.
"""

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))


class StageLogTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._old = os.environ.get("VIDEOTRANS_LOG_DIR")
        os.environ["VIDEOTRANS_LOG_DIR"] = self.dir
        import stage_log
        self.log = stage_log
        self.log._DIR = None  # re-resolve against the temp dir

    def tearDown(self):
        if self._old is None:
            os.environ.pop("VIDEOTRANS_LOG_DIR", None)
        else:
            os.environ["VIDEOTRANS_LOG_DIR"] = self._old
        self.log._DIR = None

    def test_a_stage_record_is_written_and_read_back(self):
        self.log.stage("job-a", "diarize", speaker_count=2,
                       speakers=["SPEAKER_00", "SPEAKER_01"])
        records = self.log.read("job-a")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["stage"], "diarize")
        self.assertEqual(records[0]["speaker_count"], 2)
        self.assertEqual(records[0]["job"], "job-a")

    def test_records_are_appended_in_order(self):
        for name in ("probe", "transcribe", "diarize", "translate"):
            self.log.stage("job-b", name, ok=True)
        stages = [r["stage"] for r in self.log.read("job-b")]
        self.assertEqual(stages, ["probe", "transcribe", "diarize", "translate"])

    def test_each_line_is_valid_json(self):
        self.log.stage("job-c", "verify", sync_ok=True, nested={"a": [1, 2]})
        self.log.event("job-c", "verify", "a note", detail="x")
        path = self.log.path_for("job-c")
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertIsInstance(json.loads(line), dict)

    def test_a_value_that_cannot_serialise_does_not_raise(self):
        class Odd:
            def __repr__(self):
                return "<odd>"

        self.log.stage("job-d", "probe", thing=Odd())
        record = self.log.read("job-d")[0]
        self.assertEqual(record["thing"], "<odd>")

    def test_an_unwritable_log_directory_is_not_fatal(self):
        self.log._DIR = ""
        # Must return quietly rather than raising into a render.
        self.log.stage("job-e", "probe", x=1)
        self.assertIsNone(self.log.path_for("job-e"))

    def test_reading_a_run_with_no_log_returns_empty(self):
        self.assertEqual(self.log.read("never-ran"), [])

    def test_a_human_readable_line_is_written_beside_the_json(self):
        self.log.stage("job-f", "diarize", speaker_count=3)
        human = self.log.path_for("job-f").replace(".jsonl", ".log")
        self.assertTrue(os.path.isfile(human))
        self.assertIn("diarize", open(human, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
