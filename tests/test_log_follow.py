"""
The log file must not eat itself.

`start_server` mirrors stdout into `server.log`. The notebook's follow cell used
to print the lines it had just read with a plain `print()`, so every line was
appended back into the file being read and copied forever: one session produced
a 548 MB `server.log` with 9,124,224 lines, of which only 86 were real, and it
kept the kernel busy for hours.

`follow_log` writes with the mirror suspended for its own thread. These tests
reproduce the original loop and assert it cannot form.
"""

from __future__ import annotations

import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import deploy_helpers as runtime  # noqa: E402

ORIGINAL = "line one\nline two\nline three\n"


class LogFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.log = Path(self.tmp.name) / "server.log"
        self.log.write_text(ORIGINAL, encoding="utf-8")

        # The real arrangement: the log is appended to by a handle that the tee
        # writes through, and stdout is a _Tee over it.
        self.mirror = open(self.log, "a", encoding="utf-8", buffering=1)
        self.addCleanup(self.mirror.close)

        self.cell = io.StringIO()          # stands in for the notebook's cell output
        self.real_stdout = sys.stdout
        sys.stdout = runtime._Tee(self.cell, self.mirror)
        self.addCleanup(self._restore)

    def _restore(self):
        sys.stdout = self.real_stdout

    def _text(self) -> str:
        return self.log.read_text(encoding="utf-8")

    def _lines(self) -> list[str]:
        return self._text().splitlines()

    # --- the fix ---------------------------------------------------------
    def test_following_a_static_log_does_not_grow_it(self):
        runtime.follow_log(self.log, stop_at_eof=True)
        self.assertEqual(self._text(), ORIGINAL)
        self.assertIn("line one", self.cell.getvalue(),
                      "the follower must still show the log in the cell")

    def test_repeated_following_never_grows_the_log(self):
        for _ in range(5):
            runtime.follow_log(self.log, stop_at_eof=True)
        self.assertEqual(self._text(), ORIGINAL,
                         "each pass copied the log back into itself")

    def test_a_server_line_is_shown_once_and_not_echoed(self):
        # A real server write: through the tee, so it IS mirrored (as intended).
        sys.stdout.write("[APP] render started\n")
        self.assertIn("[APP] render started", self._text())

        runtime.follow_log(self.log, stop_at_eof=True)
        self.assertEqual(self._lines().count("[APP] render started"), 1)

    def test_suspension_is_per_thread(self):
        """A render worker logging at the same moment must still reach the file."""
        def worker():
            sys.stdout.write("from a render worker\n")

        with runtime.mirror_suspended():
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

        self.assertIn("from a render worker", self._text())

    def test_nested_suspension_restores_mirroring(self):
        with runtime.mirror_suspended():
            with runtime.mirror_suspended():
                print("hidden", end="")
        print("visible")

        text = self._text()
        self.assertNotIn("hidden\n", text)
        self.assertIn("visible", text)

    # --- the bug this replaced -------------------------------------------
    def test_the_old_plain_print_loop_did_feed_back(self):
        """
        The removed pattern, kept as a test so the reason survives.

        If this ever stops failing to grow the file, the tee no longer mirrors
        and `follow_log`'s suspension is unnecessary - not the other way round.
        """
        before = len(self._lines())
        with open(self.log, encoding="utf-8") as fh:
            print(fh.read(), end="")       # exactly what the old cell did
        self.assertGreater(len(self._lines()), before,
                           "printing a read line is mirrored back into the log")


if __name__ == "__main__":
    unittest.main()
