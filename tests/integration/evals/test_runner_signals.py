"""SIGTERM to the eval runner cleans up before it exits.

The runner runs in a child process here: the default SIGTERM action would kill pytest
itself. Its stack is the fake tier's (local processes, a temporary directory), started
for real but with no case run.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evals.run import EXIT_TERMINATED
from evals.stack import ROOT, run_command

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

# Replaces evaluate_cases with one that opens the real fake-tier stack, says what it
# started, and waits to be terminated.
_RUNNER = """
import asyncio, json, sys
from pathlib import Path
from evals import run

state = Path(sys.argv[1])

async def wait_inside_the_stack(*_, **__):
    async with run.open_stack(True, None) as stack:
        pids = [service.process.pid for service in stack._services.values()]
        workdir = str(stack.trace_dirs[0].parent)
        state.write_text(json.dumps({"pids": pids, "workdir": workdir}))
        await asyncio.sleep(120)

run.evaluate_cases = wait_inside_the_stack
sys.exit(run.main(["--fake", "--case", "status_question", "--out", sys.argv[2]]))
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(path: Path, timeout_s: float = 30.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists() and (text := path.read_text()):
            return text
        time.sleep(0.05)
    raise AssertionError(f"{path} was not written")


def test_sigterm_stops_the_fake_stack_and_removes_its_directory(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    runner = subprocess.Popen(
        [sys.executable, "-c", _RUNNER, str(state), str(tmp_path / "out")],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        started = json.loads(_wait_for(state))
        assert started["pids"]
        assert all(_alive(pid) for pid in started["pids"])
        runner.send_signal(signal.SIGTERM)
        _, err = runner.communicate(timeout=60)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()
    assert runner.returncode == EXIT_TERMINATED, err
    assert "terminated" in err
    assert not any(_alive(pid) for pid in started["pids"])
    assert not Path(started["workdir"]).exists()
    assert not (tmp_path / "out").exists()


async def test_a_cancelled_command_does_not_outlive_the_runner(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    command = run_command(["sh", "-c", f'echo $$ > "{pid_file}"; exec sleep 60'])
    task = asyncio.create_task(command)
    pid = int(await asyncio.to_thread(_wait_for, pid_file))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _alive(pid)
