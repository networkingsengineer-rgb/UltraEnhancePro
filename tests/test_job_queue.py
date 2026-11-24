import threading
import time

from ultraenhance.job_queue import JobQueue, JobState, friendly_error


def wait_state(job, states, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if job.state in states:
            return True
        time.sleep(0.01)
    return False


def test_runs_in_order_and_reports_progress():
    q = JobQueue()
    order, progress = [], []
    q.on("progress", lambda j: progress.append(j.progress))

    def make(i):
        def f(ctl):
            ctl.progress(0.5, "half")
            order.append(i)
            return i * 10
        return f
    jobs = [q.add("photo", f"f{i}", make(i)) for i in range(3)]
    assert q.wait_all(5)
    assert order == [0, 1, 2]
    assert [j.result for j in jobs] == [0, 10, 20]
    assert all(j.state == JobState.DONE for j in jobs)
    assert 0.5 in progress
    q.shutdown()


def test_failure_is_reported_in_plain_english():
    q = JobQueue()

    def bad(ctl):
        raise MemoryError()
    j = q.add("photo", "x", bad)
    q.wait_all(5)
    assert j.state == JobState.FAILED
    assert "Tile size" in j.error
    q.shutdown()


def test_cancel_running_job():
    q = JobQueue()
    started = threading.Event()

    def long(ctl):
        started.set()
        for _ in range(500):
            ctl.check()
            time.sleep(0.01)
        return "finished"
    j = q.add("video", "v", long)
    assert started.wait(2)
    q.cancel(j)
    assert wait_state(j, {JobState.CANCELLED})
    q.shutdown()


def test_cancel_queued_job_never_runs():
    q = JobQueue()
    gate = threading.Event()
    ran = []
    q.add("photo", "a", lambda ctl: gate.wait(2))
    b = q.add("photo", "b", lambda ctl: ran.append(1))
    q.cancel(b)
    gate.set()
    q.wait_all(5)
    assert b.state == JobState.CANCELLED and not ran
    q.shutdown()


def test_pause_and_resume():
    q = JobQueue()
    steps = []
    started = threading.Event()

    def work(ctl):
        started.set()
        for i in range(30):
            ctl.check()
            steps.append(i)
            time.sleep(0.01)
    j = q.add("video", "v", work)
    assert started.wait(2)
    q.pause(j)
    time.sleep(0.15)
    n = len(steps)
    time.sleep(0.2)
    assert len(steps) <= n + 1  # stopped advancing
    assert j.state == JobState.PAUSED
    q.resume(j)
    assert q.wait_all(5)
    assert j.state == JobState.DONE and len(steps) == 30
    q.shutdown()


def test_pause_all_holds_new_jobs():
    q = JobQueue()
    q.pause()
    ran = []
    j = q.add("photo", "p", lambda ctl: ran.append(1))
    time.sleep(0.3)
    assert not ran and j.state == JobState.QUEUED
    q.resume()
    assert q.wait_all(5) and ran == [1]
    q.shutdown()


def test_retry_failed_job():
    q = JobQueue()
    calls = []

    def flaky(ctl):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "ok"
    j = q.add("photo", "p", flaky)
    q.wait_all(5)
    assert j.state == JobState.FAILED
    q.retry(j)
    q.wait_all(5)
    assert j.state == JobState.DONE and j.result == "ok"
    q.shutdown()


def test_move_and_remove_finished():
    q = JobQueue()
    q.pause()
    a = q.add("photo", "a", lambda c: 1)
    b = q.add("photo", "b", lambda c: 2)
    q.move(b, -1)
    assert q.jobs[0] is b
    q.resume()
    q.wait_all(5)
    q.remove_finished()
    assert q.jobs == []
    q.shutdown()


def test_friendly_errors():
    assert "driver" in friendly_error(RuntimeError("vkCreateInstance failed -9"))
    assert "disk is full" in friendly_error(OSError(28, "No space left on device"))
