from __future__ import annotations

import threading
import time

from wavestereo.visual.core import LatencyMonitor


def _monitor_with_one_displayed_frame() -> LatencyMonitor:
    monitor = LatencyMonitor()
    monitor.record_frame_capture_start(1, time.time() - 0.01)
    monitor.record_display(1)
    return monitor


def test_latency_stats_waits_for_monitor_lock():
    monitor = _monitor_with_one_displayed_frame()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    result = {}

    def read_stats():
        reader_started.set()
        result["stats"] = monitor.get_latency_stats()
        reader_finished.set()

    monitor.lock.acquire()
    try:
        reader = threading.Thread(target=read_stats)
        reader.start()
        assert reader_started.wait(timeout=1.0)
        assert not reader_finished.wait(timeout=0.1)
    finally:
        monitor.lock.release()

    reader.join(timeout=1.0)
    assert not reader.is_alive()
    assert result["stats"] is not None


def test_latency_stats_are_safe_during_concurrent_frame_updates():
    monitor = _monitor_with_one_displayed_frame()
    writer_finished = threading.Event()
    errors = []

    def write_frames():
        try:
            for frame_id in range(2, 1002):
                monitor.record_frame_capture_start(frame_id, time.time())
                monitor.record_submit_to_inference_queue(frame_id)
                if frame_id % 3 == 0:
                    monitor.record_display(frame_id)
                if frame_id % 10 == 0:
                    time.sleep(0)
        except Exception as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)
        finally:
            writer_finished.set()

    writer = threading.Thread(target=write_frames)
    writer.start()
    while not writer_finished.is_set():
        try:
            stats = monitor.get_latency_stats()
            assert stats is not None
        except Exception as exc:  # pragma: no cover - regression capture
            errors.append(exc)
            break

    writer.join(timeout=2.0)
    assert not writer.is_alive()
    assert errors == []
