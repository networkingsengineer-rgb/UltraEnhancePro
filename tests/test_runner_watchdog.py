import sys
import time

import pytest

from ultraenhance.runner import EngineStalled, run_tool


def test_tick_runs_even_when_tool_prints_nothing():
    ticks = []
    run_tool([sys.executable, "-c", "import time; time.sleep(2.6)"], on_tick=lambda: ticks.append(1))
    assert len(ticks) >= 2


def test_watchdog_stops_a_silent_hung_tool():
    t0 = time.time()
    with pytest.raises(EngineStalled):
        run_tool([sys.executable, "-c", "import time; time.sleep(60)"],
                 progress_counter=lambda: 0, stall_seconds=1.5, name="Real-ESRGAN")
    assert time.time() - t0 < 10


def test_watchdog_leaves_a_working_tool_alone(tmp_path):
    script = ("import time, pathlib\n"
              f"d = pathlib.Path(r'{tmp_path}')\n"
              "for i in range(5):\n"
              "    time.sleep(0.5); (d / f'{i}.png').write_text('x')\n")
    run_tool([sys.executable, "-c", script],
             progress_counter=lambda: len(list(tmp_path.glob('*.png'))), stall_seconds=1.5)
    assert len(list(tmp_path.glob("*.png"))) == 5
