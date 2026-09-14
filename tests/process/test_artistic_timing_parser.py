from pathlib import Path

from src.process.simulators.artistic.parser import parse_thermo_log
from src.process.simulators.artistic.runner import _progress


def test_parser_separates_minimization_and_dynamics_loop_times(tmp_path: Path):
    log = tmp_path / "slurry.log"
    log.write_text("Step Energy\n0 -1\nLoop time of 2.5 on 4 procs for 100 steps\nMinimization stats:\nStep Temp\n100 300\n1100 301\nLoop time of 4.0 on 4 procs for 1000 steps\n", encoding="utf-8")
    parsed = parse_thermo_log(log, minimization_expected=True)
    assert [(item.phase, item.wall_seconds, item.reported_steps) for item in parsed.timings] == [("minimization", 2.5, 100), ("dynamics", 4.0, 1000)]
    progress = _progress(tmp_path, [{"stage": "slurry", "wall_seconds": 7.0}], 1_000, 1_000, None, minimization_expected=True)
    assert progress["minimization_wall_seconds"] == 2.5
    assert progress["dynamics_wall_seconds"] == 4.0
    assert progress["total_slurry_command_wall_seconds"] == 7.0
    assert progress["dynamics_steps"] == 1_000
    assert progress["dynamics_steps_per_second"] == 250.0
