"""Tests for the move_stage.py CLI (Zolix ZC300 minimal command-line tool)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import move_stage


def test_build_stage_simulate_returns_simulated():
    stage = move_stage.build_stage("COM3", simulate=True)
    from stage.simulated import SimulatedStage
    assert isinstance(stage, SimulatedStage)


def test_build_stage_real_returns_zolix():
    stage = move_stage.build_stage("COM3", simulate=False)
    from stage.zolix_zc300 import ZolixZC300
    assert isinstance(stage, ZolixZC300)


class TestStatusCommand:
    def test_status_simulate(self, capsys):
        rc = move_stage.main(["--simulate", "status"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Position (pulses):" in out
        assert "x:" in out and "y:" in out and "r:" in out
        assert "stopped" in out


class TestRelativeMove:
    def test_rel_positive(self, capsys):
        rc = move_stage.main(["--simulate", "rel", "x", "1000"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Moved x by 1000 pulses" in out

    def test_rel_negative(self, capsys):
        rc = move_stage.main(["--simulate", "rel", "y", "-500"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Moved y by -500 pulses" in out

    def test_rel_no_wait(self, capsys):
        rc = move_stage.main(["--simulate", "rel", "x", "100", "--no-wait"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Moved x by 100 pulses" in out


class TestAbsoluteMove:
    def test_abs(self, capsys):
        rc = move_stage.main(["--simulate", "abs", "y", "25000"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Moved y to 25000.0 pulses" in out

    def test_abs_float(self, capsys):
        rc = move_stage.main(["--simulate", "abs", "r", "1234.5"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Moved r to 1234.5 pulses" in out


class TestHomeCommand:
    def test_home_single_axis(self, capsys):
        rc = move_stage.main(["--simulate", "home", "x"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Homed x" in out

    def test_home_all(self, capsys):
        rc = move_stage.main(["--simulate", "home", "all"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Homed all" in out

    def test_home_default_all(self, capsys):
        rc = move_stage.main(["--simulate", "home"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Homed all" in out


class TestStopCommand:
    def test_stop(self, capsys):
        rc = move_stage.main(["--simulate", "stop"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Stopped all axes" in out


class TestSpeedCommand:
    def test_speed(self, capsys):
        rc = move_stage.main(["--simulate", "speed", "x", "2000"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Set x speed to 2000.0 pulses/s" in out


class TestErrorHandling:
    def test_invalid_axis(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            move_stage.main(["--simulate", "rel", "z", "100"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "invalid choice" in err

    def test_connection_failure(self, capsys, monkeypatch):
        def fail_connect(self):
            raise Exception("Port not found")

        from stage.zolix_zc300 import ZolixZC300
        monkeypatch.setattr(ZolixZC300, "connect", fail_connect)

        rc = move_stage.main(["--port", "COM999", "status"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "Connection failed" in err