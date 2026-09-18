"""The parts that do not need a GPU: argument wiring and clip selection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from africa_vc import config
from africa_vc.cli import main
from africa_vc.prepare import _keep


def test_seedvc_is_pinned_to_a_commit():
    # An unpinned clone makes a run reproducible only by accident.
    assert len(config.SEEDVC_COMMIT) == 40


def test_default_config_is_the_speech_preset():
    # The tiny preset is real-time and sounds it; the 44k one is for singing.
    assert "whisper_small_wavenet" in config.DEFAULT_CONFIG


def test_keep_filters_on_voice_and_language():
    row = {"voice": "Zephyr", "language": "twi"}
    assert _keep(row, None, None)
    assert _keep(row, ["Zephyr"], ["twi"])
    assert not _keep(row, ["Puck"], None)
    assert not _keep(row, None, ["ewe"])


def test_cli_requires_a_subcommand(capsys):
    try:
        main([])
    except SystemExit as exc:
        assert exc.code != 0
