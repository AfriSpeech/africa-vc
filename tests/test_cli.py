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


def test_voice_bank_covers_the_datasets_voices():
    from africa_vc.voices import VOICES
    # The dataset is 30 voices; a bank that drifts from it offers a voice the
    # checkpoint never saw, or hides one it did.
    assert len(VOICES) == 30


def test_voice_names_are_case_insensitive():
    from africa_vc.voices import resolve
    assert resolve("sulafat") == "Sulafat"
    assert resolve("ZEPHYR") == "Zephyr"


def test_unknown_voice_says_where_to_look():
    import pytest
    from africa_vc.voices import resolve
    with pytest.raises(SystemExit) as exc:
        resolve("Nope")
    assert "africa-vc voices" in str(exc.value)


def test_convert_validates_before_touching_seedvc(capsys, monkeypatch):
    """Resolving a run clones Seed-VC and installs torch — minutes of work.

    A missing flag must be reported before any of that happens, so this fails
    the test if the code path reaches Seed-VC at all.
    """
    import africa_vc.seedvc as seedvc
    monkeypatch.setattr(seedvc, "ensure",
                        lambda *a, **k: pytest_fail("touched Seed-VC"))
    assert main(["convert", "--source", "x.wav"]) == 1
    assert "voice" in capsys.readouterr().err


def pytest_fail(msg):
    raise AssertionError(msg)


def test_cli_requires_a_subcommand(capsys):
    try:
        main([])
    except SystemExit as exc:
        assert exc.code != 0
