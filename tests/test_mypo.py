from mypo import __version__


def test_version() -> None:
    assert __version__ == "0.0.0"  # refresh automatically by CI
