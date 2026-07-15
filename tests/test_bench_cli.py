"""Unit tests for the bench CLI wiring — the parts provable without a mic or a
model: `--list-models` output and the `--models` selector parsing. The record/run
bodies (Steps 3/5) are exercised by their own tests.
"""

from susurro.bench import cli


def test_list_models_shows_both_fw_ids_with_availability_and_exits_zero(capsys):
    rc = cli.main(["--list-models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fw-large-v3-turbo" in out
    assert "fw-large-v3" in out
    assert "available" in out  # fw is a core dep, so it probes as available here


def test_parse_models_splits_and_strips_a_comma_list():
    assert cli.parse_models("fw-large-v3-turbo, fw-large-v3 ") == [
        "fw-large-v3-turbo",
        "fw-large-v3",
    ]


def test_parse_models_none_or_blank_means_all_models():
    # None sentinel -> "no filter", the runner then takes all available models.
    assert cli.parse_models(None) is None
    assert cli.parse_models("") is None


def test_no_subcommand_prints_usage_and_returns_nonzero(capsys):
    rc = cli.main([])
    assert rc != 0
    # help/usage mentions the two subcommands so a bare invocation is self-documenting.
    combined = capsys.readouterr()
    assert "record" in (combined.out + combined.err)
    assert "run" in (combined.out + combined.err)
