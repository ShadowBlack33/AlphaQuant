from pathlib import Path


def test_sample_prob_summary_exists():
    assert Path("data/samples/prob_summary_sample.csv").exists()
