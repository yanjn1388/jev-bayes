from jev_bayes.analyze import summarize
from jev_bayes.figures import write_figures
from jev_bayes.problems import generate_all


def test_write_figures_creates_three_pngs(tmp_path):
    records = [
        {**p, "q": p["p_true"], "q_spread": 0.0, "n_ok": 1, "chosen": p["hidden_outcome"]}
        for p in generate_all()
    ]
    summary = summarize(records, n_problems=len(records), n_boot=50)
    paths = write_figures(records, summary, tmp_path)
    assert sorted(p.name for p in paths) == ["brier.png", "coefficients.png", "ladder.png"]
    for p in paths:
        assert p.stat().st_size > 1000
