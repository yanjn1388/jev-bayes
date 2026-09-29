import math

from jev_bayes.problems import fmt_pct, posterior, sens_fpr_from_lr


def test_posterior_classic_example():
    # 有病率 1%・感度 90%・偽陽性率 9% → 約 9.2%
    p = posterior(0.01, 0.9, 0.09)
    assert math.isclose(p, 0.009 / (0.009 + 0.99 * 0.09))
    assert abs(p - 0.0917) < 1e-3


def test_posterior_uninformative_test_returns_prior():
    assert math.isclose(posterior(0.2, 0.5, 0.5), 0.2)


def test_sens_fpr_from_lr_above_one():
    sens, fpr = sens_fpr_from_lr(20)
    assert math.isclose(sens, 0.9)
    assert math.isclose(fpr, 0.045)
    assert math.isclose(sens / fpr, 20)


def test_sens_fpr_from_lr_below_one():
    sens, fpr = sens_fpr_from_lr(0.2)
    assert math.isclose(sens, 0.18)
    assert math.isclose(fpr, 0.9)
    assert math.isclose(sens / fpr, 0.2)


def test_sens_fpr_from_lr_rejects_nonpositive():
    import pytest

    with pytest.raises(ValueError):
        sens_fpr_from_lr(0)


def test_fmt_pct_strips_trailing_zeros():
    assert fmt_pct(0.045) == "4.5"
    assert fmt_pct(0.01) == "1"
    assert fmt_pct(0.9 * 0.2) == "18"
    assert fmt_pct(0.8) == "80"
