"""Jev に渡す問題を生成する（純関数）。"""

from __future__ import annotations

SEED = 20260929
# 尤度比から感度・偽陽性率を作るときの基準値（SPEC §3 ②）
BASE_RATE = 0.9


def posterior(prior: float, sens: float, fpr: float) -> float:
    """陽性だったときの事後確率（ベイズの定理）。"""
    numerator = prior * sens
    return numerator / (numerator + (1 - prior) * fpr)


def sens_fpr_from_lr(lr: float) -> tuple[float, float]:
    """尤度比から (感度, 偽陽性率) を作る。LR>=1 は感度 0.9 固定、LR<1 は偽陽性率 0.9 固定。"""
    if lr <= 0:
        raise ValueError(f"尤度比は正の数でなければならない: {lr}")
    if lr >= 1:
        return BASE_RATE, BASE_RATE / lr
    return BASE_RATE * lr, BASE_RATE


def fmt_pct(p: float) -> str:
    """確率をパーセント表記の文字列にする（末尾の 0 と小数点を落とす）。"""
    return f"{p * 100:.2f}".rstrip("0").rstrip(".")
