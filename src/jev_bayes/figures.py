"""記事用の図3枚を描く（SPEC §4.5）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jev_bayes.analyze import load_records  # noqa: E402
from jev_bayes.problems import read_jsonl  # noqa: E402

plt.rcParams["font.family"] = ["Hiragino Sans", "Noto Sans CJK JP", "DejaVu Sans"]

STAGE_COLORS = {1: "#7a7a7a", 2: "#1f6feb", 3: "#d97706"}
STAGE_LABELS = {1: "① 手がかりなし", 2: "② 計算すれば出る", 3: "③ 書いてある"}
CONDITION_COLORS = {"noul-en": "#1f6feb", "choice-en": "#d97706", "noul-ja": "#16a34a"}
DPI = 150


def fig_ladder(records: list[dict], path: Path) -> Path:
    """横軸 真の確率 p・縦軸 報告確率 q。Noul と Choice を左右に並べる。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, qtype in zip(axes, ("noul", "choice")):
        ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="理想（報告 = 真の確率）")
        for stage in (1, 2, 3):
            rs = [r for r in records if r["stage"] == stage and r["qtype"] == qtype and r["lang"] == "en"]
            if rs:
                ax.scatter([r["p_true"] for r in rs], [r["q"] for r in rs], s=18, alpha=0.7,
                           color=STAGE_COLORS[stage], label=STAGE_LABELS[stage])
        ax.set_title(qtype.capitalize())
        ax.set_xlabel("真の確率 p")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Jev の報告確率 q")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_coefficients(summary: dict, path: Path) -> Path:
    """(b1, b2) 平面にベイズの点 (1, 1) と推定値・95% 区間を置く。"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter([1], [1], marker="*", s=220, color="black", label="ベイズ (1, 1)", zorder=3)
    for key, color in CONDITION_COLORS.items():
        s = summary["stage2"].get(key)
        if not s:
            continue
        b1, b2 = s["b1"], s["b2"]
        xerr = [[max(0.0, b1 - s["ci_low"][1])], [max(0.0, s["ci_high"][1] - b1)]]
        yerr = [[max(0.0, b2 - s["ci_low"][2])], [max(0.0, s["ci_high"][2] - b2)]]
        ax.errorbar(b1, b2, xerr=xerr, yerr=yerr, fmt="o", color=color, capsize=4, label=key)
    ax.axvline(1, color="gray", lw=0.8, ls=":")
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("b1（事前確率の重み）")
    ax.set_ylabel("b2（尤度比の重み）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _brier_rows(summary: dict) -> list[tuple[str, float, float]]:
    rows = [(f"① {k}", v["model_error"], v["irreducible"]) for k, v in summary["stage1"].items()]
    rows += [(f"② {k}", summary["stage2"][k]["model_error"], summary["stage2"][k]["irreducible"])
             for k in CONDITION_COLORS if k in summary["stage2"]]
    rows += [(f"③ {k}", v["model_error"], v["irreducible"]) for k, v in summary["stage3"].items()]
    return rows


def fig_brier(summary: dict, path: Path) -> Path:
    """条件ごとに「Jev の誤り」と「減らせない不確実性」を積み上げた横棒。"""
    rows = _brier_rows(summary)
    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.5))
    ax.barh(labels, [r[2] for r in rows], color="#cbd5e1", label="減らせない不確実性 p(1-p)")
    ax.barh(labels, [r[1] for r in rows], left=[r[2] for r in rows], color="#dc2626",
            label="Jev の誤り (q-p)²")
    ax.invert_yaxis()
    ax.set_xlabel("期待 Brier スコア")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def write_figures(records: list[dict], summary: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        fig_ladder(records, out_dir / "ladder.png"),
        fig_coefficients(summary, out_dir / "coefficients.png"),
        fig_brier(summary, out_dir / "brier.png"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="記事用の図3枚を描く")
    parser.add_argument("--problems", type=Path, default=Path("data/problems.jsonl"))
    parser.add_argument("--responses", type=Path, default=Path("data/responses.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    records = load_records(read_jsonl(args.problems), read_jsonl(args.responses))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    for path in write_figures(records, summary, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
