# jev-bayes

判断特化モデル [Jev](https://typesafe.ai/)（TypeSafe AI）に、真の確率がベイズの定理で厳密に求まる問題を渡し、返ってくる確率の較正と「基準率の無視」を測った実験のコードとデータです。

> **English summary.** We asked Jev (TypeSafe AI's decision model, `typesafe/jev-1.13-20260917` via OpenRouter) 417 probability questions × 5 attempts whose true answers are known exactly (hidden fair draws, Bayes-rule screening problems, and stated probabilities). On Bayes problems Jev did **not** show human-style base-rate neglect: the weights on prior log-odds (0.59) and on the log likelihood ratio (0.53) were both below 1 and statistically indistinguishable, i.e. a uniform shrinkage of log-odds toward 0.5. It did reproduce the "80% confidence on a hidden fair die" finding (mean 0.82). Hypotheses were preregistered before the main run (`docs/PREREGISTRATION.md`).

- 解説記事（ブログ・ラボノート）: 「無視していたのではなく、縮めていた」（ゆるメトリクスラボ）
- 出発点: 林寛太氏「[Jevはサイコロを振らない｜「較正された確率」の意外な落とし穴](https://note.com/kantahayashiai/n/n4c54eed30787)」（本実験の ① はこの記事の追試）

## 結果の要約

事前登録した仮説（`docs/PREREGISTRATION.md`）の判定:

| ID | 仮説 | 結果 | 判定 |
|---|---|---|---|
| H1 | ベイズ問題（Noul・英語）で基準率を無視する（事前確率の重み $b_1$ の 95% 区間の上端が 1 未満） | $b_1=0.59$、区間 [0.54, 0.64] | 成立 |
| H2 | Choice は Noul より 0/1 に倒れる | 対数オッズの絶対値の差の平均 0.53、区間 [0.40, 0.66] | 成立 |
| H3 | 確率が書いてある文書で、Choice は 50% 付近に段差を作る（$q(55)-q(45)>0.5$） | 0.489 | 不成立 |

**H1 の読み方に注意。** 判定基準の上では成立したが、証拠（尤度比）の重み $b_2=0.53$（区間 [0.48, 0.60]）も同じだけ 1 を下回っている。$b_2-b_1=-0.06$（区間 [−0.12, 0.003]）で 0 を含む。観察されたのは「基準率だけを軽く扱う」人間型の無視ではなく、対数オッズ全体を 0.5 の側へ縮める一様な縮みである。事前登録の基準はこの2つを区別できなかった（$b_2-b_1$ は事後の探索として `results/summary.json` に出力）。

詳細な数値はすべて `results/summary.json`、図は `results/*.png`。

## 実験の構成

| 段 | 入力 | 問題数 |
|---|---|---|
| ① 手がかりなし | 結果を伏せた公平なサイコロ・4分割と20分割の回転盤 | 120 |
| ② 計算すれば出る | 4題材（医療検査・スパム判定・工場の不良品・タクシー問題）× 事前確率 5 水準 × 尤度比 5 水準 × Noul/Choice | 200 |
| ③ 書いてある | 「確率は X% と報告されている」× 記載値 9 点 × Noul/Choice | 72 |
| 日本語 | ② の医療検査を日本語に訳して Noul で | 25 |

同じ入力でも答えが揺れる（±0.02 程度）ため、各問を 5 回呼んで平均した（計 2,085 回・費用 0.032 ドル）。設計の詳細は `docs/SPEC.md`、実装計画は `docs/PLAN.md`。

## ディレクトリ

```
docs/SPEC.md             設計書
docs/PLAN.md             実装計画
docs/PREREGISTRATION.md  事前登録（仮説・判定基準・固定した版）
src/jev_bayes/           問題生成・Jev クライアント・実行 CLI・集計・図
tests/                   pytest（実際の API は呼ばない）
data/problems.jsonl      生成した問題（Jev に送ったリクエストそのものを含む）
data/responses.jsonl     本番の生の応答（追記のみ）
data/smoke.jsonl         疎通確認の応答（集計には使わない）
results/                 summary.json と図3枚
```

## 再現する

```bash
uv sync
uv run pytest -q                                   # 55 件

# 集計と図は、同梱のデータからそのまま再計算できる（API キー不要）
uv run python -m jev_bayes.analyze
uv run python -m jev_bayes.figures

# Jev を呼び直す場合は OpenRouter のキーが要る
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env
uv run python -m jev_bayes.problems                # data/problems.jsonl を再生成（決定的）
uv run python -m jev_bayes.run --attempts 5 --out data/responses-new.jsonl
```

Jev の答えは同じ入力でも揺れ、モデルの版が変われば結果も変わりうる。本データは `typesafe/jev-1.13-20260917` のもの。

## 事前登録と時系列

| 時刻（JST, 2026-09-29） | 出来事 |
|---|---|
| 12:17:34 | 疎通確認の結果を commit |
| 12:18:27 | 事前登録を commit（`docs/PREREGISTRATION.md`） |
| 12:18:32 | 本番の最初の呼び出し（`data/responses.jsonl` の `called_at`） |
| 12:26:19 | 本番の最後の呼び出し |

git の日時は作成者が自己申告する値であり、改ざんを防ぐ証明ではない。問題ファイルは SHA-256（`f79b072d…`）で固定している。

## 履歴の書き換えについて

公開前（2026-09-29）に、コミットの作者メールアドレスを GitHub の非公開用アドレスに置き換えた（`git filter-repo --mailmap`）。内容と日時は変えていないが、コミット ID はすべて変わった。`docs/PREREGISTRATION.md` に書かれた旧 ID は次のとおり対応する。

| 旧 ID | 新 ID | コミット |
|---|---|---|
| `e9b774f` | `6cfe60c` | 図を追加（事前登録の直前の HEAD＝固定した分析コード） |
| `fabf907` | `e6e3a89` | 事前登録 |
| `1278921` | `966c058` | 本番の応答と集計結果 |

## ライセンス

- コード（`src/` `tests/` と設定ファイル）: MIT License（`LICENSE`）
- データと図（`data/` `results/`）、文書（`docs/`・本 README）: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja)

Jev は TypeSafe AI の製品であり、本リポジトリは同社と無関係の個人による検証である。
