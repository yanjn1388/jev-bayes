# jev-bayes 設計書

- 作成: 2026-09-29
- 状態: 設計確定（実装前）
- 背景ソース: My-Knowladge `wiki/sources/2026-09-29_Jevはサイコロを振らない_較正された確率の落とし穴.md`

## 1. 目的

判断特化モデル Jev（TypeSafe AI）が、**計算すれば厳密に求まる確率**を正しく報告できるかを測る。

- 主目的は**発信（ブログのラボノート）と学び**。業務への導入判断は目的に含めない
- 問い: 直感（システム1）に特化した AI は、人間のシステム1と同じ**基準率の無視**を起こすか
- 元記事（林寛太氏）が測ったのは「手がかりなし」と「確率がそのまま書いてある」の2点。本実験はその**中間**（ベイズの定理で計算すれば出る）を埋める

### 成功条件

- 真の確率が分かっている問題で、信頼性図・Brier スコア・ロジット回帰の係数を使って結論が言えること
- 事前登録した仮説 H1〜H3 の成否を、どちらの結果でも報告できること
- 図3枚とデータ・コードが、記事の根拠として公開できる形で残ること

## 2. Jev の呼び出し経路（2026-09-29 調査）

| 項目 | 内容 |
|---|---|
| 経路 | OpenRouter。TypeSafe のアカウントは不要 |
| エンドポイント | `POST https://openrouter.ai/api/v1/systemone`（旧 `/api/alpha/decisions`） |
| モデル | `typesafe/jev-1.13`（別名 `~typesafe/jev-latest`）。**実行時の応答にある `model` を全件記録する**（版の特定） |
| 質問の型 | `choice`（選択肢＋各確率）／`noul`（Yes の確率）／`score`（本実験では使わない） |
| 料金 | 入力 100 万トークンあたり 0.042 ドル・出力無料。応答の `usage.cost` を正とする |
| 上限 | コンテキスト 32K トークン。レート制限・日本語対応は未公表 |
| 認証 | 実験専用の OpenRouter キー（利用上限を設定）。`.env` の `OPENROUTER_API_KEY`（gitignore 対象） |

リクエストと応答の形（TypeSafe 公式ドキュメント `docs.typesafe.ai/primitives/choice`・`/noul` で確認）:

```json
{"model": "typesafe/jev-1.13",
 "state": "...",
 "questions": {
   "q": {"type": "choice", "instructions": "...",
         "criteria": {"yes": "説明", "no": "説明"}}}}
```

- Choice の応答: `answers.q.choice`（最も確率の高い選択肢名）・`answers.q.probabilities`（選択肢名 → 確率。合計 1）・`answers.q.confidence`
- Noul の応答: `answers.q.noul`（Yes の確率）。Noul に confidence は無い
- 質問 ID（`q`）はモデルに送られない。選択肢名と説明はモデルに送られる

> [!note] 疎通確認で確定（2026-09-29、`data/smoke.jsonl`）
> - モデル名は `typesafe/jev-1.13` のままで通る。応答の `model` は `typesafe/jev-1.13-20260917`
> - 応答には `id` と `provider`（`TypeSafe`）も付く。`usage.cost` は1回あたり約 1.5〜1.8×10⁻⁵ ドル
> - 報告確率は 0.01 刻み
> - **同じ入力でも答えが揺れる**（5回で ±0.02 程度。例: `s2-medical-pi0.01-lr20-noul-en` は 0.21〜0.23）。そのため本番は各問を $r=5$ 回呼ぶ

## 3. 実験条件（情報の3段のはしご）

### ① 手がかりなし（元記事の追試・最小限）

公平な抽選の結果を伏せて渡す。入力を区別するため、`state` に試行番号（`Draw #17` 等）を入れる。

| 条件 | 型 | 問うこと | 真の確率 | 件数 |
|---|---|---|---|---|
| 6面サイコロ | choice | どの目が出たか（1〜6） | 各 1/6 | 30 |
| 6面サイコロ | noul | 出目は 1 か | 1/6 | 30 |
| 4分割の回転盤 | noul | 区画 A に止まったか | 1/4 | 30 |
| 20分割の回転盤 | noul | 区画 A に止まったか | 1/20 | 30 |

抽選版の Brier（元記事と同じ比較）のため、各試行の隠れた結果を乱数（シード固定）で生成し `problems.jsonl` に保存する。Jev には渡さない。

### ② 計算すれば出る（本命）

**4題材 × 事前確率 5水準 × 尤度比 5水準 = 100 問**。各問を noul と choice の両方で問う（200 呼び出し）。

- 事前確率 $\pi$: 0.01 / 0.05 / 0.2 / 0.5 / 0.8
- 尤度比 $\mathrm{LR}$: 0.2 / 0.5 / 2 / 5 / 20
- 感度と偽陽性率の決め方
  - $\mathrm{LR}\ge 1$: 感度 0.9、偽陽性率 $0.9/\mathrm{LR}$
  - $\mathrm{LR}<1$: 偽陽性率 0.9、感度 $0.9\times\mathrm{LR}$
- 真の確率（事後確率）: $p=\dfrac{\pi\cdot s}{\pi\cdot s+(1-\pi)\cdot f}$（$s$ 感度、$f$ 偽陽性率）
- 補助の参照値: 条件付き確率の逆転による答え $p_{\text{inv}}=s$（基準率を無視した答えの代表）

| 題材 ID | 事象 | 「陽性」に当たる観測 |
|---|---|---|
| `medical` | その人が病気である | 検査が陽性 |
| `spam` | メールがスパムである | フィルタがスパムと判定 |
| `factory` | 部品が不良品である | 検査装置が不良と判定 |
| `taxi` | 事故のタクシーが青色の会社である | 目撃者が「青」と証言 |

英語のテンプレート（`medical` の例。他の題材は名詞だけを置き換える）:

- state: `In a population, {prior}% of people have condition X. A screening test comes back positive for {sens}% of people who have the condition, and for {fpr}% of people who do not. A randomly selected person from this population tested positive.`
- noul の instructions: `Does this person have condition X? Report the probability that they do. Do not treat the most likely outcome as certain; keep the uncertainty in the probability.`
- choice の選択肢: `has condition X` / `does not have condition X`（instructions は noul と同じ趣旨）

数値の表記は小数点以下を最小限にする（例 0.045 → `4.5`）。

### ③ 確率が書いてある（元記事の追試・最小限）

**4題材 × 記載値 9点 = 36 問**。各問を noul と choice で問う（72 呼び出し）。

- 記載値 $X$: 5 / 15 / 30 / 45 / 50 / 55 / 70 / 85 / 95（%）
- state の例: `A forecasting report states that the probability that this person has condition X is {X}%.`
- 真の確率は $X/100$

### 日本語

②の `medical` 題材 25 問を日本語に訳し、noul で問う（25 呼び出し）。同じ問題の英語版と比べる。

### 規模と費用

| 段 | 呼び出し数 |
|---|---|
| ① | 120 |
| ② | 200 |
| ③ | 72 |
| 日本語 | 25 |
| 合計 | 約 420（＋疎通確認） |

1回 300 トークン程度として費用は 0.01 ドル前後。§8 の疎通確認で同じ入力でも答えが揺れると分かった場合は、各問を $r$ 回呼ぶ（$r$ は費用上限 1 ドルの範囲で決め、PREREGISTRATION に書く）。

## 4. 分析

### 4.1 確率の直接比較と Brier の分解

Jev の報告値 $q$ を真の確率 $p$ と直接比べる（抽選しない）。Brier スコアの期待値は正確に分解できる。

$$
E\left[(q-Y)^2\right]=(q-p)^2+p(1-p)
$$

- 第1項 = Jev の誤り（較正誤差）
- 第2項 = 誰にも減らせない不確実性
- 段ごと・型ごとに両項の平均を出す
- ①のみ、元記事と同じ抽選版（隠れた結果 $Y$ との $(q-Y)^2$ と的中率）も併記する

### 4.2 基準率の無視の判定（②）

ベイズの定理のロジット形 $\mathrm{logit}(p)=\mathrm{logit}(\pi)+\log\mathrm{LR}$ に対し、次の回帰を最小二乗で当てはめる。

$$
\mathrm{logit}(q)=a+b_1\,\mathrm{logit}(\pi)+b_2\,\log\mathrm{LR}
$$

| 結果 | 解釈 |
|---|---|
| $b_1\approx1,\ b_2\approx1$ | ベイズどおり |
| $b_1<1$ | 事前確率を軽く扱う＝基準率の無視 |
| $b_1,b_2$ がともに 1 を大きく超える | 0/1 に倒している |

- $q$ は $[0.001,\,0.999]$ に切り詰めてからロジットを取る。**切り詰めた件数を必ず報告する**
- 信頼区間: 問題単位のブートストラップ（2,000 回、シード固定）で 95% 区間
- 全体の当てはめに加え、題材ごとの当てはめも出す（題材の癖と数値の効果を分けるため）
- 補助: 各問で $q$ が $p$ と $p_{\text{inv}}$ のどちらに近いかを数える

### 4.3 その他の比較

- **③** — $q$ と $X/100$ の平均絶対誤差。ロジスティック曲線 $q=\sigma(k(X-c))$ を当てはめ、傾き $k$ と中心 $c$ を出す
- **Choice と Noul** — 同じ問題で、choice の「事象あり」側の確率と noul の値を対にして比べる
- **日本語** — 同じ問題の $|q_{ja}-q_{en}|$ の平均と、4.2 の係数の差

### 4.4 仮説（本番前に `docs/PREREGISTRATION.md` へ固定する）

| ID | 仮説 | 判定基準 |
|---|---|---|
| H1 | ②の noul は基準率を無視する | $b_1$ の 95% 区間の上端が 1 未満 |
| H2 | choice は noul より 0/1 に倒れる | 対にした $\lvert\mathrm{logit}\,q_{\text{choice}}\rvert-\lvert\mathrm{logit}\,q_{\text{noul}}\rvert$ の平均の 95% 区間が 0 を含まない（正） |
| H3 | ③の choice は 50% 付近で段差を作る | 題材平均で $q(55)-q(45)>0.5$ |

仮説が外れた場合もそのまま報告する。判定基準は結果を見てから変えない。

### 4.5 図（ブログ用・3枚）

1. **はしごの図** — 横軸 $p$、縦軸 $q$。①②③を色分けし対角線を引く。noul と choice は別パネル
2. **係数の図** — $(b_1,b_2)$ 平面にベイズの点 (1, 1) と推定値（95% 区間の楕円または十字）を置く。noul / choice / 日本語を並べる
3. **Brier の内訳** — 段 × 型ごとに「Jev の誤り」と「減らせない不確実性」の積み上げ棒

## 5. リポジトリ構成

```
jev-bayes/
├── pyproject.toml          # uv 管理（httpx / numpy / matplotlib / pytest）
├── .env                    # OPENROUTER_API_KEY（gitignore 対象）
├── docs/
│   ├── SPEC.md             # 本書
│   └── PREREGISTRATION.md  # 仮説と判定基準。本番前に commit する
├── src/jev_bayes/
│   ├── problems.py         # 問題の生成（純関数）
│   ├── client.py           # Jev の呼び出し・再試行・応答の検査
│   ├── run.py              # 実行 CLI（保存済みは飛ばして再開）
│   ├── analyze.py          # 回帰・ブートストラップ・Brier・段差の当てはめ（純関数）
│   └── figures.py          # 図3枚
├── data/
│   ├── problems.jsonl      # 生成した問題（commit する）
│   └── responses.jsonl     # 生の応答（追記のみ・commit する）
├── results/                # 図（png）と summary.json
└── tests/
```

## 6. データ形式

### problems.jsonl（1行1問）

| フィールド | 内容 |
|---|---|
| `id` | 条件から決定的に作る。例 `s2-medical-pi0.05-lr20-noul-en` |
| `stage` | `1` / `2` / `3` |
| `theme` | `die6` / `spin4` / `spin20` / `medical` / `spam` / `factory` / `taxi` |
| `qtype` | `noul` / `choice` |
| `lang` | `en` / `ja` |
| `params` | 段に応じたパラメータ（`prior` `lr` `sens` `fpr` / `stated` / `draw`） |
| `p_true` | 真の確率（choice は「事象あり」側、①の choice は選ばれた目の確率＝1/6） |
| `p_inv` | ②のみ。条件付き確率の逆転による答え |
| `hidden_outcome` | ①のみ。抽選版 Brier 用の隠れた結果 |
| `request` | Jev に送る `state` と `questions` そのもの |

### responses.jsonl（1行1呼び出し・追記のみ）

| フィールド | 内容 |
|---|---|
| `problem_id` | problems.jsonl の `id` |
| `attempt` | 同じ問題を複数回呼ぶ場合の番号（0 始まり） |
| `called_at` | 呼び出し時刻（ISO 8601） |
| `status` | `ok` / `error` / `invalid` |
| `http_status` | HTTP ステータス |
| `raw` | 応答の JSON そのもの |
| `q` | 抽出した報告確率（`ok` のみ） |
| `model` | 応答の `model`（版の特定） |
| `cost` | `usage.cost` |
| `error` | 失敗・不正時の理由 |

分析は常に `raw` から計算し直せるようにする（`q` は便宜上の抽出値）。

## 7. 誤りの扱い

| 事象 | 対応 |
|---|---|
| 429・5xx | 待ち時間を倍々に延ばして最大3回再試行。それでも失敗したら `status=error` で記録して次へ |
| その他の 4xx | 即停止（リクエストの形の誤りで全件を失敗させない） |
| 応答の形が不正（確率が [0,1] 外、choice の確率の合計が 1 から ±0.01 を超えてずれる、期待したフィールドが無い） | `status=invalid` で記録し分析から除外。件数を報告に出す |
| 費用 | `usage.cost` を累計し、1 ドルを超えたら停止。OpenRouter のキー側の上限と二重に守る |
| 中断 | `run.py` は `responses.jsonl` に `ok` がある (`problem_id`, `attempt`) を飛ばすので、再実行で続きから再開できる |

## 8. 実行の順番

1. **疎通確認** — noul と choice を1回ずつ呼び、応答の形を確定して §2 を更新する。同じ入力を5回呼び、答えが毎回同じかを確かめる（同じなら $r=1$）
2. **事前登録** — `docs/PREREGISTRATION.md`（§4.4 の仮説・判定基準、$r$、プロンプトの確定文面）を commit する
3. **本番の実行** — ①②③・日本語
4. **分析と図** — `results/` に出力
5. **記録** — My-Knowladge に `analyses/` として保存。ラボノートの執筆は既存の独立フローで別タスクとして行う

## 9. テスト（TDD・実際の API は呼ばない）

- ベイズの計算が既知の例と一致する（有病率 1%・感度 90%・偽陽性率 9% → 約 0.0917）
- 生成したグリッドの確率がすべて (0, 1) に収まる。感度と偽陽性率が (0, 1] に収まる。ID が重複しない
- 回帰が既知の係数を復元する（$b_1=0.5,\ b_2=1$ で作った合成データから推定値が 0.5 と 1 の近くに出る）
- Brier の恒等式が数値的に成り立つ
- 保存済みの (`problem_id`, `attempt`) を飛ばして再開できる（偽の client）
- 応答の検査が不正な応答を `invalid` にする
- 4xx で停止し、429 で再試行する（偽の HTTP 層）

## 10. 範囲外

- Score 型、Vercel AI Gateway 経路
- 人間の回答データの収集（人間の基準率無視は既存研究の文献値を記事で引くにとどめる）
- 頻度表記（「1000人中10人」）と確率表記の比較（次の実験の候補として残す）
- Jev Router・業務運用への導入判断
- リポジトリの公開（記事を書く段で判断する）
