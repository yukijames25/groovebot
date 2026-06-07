# CLAUDE.md — GrooveBot プロジェクトメモリ

> このファイルと `docs/SYSTEM_SPEC.md` を毎セッションの最初に必ず読むこと。
> 仕様書が正典。本ファイルは作業上の不変条件と段取りの要約。

## プロジェクト一行説明
アカペラ/鼻歌の歌声に、テンポとテンションでリアルタイム同期して「ノる」ロボット。
ミラー型・「楽しい系」。身体（シム/Pepper/NAO/自作機）は差し替え可能な部品。

## 絶対に守る不変条件（破ったらNG）
1. **脳は身体を import しない。** 知覚・生成ロジックは `mujoco`/`pybullet`/NAOqi を直接
   import せず、`RobotBackend` インターフェース越しにのみ身体を扱う。
2. **インターフェースの契約は固定。** `GrooveContext` / `JointCommand` と各 `Protocol`
   （`BeatTracker` / `ArousalEstimator` / `VoiceEncoder` / `GrooveGenerator` /
   `RobotBackend` / `FeedbackRenderer`）のシグネチャを変える時は、先に仕様書を更新し理由を残す。
3. **関節出力は送出直前に URDF 可動域へクランプ**（NFR-4）。範囲外指令を絶対に身体へ送らない。
4. **causal/online のみ**（NFR-2）。未来情報を使わない・過去出力を訂正しない。
5. **制御ループは固定 30–50 Hz、知覚とは疎結合**（NFR-7, §6）。重い知覚が動作のガタに直結しないこと。

## 段取り（一度に1フェーズだけ。デモは常に動く状態を保つ）
- **M0**: 歌声/鼻歌で拍追跡が壊れる箇所を把握（音声のみ）。
- **M1**: メトロノーム＋手付けノリで端到端を通す。
- **M2（必達）**: 声→テンション（B-2）＋画面フィードバック。
- **M3（目標）**: 学習モデル（VQ-VAE＋Transformer）でノリ生成。

## 現在地と次の一手
> `/clear` や新セッション後でもこのセクションだけ読めば文脈を復元できる、を目標に維持する。
> 進捗が動いたら都度更新する（古いまま放置しない）。

最新コミット: `26c6957` ／ タグ: `m0-1`（既存タグ: `m0`, `m1`, `m0-1`）。

### 完了
- **M1**: リアルタイム groove ループ（`orchestrator` + URDF 可動域クランプ + tests）。`python demo_groove.py` で MuJoCo 上で動く端到端デモ。
- **M0**: 拍追跡評価ハーネス。合成クリック＋`mir_eval` で F値／CMLt／AMLt／RT-factor。`tools/eval_beat.py`, `tools/prep_dataset.py`（Demucs で公開データから vocal 分離＋拍注釈を `--beats` 形式へ）、`experiments/run_gtzan_eval.py`, `notebooks/m0_gtzan_eval.ipynb`（Colab turnkey）。
- **M0-1**: 注釈ファイル名マッチの修正（拡張子・大文字小文字・stem 一致）、`PipelineReport` によるステージ別カウント（detected wav / matched annot / demucs ok / beatnet ok / scored）の透明化、`demucs` を `requirements-experiments.txt` に追加、Demucs 出力 vocal パスのフォールバック対応。

### 次の一手（手動・Claude Code 側のタスクではない）
1. `notebooks/m0_gtzan_eval.ipynb` を Colab で開いて実行（GPU は任意）。
   - `gtzan_mini` に音源を同梱済みなので Kaggle 認証は不要。
   - 取得したいもの: GTZAN vocal-stem に対する盲目 BeatNet の **初の実数字**（F値 / CMLt / AMLt / RT-factor）。
   - 実行後は `report.summary()` でステージ別件数を確認（どこで落ちたかが分かる）。

### その後の判断
- 上記の数字を見て、§14（参照情報つきビート追跡）にどれだけ寄せるかを決める。
- 続いて **M2**（arousal 推定 + 画面フィードバック）→ **M3**（学習 groove 生成）。

不変条件（脳は身体を import しない、causal/online、関節クランプ、固定 30–50 Hz 制御）は冒頭通りで不変。

## 現状（既にあるもの）
- `robot/groovebot.urdf` … 10自由度・上半身（身体の契約）
- `groovebot/backend.py` … `RobotBackend` ＋ MuJoCo/PyBullet/RealServo(未実装)
- `groovebot/groove.py` … `GrooveController.compute(beat_pos, energy)`（M1簡易版）
- `demo_groove.py` … 端到端デモ。`python demo_groove.py` で MuJoCo 上で動く
- `docs/SYSTEM_SPEC.md` … 正典

## 進め方のルール
- 小さく刻む。1変更ごとに `demo` とテストを実行して動作確認してから次へ。
- 新しい依存を足す前に必ず一度止めて相談する（重い依存を勝手に入れない）。
- 新モジュールには pytest を書く（最低: 関節クランプのプロパティテスト、
  MuJoCo↔PyBullet の身体非依存スモークテスト）。
- 型ヒント・dataclass・小さなモジュールを徹底。リポジトリ構成は仕様書 §12 に従う。
- `RealServoBackend` は配属後。今は実装しない。

## 環境（Windows 開発）
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   # 最低: mujoco（任意で pybullet）
python demo_groove.py --backend mujoco --bpm 120 --energy 0.85 --seconds 8
pytest
```
ML 系（PyTorch/Demucs/WavLM/AIST++）は M3 に入る時に追加。学習は Kaggle 無料GPU、
本番は研究室GPU（このPCでは回さない想定）。
