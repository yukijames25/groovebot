# CLAUDE.md — GrooveBot プロジェクトメモリ

> このファイルと `docs/SYSTEM_SPEC.md` を毎セッションの最初に必ず読むこと。
> 仕様書が正典。本ファイルは作業上の不変条件と段取りの要約。

## プロジェクト一行説明
アカペラ/鼻歌の歌声に、テンポとテンションでリアルタイム同期して「ノる」ロボット。
ミラー型・「楽しい系」。身体（シム/Pepper/NAO/自作機）は差し替え可能な部品。

## 絶対に守る不変条件（破ったらNG）
1. **脳は身体を import しない。** 知覚・生成ロジックは `mujoco`/`pybullet`/NAOqi を直接
   import せず、`RobotBackend` インターフェース越しにのみ身体を扱う。
2. **インターフェースの契約は固定。** `GrooveContext` / `JointCommand` / `SongReference`
   と各 `Protocol`（`ReferenceAligner` / `BeatTracker` / `ArousalEstimator` / `VoiceEncoder` /
   `GrooveGenerator` / `RobotBackend` / `FeedbackRenderer`）のシグネチャを変える時は、
   先に仕様書を更新し理由を残す。
3. **関節出力は送出直前に URDF 可動域へクランプ**（NFR-4）。範囲外指令を絶対に身体へ送らない。
4. **causal/online のみ**（NFR-2）。未来情報を使わない・過去出力を訂正しない。
   ReferenceAligner も online DTW / score following のみ、offline DTW は禁止。
5. **制御ループは固定 30–50 Hz、知覚とは疎結合**（NFR-7, §6）。重い知覚が動作のガタに直結しないこと。
6. **知覚の主軸は `ReferenceAligner`（曲選択 → 参照アライメント、§14）**。
   盲目オンラインビート追跡（`BeatTracker` 系）は未知曲・即興鼻歌向けの**フォールバック**。
   タイミングは「弱い歌声から拍を再抽出する」のではなく「ユーザが選んだ曲の参照拍グリッド」から得る。

## 段取り（一度に1フェーズだけ。デモは常に動く状態を保つ）
- **~~M0~~**: 盲目オンライン拍追跡の評価（**棚上げ**、§10.2 副次。`tools/eval_beat.py` 等は残置）。
- **M0'**: 小規模な参照曲セット作成 ＋ オフラインで歌唱/鼻歌を参照拍グリッドへ整合できるか実現性検証。
- **M1**: メトロノーム＋手付けノリで端到端を通す（**完了**）。
- **M2（必達）**: オンライン `ReferenceAligner` を Orchestrator に接続 ＋ arousal 推定 ＋ 顔/画面フィードバック。
- **M3（目標）**: 拍/小節頭/**曲構造**/arousal/voice-embedding で条件付けした学習 groove 生成。

## 現在地と次の一手
> `/clear` や新セッション後でもこのセクションだけ読めば文脈を復元できる、を目標に維持する。
> 進捗が動いたら都度更新する（古いまま放置しない）。

最新コミット: `5385aa3`（M0-2: Colab numpy<2 ピン）／ タグ: `m0`, `m1`, `m0-1`。

### 完了
- **M1**: リアルタイム groove ループ（`orchestrator` + URDF 可動域クランプ + tests）。`python demo_groove.py` で MuJoCo 上で動く端到端デモ。
- **評価ハーネス**: 合成クリック＋`mir_eval` で F値／CMLt／AMLt／RT-factor（`tools/eval_beat.py`、`--bpm` も `--beats` も受ける）。**新方針では M0' のアライメント精度の主指標として流用する**。
- **公開データ前処理**: `tools/prep_dataset.py`（Demucs で公開データから vocal 分離 ＋ 拍注釈を `--beats` 形式へ）、`experiments/run_gtzan_eval.py`, `notebooks/m0_gtzan_eval.ipynb`（Colab turnkey）。
- **棚上げ**: 上記 Colab パイプラインによる**盲目 BeatNet** の数値化は、madmom が Python 3.10 縛りで Colab/Kaggle に乗らないため棚上げ（コードは削除しない、§10.2 副次・§14.3）。

### 決定（このセッションで方針転換）
- 知覚の主軸を「盲目オンラインビート追跡」から「**参照情報つきオンラインアライメント（曲選択型）**」へ再センタリング。理由: (a) madmom 互換問題で死に筋、(b) カラオケ標的では参照アライメントの方が堅牢で、楽曲構造から先読みのノリも引き出せる。
- 自動曲識別（QBH/旋律照合）は当面**不要**（ユーザが選曲する UX）。

### 次の一手（M0'）
1. 小規模な**参照曲セット**を用意（例: 既存の GTZAN-Rhythm / Isophonics の数曲分を `SongReference` 形式に変換するスクリプトを書く）。
2. **オフラインで** 歌唱/鼻歌を DTW で参照に整合し、復元された拍時刻列を `--beats` 形式に出力。
3. 既存 `tools/eval_beat.py` で F／CMLt／AMLt を測定し、「ユーザ録音が無くてもアライメント精度の感触」を得る。
4. 結果次第で M2 のオンライン化（online DTW / score following 系）へ移行。

### その後
- **M2**（必達）: オンライン `ReferenceAligner` を Orchestrator の Perception に接続 → arousal 推定 → 顔/画面フィードバック。
- **M3**（目標）: 拍/構造/arousal/voice-embedding 条件付けの学習 groove 生成。`SongReference.sections` から「次サビ来る」を予期した先読みのノリ。

不変条件（脳は身体を import しない、causal/online、関節クランプ、固定 30–50 Hz 制御、主軸は ReferenceAligner）は冒頭通りで不変。

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
