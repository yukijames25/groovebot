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

最新コミット: GrooveStyleSelector v2（genre ヘッド本物の数字: full GTZAN 1000曲、fault-filtered+artist-aware と naive stratified の両方を計装、リーク bias 可視化）／ タグ: `m0`, `m1`, `m0-1`。

### 完了
- **M1**: リアルタイム groove ループ（`orchestrator` + URDF 可動域クランプ + tests）。`python demo_groove.py` で MuJoCo 上で動く端到端デモ。
- **評価ハーネス**: 合成クリック＋`mir_eval` で F値／CMLt／AMLt／RT-factor（`tools/eval_beat.py`、`--bpm` も `--beats` も受ける）。**新方針では M0' のアライメント精度の主指標として流用する**。
- **公開データ前処理**: `tools/prep_dataset.py`（Demucs で公開データから vocal 分離 ＋ 拍注釈を `--beats` 形式へ）、`experiments/run_gtzan_eval.py`, `notebooks/m0_gtzan_eval.ipynb`（Colab turnkey）。
- **棚上げ**: 上記 Colab パイプラインによる**盲目 BeatNet** の数値化は、madmom が Python 3.10 縛りで Colab/Kaggle に乗らないため棚上げ（コードは削除しない、§10.2 副次・§14.3）。
- **GrooveStyleSelector v1 縦スライス**（並行トラック、§14 モジュール節）: `groovebot/style/`（features=log-mel／model=小CNN genre+mood マルチヘッド／attributes=tempo+arousal ヒューリスティック／table=Yuki のノり方表ソフト mood 重み／select=起動窓→GrooveStyle）。`experiments/train_style.py` で GTZAN-mini で genre 学習、mood は MTG-Jamendo 待ちの決定的 stub。pytest 32 件、出力はテキストラベルのみ（JointCommand 橋渡しは後段）。著作権 J-pop は不使用、PyTorch CPU で完結。
- **GrooveStyleSelector v2 genre ヘッド本物の数字**（2026-06-14、full GTZAN 1000曲、CPU）: `tools/gtzan_split.py` で fault-filtered+artist-aware split (jongpillee/music_dataset_split、Kereliuk 2015／Sturm 2013 出典) と naive stratified split を計装。`groovebot/style/augment.py` で SpecAugment + random time crop、`StyleCNN(dropout=0.3)` + val-acc early stopping (patience=8)。`jazz/jazz.00054.wav` のみ破損で除外ログ。**fault split: 最良 val 0.498 @ ep 12、test 0.431、train-val gap +0.222**。**naive split: 最良 val 0.620 @ ep 28、test 0.573、train-val gap +0.241**。naive−fault = **+12.2 pp val / +14.2 pp test** がリーク bias（GTZAN の既知 fault が稼ぐ楽観バイアス）。混同行列: fault 側は pop が sink クラス（country/hiphop/jazz/reggae/rock の自信ある誤分類を吸う）、classical のみ綺麗に分離。mood ヘッドは `--mood-weight 0.0` で勾配ゼロにしたまま wired のみ（MTG-Jamendo 待ち）。pytest 225 passed / 2 skipped。

### 決定（このセッションで方針転換）
- 知覚の主軸を「盲目オンラインビート追跡」から「**参照情報つきオンラインアライメント（曲選択型）**」へ再センタリング。理由: (a) madmom 互換問題で死に筋、(b) カラオケ標的では参照アライメントの方が堅牢で、楽曲構造から先読みのノリも引き出せる。
- 自動曲識別（QBH/旋律照合）は当面**不要**（ユーザが選曲する UX）。

### 次の一手（M0'）
1. 小規模な**参照曲セット**を用意（例: 既存の GTZAN-Rhythm / Isophonics の数曲分を `SongReference` 形式に変換するスクリプトを書く）。
2. **オフラインで** 歌唱/鼻歌を DTW で参照に整合し、復元された拍時刻列を `--beats` 形式に出力。
3. 既存 `tools/eval_beat.py` で F／CMLt／AMLt を測定し、「ユーザ録音が無くてもアライメント精度の感触」を得る。
4. 結果次第で M2 のオンライン化（online DTW / score following 系）へ移行。

### M0' Tier 2 DAMP-S-AG（MIDI 参照、Amazing Grace） 初の実数字（2026-06-10）
20 rendition × 2 経路 = 40 行。raw `data/amazing_grace.tar.gz`（18 GB、Smule ライセンス）から
`tools.ingest_damp damp-s-ag --max-n 100` で先頭 100 件を `data/m0p_t2_damp/` に展開、
うち先頭 20 件で `--reference-source midi` を走らせた（pyin が1件 ~78 秒で支配的なため、100 件は ~2.6 時間 = 後段）。

- **chroma 経路** (n=20): F=0.298, CMLt=0.452, AMLt=0.484, RT=0.029
- **pitch 経路** (n=20): F=0.199, CMLt=0.212, AMLt=0.225, RT=0.019

所見:
- Tier 1（~F=0.98）から大きく落ちる。理由: Amazing Grace は rubato 強い賛美歌で
  別演者間のテンポ揺らぎが大きい、加えてアマチュア歌唱の音程/タイミング揺らぎ。
- **chroma > pitch**: 弱い旋律でも chroma_cqt は formant 系のエネルギーから旋律寄りの
  時間変化を拾えるのに対し、pyin は素人録音の breath/環境ノイズに弱い。
- **CMLt ≈ AMLt**（両経路で差ほぼ無し）→ DTW は metric-level 取り違えは起こさない（構造的強み維持）。
- RT-factor ≈ 0.02-0.03（オフライン、超高速）。
- ばらつき大: chroma F は 0.07-0.52 と rendition 依存。歌唱品質との相関は次の集計で見る。

`data/m0p_t2_damp_work/` に CSV 4 種。MIDI 参照ルートは ffmpeg 不在でも動くことを確認。

### M0' Tier 2 DAMP-S-AG レバー実験（2026-06-11、20件）
診断（前セッション）で「低スコアの大半は方法論アーティファクト」という仮説を立て、
2つのレバーを実装して 20件で寄与を切り分けた。

実装した opt-in 経路（default off で IF 後方互換）:
- `OfflineDTWAligner.subseq` ＋ `groovebot.align.features.trim_silence` → `--dtw-subseq` / `--silence-trim`
- `groovebot.align.features.pitch_contour_feature`（2次元: キー正規化セミトーン + voicing channel）
  ＋ `MidiReference.pitch_contour` → `--pitch-mode {one-hot, continuous}`

20件の F-measure 結果（同一サブセット、`subset20`）:

| 条件 | F_chroma | F_pitch | CMLt_pitch | AMLt_pitch |
|---|---:|---:|---:|---:|
| Baseline | 0.298 | 0.199 | 0.212 | 0.225 |
| Lever 1 のみ (subseq + trim) | **0.062** ↓↓ | 0.076 ↓↓ | 0.064 | 0.079 |
| Lever 2 のみ (continuous pitch) | 0.298 = | 0.204 ≈ | 0.167 ↓ | 0.172 ↓ |
| Lever 1 + Lever 2 | 0.062 ↓↓ | 0.116 ↓ | 0.078 | 0.089 |

**判定**: 仮説に反して Lever 1 は両経路で大幅悪化、Lever 2 は ほぼ neutral。

**失敗の理由（事後解析）**:
- subseq: query が 70-85% 有声 vs MIDI 99% アクティブの非対称下では、DTW が「コスト最小の
  小さなマッチ領域」に縮退し、`map_reference_beats` で大半の MIDI 拍がドロップされる典型パターン。
  Amazing Grace のように reference 全長をクエリ全長で覆う前提のタスクには subseq の境界スラックが過剰。
- silence_trim: 診断で既に判明していた通り、先頭/末尾無音は full DTW が境界スラックとして
  productively 使っており、剥がすと逆に micro-misalignment を吸収できなくなる。
- 連続セミトーン: pitch 経路の F は微増（+0.005）したが CMLt/AMLt が下がり、ネット neutral。
  octave-folded one-hot の理論的問題はあるが、それは Amazing Grace の支配的な誤差要因ではない模様。

**コードは残置**（後方互換の opt-in、テスト緑）。本番ルートでは引き続き baseline（subseq=False、
silence_trim=False、pitch_mode=one-hot）を default として使う。

**真の天井に近い候補**（未実施）:
1. **per-rendition origin calibration**: 診断で LOW が +0.15s offset で F 0.07→0.41 に跳ねた。
   DTW 後の小範囲 offset 掃引で「ピーク offset を採用」する後処理。
2. **テンポ事前推定**: rendition vocal に `librosa.beat.tempo` をかけ、reference を伸縮してから DTW
   （rubato 賛美歌の最大の誤差源を緩和）。
3. **Sakoe-Chiba band**: `librosa.sequence.dtw(band_rad=0.1)` 等で warp の最大斜度を制約し、
   subseq の縮退と対角逸脱を同時に防ぐ。
4. **データ多様性**: 賛美歌1曲（Amazing Grace）の代表性は限定的。次は DAMP-VSEP（多曲、テンポ安定）で
   同じ計装が baseline で出す数字を見たほうが本来の答えに近い。

**次の手動アクション候補**: 上記のうち (1) か (3) を 20件で試すのが最小ステップ。

### M0' Tier 2 DAMP-S-AG Lever A/B 実験（2026-06-13、20件）
上記「真の天井候補」の (3) Sakoe-Chiba band を **Lever A**、(1) per-rendition origin calibration を
**Lever B** として実装し、同じ subset20 で再測定。実装は **public IF 不変**、新パラメータは
default 無効の opt-in。

**実装した経路**:
- `OfflineDTWAligner.band_rad` → librosa.sequence.dtw に `global_constraints=True, band_rad=value`。
  full DTW (subseq=False) を維持しつつ warp の対角逸脱を制限。CLI: `--band-rad`。
- `groovebot.align.origin.estimate_origin_offset` → query の `librosa.onset.onset_strength` と、
  MIDI **note-on** 時刻列（`MidiReference.note_onsets`、GT である `MidiReference.beats` とは別）から
  合成した onset 包絡を相互相関し、±max_lag 内のピーク lag を返す。
  推定 lag を recovered beat から差し引いてから採点。CLI: `--origin-anchor`。
- 補助: `experiments/run_m0p_t2_damp.py --max-renditions N` で先頭 N 件に限定。
  `tools/_eval_levers.py` は throwaway の多コンフィグ掃引（pyin を rendition ごとに1回キャッシュ）。

**★絶対規則（記録）**: lag 推定は **GT (`midi_ref.beats`) を一切参照しない**（テストリーク防止）。
xcorr の参照側は `midi_ref.note_onsets`（audible attack 時刻、score の metric grid とは別量）。

20件 F-measure（同一 `subset20`、`band_rad=0.10`、`max_lag_sec=2.0`）:

| 条件 | F_chroma | F_pitch | CMLt_pitch | AMLt_pitch | lag_med [s] |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.298 | 0.199 | 0.212 | 0.225 | 0.000 |
| Lever A のみ (band) | 0.298 = | 0.196 ≈ | 0.219 | 0.230 | 0.000 |
| Lever A + Lever 2 (continuous pitch) | 0.298 = | 0.206 ↑ | 0.170 ↓ | 0.175 ↓ | 0.000 |
| Lever A + Lever B (band + anchor) | **0.324** ↑ | 0.210 ↑ | 0.228 | 0.239 | 0.046 |
| Lever B のみ (anchor) | **0.324** ↑ | **0.213** ↑ | 0.221 | 0.233 | 0.046 |

**判定**:
- **Lever A は両経路で neutral**。`band_rad=0.10`（≈±19s 相当）は診断で見た 12-19s の対角逸脱を
  そのまま許容してしまうため拘束として効かない。逆により狭い band は full DTW が無音区間を
  境界スラックとして使う余地を奪う（前回の trim 失敗と同じ機序）ので安全側に倒れた。
- **Lever B が両経路で +0.014 〜 +0.026 の小幅な実利得**。chroma F: 0.298 → 0.324 (+8.7% rel)、
  pitch F: 0.199 → 0.213 (+7.0% rel)。median 推定 lag = 0.046s。
- **Lever A + Lever B は Lever B 単独と同水準**（chroma 同値、pitch は微減 0.213→0.210）。
  band は依然 neutral、ゲインは Lever B が独占。
- **Lever 2（連続セミトーン）+ Lever A**: pitch F 微増 (+0.007) だが CMLt/AMLt が顕著に低下
  （0.212→0.170, 0.225→0.175）。前回（Lever 2 のみ）と同じ「F が拍数で稼げても metric-level
  整合が悪化」のパターン。本番では引き続き one-hot を default。

**コードは残置**（後方互換の opt-in、テスト 173 passed / 2 skipped）。Lever B は **default off**
だが、DAMP-S-AG MIDI ルートでは `--origin-anchor` を付けるのが推奨。

**所見**:
- 診断（LOW: +0.15s で F 0.07→0.41）が示唆した「per-rendition 系統オフセット」は実在するが、
  median 0.046s（mean ~0.05s 程度）と思ったより小さい。20件の半分以上は |lag| < 0.05s で
  本質的ゲインなし、一部の高 |lag| rendition だけが救われている形と推定。
- 「真の天井」は band ではなく **anchor + 何か別の手** にある。次の梃子候補:
  - **テンポ事前推定**（候補 2）: rubato による per-rendition テンポ揺らぎは band では取れない。
    `librosa.beat.tempo` で rendition のテンポを推定し、reference を伸縮してから DTW。
  - **データ多様性**（候補 4）: 賛美歌1曲では母集団効果が読めない。DAMP-VSEP（多曲、テンポ安定）
    で同計装を走らせると baseline / Lever B のゲインがそのまま乗るかが見える。
  - **band_rad のチューニング**: 0.10 は広すぎた。0.02-0.05 を試す価値はあるが、上記2つの
    後にすべき（band 単体の上限は元々低い）。

**次の手動アクション候補**: 候補 2（テンポ事前推定）を 20件で試す → だめなら候補 4 (DAMP-VSEP)。

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
