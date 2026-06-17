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

最新コミット: `a02e070`（JointCommand 橋渡し v1.1 — `groovebot/style/narrate.py` で window summary + per-beat trace、`render_groove` に `--narrate`/`--verbose`/`--narrate-max-beats`、本番出力は `JointCommand` のまま不変、公開 IF 不変、新内部状態ゼロ、リアルタイム制御ループ外、pytest 421 passed / 2 skipped）／ タグ: `m0`, `m1`, `m0-1`。

### 完了
- **M1**: リアルタイム groove ループ（`orchestrator` + URDF 可動域クランプ + tests）。`python demo_groove.py` で MuJoCo 上で動く端到端デモ。
- **評価ハーネス**: 合成クリック＋`mir_eval` で F値／CMLt／AMLt／RT-factor（`tools/eval_beat.py`、`--bpm` も `--beats` も受ける）。**新方針では M0' のアライメント精度の主指標として流用する**。
- **公開データ前処理**: `tools/prep_dataset.py`（Demucs で公開データから vocal 分離 ＋ 拍注釈を `--beats` 形式へ）、`experiments/run_gtzan_eval.py`, `notebooks/m0_gtzan_eval.ipynb`（Colab turnkey）。
- **棚上げ**: 上記 Colab パイプラインによる**盲目 BeatNet** の数値化は、madmom が Python 3.10 縛りで Colab/Kaggle に乗らないため棚上げ（コードは削除しない、§10.2 副次・§14.3）。
- **GrooveStyleSelector v1 縦スライス**（並行トラック、§14 モジュール節）: `groovebot/style/`（features=log-mel／model=小CNN genre+mood マルチヘッド／attributes=tempo+arousal ヒューリスティック／table=Yuki のノり方表ソフト mood 重み／select=起動窓→GrooveStyle）。`experiments/train_style.py` で GTZAN-mini で genre 学習、mood は MTG-Jamendo 待ちの決定的 stub。pytest 32 件、出力はテキストラベルのみ（JointCommand 橋渡しは後段）。著作権 J-pop は不使用、PyTorch CPU で完結。
- **GrooveStyleSelector v2 genre ヘッド本物の数字**（2026-06-14、full GTZAN 1000曲、CPU）: `tools/gtzan_split.py` で fault-filtered+artist-aware split (jongpillee/music_dataset_split、Kereliuk 2015／Sturm 2013 出典) と naive stratified split を計装。`groovebot/style/augment.py` で SpecAugment + random time crop、`StyleCNN(dropout=0.3)` + val-acc early stopping (patience=8)。`jazz/jazz.00054.wav` のみ破損で除外ログ。**fault split: 最良 val 0.498 @ ep 12、test 0.431、train-val gap +0.222**。**naive split: 最良 val 0.620 @ ep 28、test 0.573、train-val gap +0.241**。naive−fault = **+12.2 pp val / +14.2 pp test** がリーク bias（GTZAN の既知 fault が稼ぐ楽観バイアス）。混同行列: fault 側は pop が sink クラス（country/hiphop/jazz/reggae/rock の自信ある誤分類を吸う）、classical のみ綺麗に分離。mood ヘッドは `--mood-weight 0.0` で勾配ゼロにしたまま wired のみ（MTG-Jamendo 待ち）。pytest 225 passed / 2 skipped。
- **GrooveStyleSelector v3 転移学習＋mood 実装**（2026-06-14、§14 v3 仕様節）: `groovebot/style/backbone.py` で **PANNs CNN14**（Kong 2020、AudioSet 事前学習、Zenodo 3987831、340 MB ckpt→`data/raw/`）を凍結ラップ、`.npy` キャッシュ（`data/style_emb_*/`、gitignore）で 1 曲 1 度だけ embed。`groovebot/style/model.py` に `StyleHead`（2048d 埋め込み→256 隠れ→genre/mood マルチヘッド、dropout 0.3）。`select.py` は v1/v2 (StyleCNN) と v3 (backbone+head) を分岐、`GrooveStyleSelector.from_panns(ckpt, head_weights)` で v3 構築。public 出力契約 `GrooveStyle` は不変。**genre fault test: 0.431 → 0.817 (+38.6 pp)、val gap +0.222 → +0.026** — 凍結バックボーン×小ヘッドで過学習が消える。混同行列の **pop sink が解消**（v2: pop 1.00 だが他クラス収束 → v3: pop 0.77 で対角支配、classical 1.00、jazz 1.00、disco 0.90、hiphop 0.85、metal 0.78、reggae 0.73、rock 0.72、country 0.73、blues 0.71）。precompute 285s/1000曲、ヘッド学習 0.1s/epoch。mood は `groovebot/style/mood_mapping.py`（MTG 59 タグ→6 クラス、38 mapped/18 theme drop/3 ambiguous drop、`drop_on_disagreement` default）＋`tools/ingest_mtg_moodtheme.py`（TSV→manifest CSV）＋`experiments/train_mood_tl.py`（artist 非重複 split、`--synthetic-stub` で配線確認、実 MTG は upstream `download.py --from --to` で有界 DL）。pytest 257 passed / 2 skipped（v3 で +32 件）。著作権 J-pop 不使用、PANNs ckpt/MTG audio とも gitignore。JointCommand 橋渡しは引き続き後段。
- **GrooveStyleSelector v3 mood 本物の数字**（2026-06-15、MTG-Jamendo audio-low 6 archives、~3.1 GB ローカル、artist 非重複 split、CPU）: 直接 curl で archive 00-05 を取得（MTG `download.py` には範囲フラグ無し → README を修正）。`tools/ingest_mtg_moodtheme.py` に `_resolve_audio_path()` 追加（audio-low の `.low.mp3` infix 解決）。**780 mood-kept clips（aggressive 105 / happy 128 / sad 125 / calm 200 / dark 130 / epic 92、min/max=2.2×）**。**conflict rule は MTG コーパス全体で moot**: 18,486 行のうち 34% が theme/ambiguous のみ、66% が単一 mood タグ、**multi-mood-tag 0 件**。`drop_on_disagreement` と `first_match` の manifest は **md5 一致**。学習結果: drop test **0.350** (val 0.359 @ ep 2, gap +0.059)、first_match test 0.333 (val 0.342 @ ep 3, gap +0.109)、差 ±0.017 は確率的ノイズ。**calm が sink クラス**（sad→calm 62%, aggressive→calm 44%, dark→calm 43%, epic→calm 39%; calm 自身は 70% 正答）。**calm↔sad は非対称**（sad→calm 62%, calm→sad 13%）。**aggressive↔epic は想定より小**（aggressive→epic 0%, epic→aggressive 8% — aggressive 誤分類は calm/dark に流れる）。epic は 13 test clips で 46% 正答。pytest 261 passed / 2 skipped（ingest 用に +4 件）。著作権・MTG 規約遵守、audio は data/raw/ gitignore、ローカル限定。
- **GrooveStyleSelector v3 arousal/valence 本物の数字**（2026-06-15、DEAM 1802 曲、CPU、§14 v3 arousal 仕様節）: DEAM(MediaEval)を公式 cvml.unige.ch から直接 curl(`DEAM_audio.zip` 1.25 GB + `DEAM_Annotations.zip` 4.5 MB)、`data/raw/deam/`(gitignore)。Kaggle 経路は認証回避で不採用。`groovebot/style/deam.py` に CSV ペアローダ(leading-space 列名正規化、MEMD_audio/flat 両レイアウト対応) ＋ song-disjoint split ＋ `sam_to_unit` 較正(SAM 1..9 → 0..1)。`groovebot/style/model.py::StyleRegressionHead`(凍結 PANNs 2048d → 256 隠れ → arousal/valence 個別 Linear、dropout 0.3、既存 `StyleHead` と同 ModuleDict パターン)。`experiments/train_arousal_tl.py` で MSE 学習 ＋ 回帰指標(R²/RMSE/Pearson r) ＋ 既存ヒューリスティック arousal vs DEAM 真値の相関測定を 1 run で。embedding precompute 0.37-0.39s/clip × 1802 ≈ 12分(.npy キャッシュ、`data/style_emb_deam/`、gitignore)、ヘッド学習 0.1s/epoch、early stop @ ep 26 patience 15。**test (n=270): arousal R²=0.522 RMSE=0.886(SAM) r=0.723 / valence R²=0.451 RMSE=0.875 r=0.675**。文献目安(arousal R²≈0.6 / valence≈0.4)に対し arousal は -0.08、valence は +0.05。**ヒューリスティック vs DEAM 真値 r=0.422**(RMS×onset 密度は方向性正しいが弱、説明分散率 ~18%)、学習ヘッドは r=0.723(~52%)で **+34.4 pp の R² ゲイン**、+0.30 の r ゲイン。判定: ヒューリスティックは default-fast パスとして残し、learned arousal_fn を opt-in の上位経路に。`groovebot/style/select.py` に `arousal_fn: (audio, sr) -> 0..1` 引数(後方互換、デフォルトはヒューリスティック)、`make_panns_arousal_fn(backbone, head, target='arousal', calibrator=sam_to_unit)` で v3 配線。`GrooveStyle` public 出力契約は不変。pytest 293 passed / 2 skipped(+32 件: deam loader/regression head/trainer stub/select arousal_fn)。DEAM 規約(CC BY-NC)遵守、audio/embedding とも gitignore。JointCommand 橋渡しは引き続き後段。
- **JointCommand 橋渡し v1.1 ナレーション層**（2026-06-17、§14 v1.1 narrate 仕様節）: v1 の本番出力は `JointCommand` のまま不変、読み取り専用の観測層 `groovebot/style/narrate.py` を追加。`format_window_summary(style, *, t_start, t_end|seconds, rate, reason)` で「[t_start-t_end s, NN Hz] 知覚: genre/mood/arousal/tempo → 判断: GrooveStyle=move@intensity (reason) → 動作: descriptor/サイクル/主関節/SOFT_AMP」の 3 行段落、`format_beat_trace(style, *, commands, rate, bpm, max_beats)` で per-tick `JointCommand.targets`(`render_groove` が CSV へ落としている同じ値) を拍 index にバケットし主関節ピーク角 ＋ 拍内位相 ＋ on-beat/off-beat ラベルを 1 拍 1 行、`narrate(...)` で window+optional beat trace。静的表 `PRIMARY_JOINTS` / `CYCLE_BEATS` / `MOVE_DESCRIPTOR` は 8 ムーブ網羅、`MOVE_PRIMITIVES[move](0.25, 1.0)` の実出力との一致をテストで固定(語彙ドリフト検出)。`render_groove.py` に `--narrate` / `--verbose` / `--narrate-max-beats` 追加、`--skip-render --narrate` で「音声→ナレーションのみ(GIF 無し)」の内部検証パス。設計の決まり: (1) 公開 IF 不変、(2) 新内部状態ゼロ(全て既存の `GrooveStyle` と CSV 由来の `commands` から導出)、(3) リアルタイム制御ループからは呼ばない(NFR-7 予算保護)、(4) on-beat/off-beat ラベルは宣言的で headbang の b+0.5 ピーク=off-beat を honest に報告(narration 側で歪めない)。pytest **421 passed / 2 skipped**(+24 narrate / +1 render plumbing smoke)。正直な限界: 記述は手作りの `MOVE_DESCRIPTOR` / `PRIMARY_JOINTS` 表で、プリミティブを足す時は narrate 表も同時更新が必要(`test_narration_tables_cover_full_move_vocabulary` で抜け検出)。
- **JointCommand 橋渡し v1**（2026-06-17、§14 v1 bridge 仕様節）: 検証済みスタイル属性(`GrooveStyle`)を関節動作に落とす最初のリング。`groovebot/groove_style.py` に `StyleGrooveGenerator(GrooveGenerator)` ＋ プリミティブ表 `MOVE_PRIMITIVES`(8 種、`table.MOVES` と網羅一致: headbang / bob_nod / sway / rock / fist_pump / clap / penlight_wave / quiet_listen) ＋ `SOFT_AMP`(URDF 内側に取ったソフト上限) ＋ `metronome_from_style(style)`(`style.tempo_bpm` 駆動の `MetronomePerception` ビルダ、退化 BPM は `floor_bpm=40` でループ凍結回避)。全プリミティブ全項に `intensity` を掛ける契約で `intensity=0` は必ず neutral pose に戻る。返すのは触る関節だけ、generator が `neutral_pose()` で残りを埋めるので毎 tick `JointCommand.targets` は `JOINT_NAMES` 完全網羅。URDF クランプは二重ガード(`SOFT_AMP` 設計上の天井 + Orchestrator `clamp_command` の NFR-4 ハード安全)。配線: `GrooveStyleSelector` → `set_style(style)`(秒オーダ) → Orchestrator(30-50 Hz) → `generate(ctx)` → `JointCommand` → MuJoCo。`GrooveContext` 公開 IF は不変、`use_ctx_arousal=True` opt-in で `ctx.arousal` 振幅変調(M2 用)。`experiments/render_groove.py` で `--all-moves`(全プリミティブ別 GIF) / `--style MOVE` / `--audio PATH`(セレクタ経由) / `--skip-render`(GL 無し環境) の開発ツール、`data/renders/<tag>.{csv,gif}` 出力(PIL ベース GIF、CSV は常時)。pytest **396 passed / 2 skipped**(+52 件: groove_style 49 / render_groove 3)。正直な限界: sim ≠ 実サーボ(MujocoBackend は `smoothing=0.35` の一階低域通過のみ、PD/トルク/整定は別物) / 拍は BPM メトロノーム駆動で実歌唱タイミング未追従(M2 で `ReferenceAligner` 差替) / プリミティブは閉形式手作り(学習動作は M3) / render_groove は ckpt 未指定だとセレクタヘッドはランダム重み(出力は legal だが genre/mood は意味なし、視覚化用途)。
- **GrooveStyleSelector v3.1 affect 統合**（2026-06-16、§14 v3.1 仕様節）: v3 で学習済みだった DEAM arousal/valence をデフォルトパスへ昇格。`groovebot/style/select.py` に `regression_head: StyleRegressionHead | None` を追加 — backbone+head と一緒に渡すと、PANNs embed を 1 回計算して **分類ヘッドと回帰ヘッドで共有**(M2 realtime ループの予算保護、`test_embedding_shared_across_head_and_regression_head` で固定)、`arousal_fn`/`valence_fn` 未指定なら DEAM-learned を自動で既定に。明示 `arousal_fn`/`valence_fn` は常にオーバライド。ヒューリスティック(r=0.42)は PANNs 無し / lean ループの fallback として残す。`groovebot/style/mood_from_va.py`(新規): circumplex (Russell 1980) で V/A 0..1 平面の **4 象限 → mood 写像**(happy=(1,1) / aggressive=(0,1) / calm=(1,0) / sad=(0,0))、Gaussian soft membership(sigma=0.45)、argmax 回避で table.select_move の mood-soft 入力にそのまま流せる。**epic/dark は純 V/A 座標ではない**(epic は valence 曖昧、dark は sad と重複) → 既定では確率 0、`PROTOTYPES_WITH_AUX` 草案(`epic=(V=0.55, A=0.95, w=0.5)`, `dark=(V=0.10, A=0.30, w=0.5)`、weight 0.5 で corner mass を盗まない設計)で opt-in。`mood_source: Literal["head","va"]` 切替(既定 "head" = MTG-trained head 維持) — VA に切替えると MTG head はそのまま残し backbone embed を流し続ける(genre_probs のため)ので切替コストゼロ。`experiments/compare_va_mood_vs_mtg.py`(新規): 同じ clip 集合で両 pipeline を流し、agreement_rate / 6×6 confusion / `per_class_profile_by_mtg_mood`(MTG 各クラスについて DEAM (a,v) の mean/std — **MTG 退役診断の本体**) / `calm_sad_stability`(v3 で見えた sad→calm 62% 崩壊が VA で解けるか) / `accuracy_vs_gt`(両 pipeline)。`--synthetic-stub` で wiring 確認のみ可能。MTG mood ヘッド退役判定: (1) V/A の `accuracy_vs_gt` ≥ MTG、(2) 手付け subset の `calm_sad_stability.agreement_rate` ≥ 0.8、(3) listening test 合格、の 3 条件。pytest **344 passed / 2 skipped**(+51 件: mood_from_va 25 / select_affect_default 16 / compare_va_mood_vs_mtg 11)。`GrooveStyle` public 出力契約は不変。CI は PANNs backbone 全モック(340 MB ckpt 不要)。

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
