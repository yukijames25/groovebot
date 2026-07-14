# GrooveBot

*[Read in English 🇬🇧](./README.md)*

アカペラやハミングに対して、ロボットがテンポとテンションを合わせて一緒に「ノる」システム。
ソフトウェアファーストで構築されており、以下のコードはすべて**ハードウェアなし**・**無料の計算環境**で動作します。
アルゴリズムの検証後、`robot/groovebot.urdf` に合わせて10自由度の上半身ロボットを組み立て、実機のサーボバックエンドに差し替えることができます（頭脳側のコードは一切変更不要です）。

## アーキテクチャ — 頭脳（Brain）と身体（Body）の分離

```text
 音声 ─► [ビートトラッカー] ─► 拍位相 ┐
      [感情モデル]      ─► 強度   ├─► GrooveController ─► RobotBackend ─► 身体
      [音声埋め込み]    ─► (M3)   ┘                         (インターフェース) │
                                                                           ├ MuJoCo  (現在・可視化用)
                                                                           ├ PyBullet(動力学シミュレータ)
                                                                           └ RealServo(実機組立後)
```

頭脳は常に `RobotBackend` インターフェースのみを呼び出し、特定のシミュレータを直接インポートすることはありません。MuJoCo、PyBullet、Isaac（薄いアダプタ経由）、あるいは実機サーボなど、どの「身体」で動かす場合でも、インターフェースを一度実装して注入するだけです。`robot/groovebot.urdf` が、シミュレータと物理ロボットの両方が従う唯一の情報源（「契約」）となります。

## ロードマップ

- **~~M0~~** *(保留 — `docs/SYSTEM_SPEC.md` §14.3 参照)*: アカペラとハミングに対する盲目オンラインビート追跡（BeatNet）。フォールバック用のコードとしてのみ保持し、現在は主軸の知覚パスではありません。
- **M0'** *(現在のイテレーション)*: 参照アライメントの実現性検証 — 既知の曲を選び、その音声を時間伸縮（synth_warp）させ、オフラインDTWを実行して、参照となる拍グリッドがどの程度復元できるかをスコア化します。ローカル、CPU、librosaのみで動作。後述の **M0' — 参照アライメントの実現性検証** を参照。
- **M1** *(完了)*: メトロノームと手作業で作成したグルーヴによるエンドツーエンドのループ。`python demo_groove.py`。
- **M2** *(必須)*: オンラインの `ReferenceAligner`（オンラインDTW / スコアフォロイング）をOrchestratorの知覚スレッドに組み込み、アローザル（覚醒度）推定＋画面/顔のフィードバックを追加。
- **M3** *(目標)*: 学習ベースの生成グルーヴ。VQ-VAEのグルーヴコードブックを、拍位相＋小節頭＋**楽曲構造**＋アローザル＋音声埋め込みで条件付けたTransformerでシーケンス化します。ボーカル分離したAIST++データセットで学習。`train/PIPELINE.md` を参照。

## 3つのプラグインの継ぎ目 (Seams)

すべての要素は安定したインターフェース（`docs/SYSTEM_SPEC.md` §5.2参照）の背後にあるため、1つずつ個別に実装できます。
- 拍 / アローザル源 → `Perception` プロトコル（`groovebot/orchestrator.py`）。M1では `MetronomePerception`（スタブ）を提供。M2でこれをライブの `BeatTracker` ＋ `ArousalEstimator` に置き換えます。
- グルーヴモデル → `GrooveGenerator` プロトコル（`groovebot/groove.py`）。M1では `RuleGrooveGenerator` を提供。M3で学習済みモデルに差し替えます。Orchestratorとデモは変更不要です。

## 無料で本格的な技術スタック (SOTA論文で実際に使われているもの)

- **PyTorch** — モデリング
- **AIST++** — 音楽↔3Dダンスのペアデータ（巨大なマルチ視点ビデオは不要で、SMPLモーション＋音声のみで済むため軽量）
- **Demucs** — フル楽曲からボーカルを分離 → 分離されたボーカルは元の曲の拍ラベルを継承（アカペラの拍データを用意するためのトリック）
- **WavLM / HuBERT** (HuggingFace) — 自己教師あり音声フロントエンド
- **Kaggle Notebooks** — 週約30時間の無料GPU（T4/P100 16GB）。データはクラウド上に留まるため、手元のPCのストレージやVRAMは不要です。本格的な学習時には研究室のGPUを使用します。

注: M3のコアは**教師あり生成モデリングであり、強化学習（RL）ではありません**。そのため、重いRLシミュレータやGPUファームは不要です。

## 実行方法 (ハードウェア不要)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# アクティベーションがブロックされる場合は、現在のユーザーに対して一度だけ許可を出します:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

pip install -r requirements.txt
# オプション: 動力学 + ウィンドウ表示 (PyBullet は requirements.txt には含まれていません)
# pip install pybullet

python demo_groove.py --backend mujoco --bpm 120 --energy 0.85 --seconds 8
python demo_groove.py --backend pybullet --gui      # pybulletインストール後のみ
pytest
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo_groove.py --backend mujoco --bpm 120 --energy 0.85 --seconds 8
pytest
```

## M0' — 参照アライメントの実現性検証 (ローカル, CPU, librosaのみ)

> 仕様書変更後の主要な知覚パス（§9.xおよび§14）。フォールバックとして保持している古い盲目トラッキングフローについては、後述の保留中のM0セクションを参照してください。

目的: **オフラインDTWアライメントにより、時間伸縮されたクエリから参照曲の拍グリッドを復元できるか**を証明すること。録音を行わずにグラウンドトゥルース（正解）を得るため、既知のレートで参照を時間伸縮させてクエリを合成します。

**実行されるもの:**

- `groovebot/align/features.py` — クロマまたはpyin-pitch特徴量 (12 × T)。
- `groovebot/align/dtw_align.py` — `OfflineDTWAligner` (librosa DTW) ＋ 参照拍をワープパスを介してクエリのタイムラインにマッピングする `map_reference_beats` ヘルパー。
- `tools/synth_warp.py` — wavに時間伸縮レートを適用し、ワープされた音声＋ワープされた拍注釈を出力。
- `experiments/run_m0p_align.py` — 完全なスイープを実行し、トラック別/レート別/全体のCSVと、トラック別のオーバーレイPNGを出力。

スコアリングは、盲目パスと同じ `mir_eval` ハーネス（`tools/eval_beat.py::score_beats`）を再利用するため、M0' の数値とM0の数値（復活時）は同じ表に収まります。

**データ準備**

1. 参照となる音源をいくつか選びます（`mir_eval.beat.trim_beats` 後でも十分な拍が残るよう、それぞれ10秒以上）。リポジトリの GTZAN-Rhythm 便利スクリプト（`tools/prep_dataset.py`, `experiments/run_gtzan_eval.py`）は、すでに音声と `.beats` アノテーションをペアにして処理します。
2. フルミックス音声の場合、DTWはボーカルステムで動作させる方がうまくいくことがあります。Demucsを一度実行し（ローカルにはインストールしないためColab/Kaggleで実行）、生成された `vocals.wav` を `.beats` ファイルの隣に配置します。初期の簡易テストではフルミックス音声でも機能します。
3. すべてを `data/m0p_refs/`（git-ignored）の下に、`<stem>.wav` ＋ `<stem>.beats` のペアとして配置します。

**インストール＋実行 (ローカル, Windows / mac / Linux)**

```powershell
# ローカルプロファイルには librosa のみ必要です（torch, madmom, Demucsは不要）。
pip install librosa

python -m experiments.run_m0p_align `
       --root    data/m0p_refs `
       --out-dir data/m0p_work `
       --feature chroma `
       --rates 0.9 0.95 1.0 1.05 1.1
```

出力先 `data/m0p_work/`:

```text
m0p_per_track.csv     (トラック, レート)ごとの各行: F / CMLt / AMLt / RT-factor
m0p_per_rate.csv      伸縮レートごとの平均
m0p_overall.csv       全体の平均
<stem>_r<rate>.png    ワープパス ＋ GT（正解）と復元された拍の波形オーバーレイ
```

指標の読み方はM0リアリティチェックと同じです: F (70msウィンドウ), CMLt (テンポ固定), AMLt (倍テンポ/半テンポの許容), RT-factor (処理秒数 / 音声秒数; オフラインDTWは1.0以下にはなりません。これはオンライントラッカーの特性であり、オンラインアライメントはM2で実装されます)。

**注意 (Tier 1)**: クエリは参照自体を時間伸縮させたものであるため、これはアライメントの*メカニズム*とテンポの堅牢性のみをチェックしており、パフォーマーが異なるより困難なケースは検証していません。以下のTier 2が真のテストとなります。

## M0' Tier 2 — 実際の演奏 (歌唱とハミング)

Tier 2では、自己ワープされたクエリを**同じ曲の別の演奏**（通常は、イヤホンで原曲を聴きながらユーザーが歌ったりハミングしたりしたもの）に置き換えます。これにより、実際の音色や微細なタイミングの変動が加わり、ハミングの最悪ケース（倍音構成がないためクロマがノイジーになり、pyinピッチの輪郭だけでアライメントを行う必要がある状態）が露呈します。

**入力レイアウト** (1曲につき1ディレクトリ):

```text
data/m0p_t2/<song>/
  original.wav            フルミックスの参照（ここからDemucsボーカルステムが派生）
  original.beats          参照となる拍時刻（1行につき1つの秒数）
  rendition_sing*.wav     (任意の数) 歌唱演奏 -> chroma DTW
  rendition_hum*.wav      (任意の数) ハミング演奏 -> pyin-pitch DTW
  rendition_*.wav         (任意の数) 汎用; デフォルトでchromaを使用
  gt.beats                全演奏共通のグラウンドトゥルース（正解）の拍時刻
```

ファイル名によって適切な特徴量が選択されます: ベース名のどこかに `hum`（または `humming`）が含まれていればpyinメロディパスが選択され、それ以外はchromaが使用されます。`gt.beats` はディレクトリ内のすべての演奏で共有されます。原曲に合わせて録音する場合、GT == `original.beats` となるため、そのままコピーして使用できます。

**データ準備 (推奨, 著作権クリーン)**

1. 合法的にアクセス可能な曲を2〜3曲選び、フルミックスの `original.wav` を `data/m0p_t2/<song>/` に配置します（gitignored — 音声は絶対にコミットしないでください）。
2. `original.beats` を配置します。GTZAN-Rhythmの場合はすでにアノテーションがあります。それ以外の場合はメトロノーム / 手動タップ / `tools/prep_dataset.py` を使用します。
3. イヤホンを装着して原曲を再生し、別のチャンネルで自分の歌声/ハミングを録音します。`rendition_sing*.wav` や `rendition_hum*.wav` として保存します。原曲に*合わせて*歌うため、原曲の拍グリッドが演奏の正解となります。`original.beats` を `gt.beats` にコピーしてください。
4. 代替案: GT拍がデータセットから提供されている、ライセンス済みのカラオケやカバー研究データセットを使用します。

> 元の録音は `data/` の下に保持してください（`.gitignore`でカバー済み）。パブリックサービスへのコミット、プッシュ、アップロードは**絶対に行わないでください** — 削除後もキャッシュされる可能性があります。

**インストール＋実行**

```powershell
# ローカルプロファイル (librosa)。DemucsはColab/Kaggle等の実験プロファイル上にあり、
# ランナーはtools.prep_dataset経由で遅延インポートし、build_reference()に失敗した曲をスキップします。
pip install librosa

python -m experiments.run_m0p_t2 `
       --root    data/m0p_t2 `
       --out-dir data/m0p_t2_work
```

出力先 `data/m0p_t2_work/`:

```text
m0p_t2_per_rendition.csv  演奏ごとの各行: F / CMLt / AMLt / RT-factor
m0p_t2_per_kind.csv       特徴量種類ごとの平均 (chroma vs pitch)
m0p_t2_per_song.csv       曲ごとの平均
m0p_t2_overall.csv        全体の平均
<song>_<rendition>.png    クエリ波形 ＋ GTと復元された拍 ＋ ワープパス
```

スコアリングは、Tier 1や保留中の盲目パスと**同じ `tools/eval_beat.py::score_beats` mir_eval ハーネス**を使用するため、すべての数値が同じ表に収まります。

## M0' Tier 2 — DAMPルート (録音不要、大規模な実際のアマチュアデータ)

録音を行わずに、**DAMP-VSEP** または **DAMP-S-AG** コーパス（Smule Research Data License）を使用して何千もの独立したアマチュアパフォーマンスにスケールさせるTier 2のバリエーションです。これらのデータセットには事前に分離されたステム（ボーカル / 伴奏 / ミックス）が含まれているため、ここで**Demucsは不要**です。詳細な理由と制限については、仕様書 §9.x DAMP を参照してください。

各アレンジメント（多くの歌手が同じバッキングトラックに合わせて歌った同じ曲）に対して:

- **beats**: バッキングに対する `librosa.beat.beat_track`。バッキングはインストゥルメンタルでありオフラインであるため信頼性が高いです（これは保留された盲目ボーカル拍パスではありません）。
- **chroma ref**: バッキングに対する `librosa chroma_cqt`。
- **melody ref**: **指定された**1つの演奏のpyin F0、またはすべての演奏のF0のフレーム単位のnanmedianから構築された**コンセンサス**（クエリごとのleave-one-out）。

すべてのクエリは同じスコアラーで**両方のパス**について評価されます:

- **chromaパス**: クエリボーカルchroma -> バッキングchroma。
- **pitchパス**: クエリF0 -> メロディリファレンス。

**ライセンスとデータの取り扱い**

DAMPデータセットはSmuleへのリクエストが必要であり、Research Data License（**非営利、再配布禁止**）の下でリリースされています。他の `data/` と完全に同じように扱ってください:

- ローカルのみ。`.gitignore`（`data/` ツリー全体）でカバーされています。
- DAMP音声を**絶対にコミット、プッシュ、アップロードしないでください**。パブリックキャッシュは削除後もコピーを保持する可能性があります。
- 派生したアーティファクト（分離されたステム、F0輪郭、さらにはスペクトログラムなど）を、プロジェクトのローカル作業ツリーの外部に再配布しないでください。

**正規化された入力レイアウト** (サブディレクトリごとに1つのアレンジメント):

```text
data/m0p_t2_damp/<arrangement_id>/
  backing.wav
  vocal_<rendition_id>.wav     (1つ以上)
```

DAMP-VSEPの場合、各（またはフィルタリングされた各）セグメントを1つの演奏を持つアレンジメントとして扱います。DAMP-S-AGの場合、すべての歌手のテイクが同じアレンジメント（`amazing_grace`）の演奏になります。生のDAMPファイル名はリリースによって異なります。一致しない場合は、上記のレイアウトにコピーまたはシンボリックリンクする1回限りのスクリプトを記述してください。`tools/ingest_damp.py list --root data/m0p_t2_damp` は簡単なサニティリストを出力します。

**実行**

```powershell
pip install librosa

# 指定されたメロディリファレンス (IDによる最初の演奏)
python -m experiments.run_m0p_t2_damp `
       --root    data/m0p_t2_damp `
       --out-dir data/m0p_t2_damp_work

# コンセンサスメロディ (演奏間のleave-one-outメジアン)
python -m experiments.run_m0p_t2_damp `
       --root    data/m0p_t2_damp `
       --out-dir data/m0p_t2_damp_work `
       --melody-mode consensus
```

### DAMP-S-AG バリエーション — MIDIリファレンス (バッキング音声 / ffmpeg 不要)

DAMP-S-AG（Sing! Amazing Grace, 1曲の17,582の演奏）には、音声の隣に `amazing_grace.midi` が同梱されています。M4Aバッキングトラックをデコードする必要を回避し、このMIDIを直接リファレンスとして使用します。`pretty_midi` を使用すると、chroma特徴量と同じ `(12, T)` 形状の拍、one-hotピッチchromaメロディ、および列L2 chromaテンプレートが得られます。ffmpegは不要です。演奏はlibsndfileで読み取り可能です（`.m4a` 拡張子、OGG/VORBISペイロード）。

**データ準備**

`tools/ingest_damp damp-s-ag` は、生の `amazing_grace.tar.gz` から直接、正規化されたアレンジメントレイアウトにサブセットをストリーム抽出します。ソースのtarballは**変更されません**。

```powershell
pip install librosa pretty_midi

# 最初の100件の演奏（アーカイブ順）とMIDIを抽出します。
python -m tools.ingest_damp damp-s-ag `
       --tarball data/amazing_grace.tar.gz `
       --out     data/m0p_t2_damp `
       --max-n   100

# --max-nが適用される前のオプションのフィルタ:
#   --headphones-only        (TSV列 `headphones == 1`)
#   --country US             (TSV列 `country`)
```

**実行**

```powershell
python -m experiments.run_m0p_t2_damp `
       --root             data/m0p_t2_damp `
       --out-dir          data/m0p_t2_damp_work `
       --reference-source midi `
       --vocal-glob       "vocal_*.m4a"
```

ピッチパス（クエリF0 vs MIDIメロディ）はハミング動作の主要な評価用であり、chromaパス（クエリchroma vs MIDI chromaテンプレート）は比較のために報告されます。MIDIモードはすべての演奏をスコア化します（指定/コンセンサス選択なし）。

**DAMP-S-AG の具体的な制限**

- 1曲のみ。DAMP-S-AGは単一アレンジメントのディープダイブです。アレンジメントの網羅性については依然としてDAMP-VSEPが必要です。
- *Amazing Grace* は**遅く、ルバート（テンポの揺らぎ）が強い賛美歌**です。演奏はテンポとフレージングが異なります。リファレンスグリッドは依然としてすべての歌手が聴いた同じ共有バッキングトラックから来ているため、DTWが吸収しなければならないワープは本物のパフォーマー間の変動です。
- 前セクションのその他のDAMP-S-AGの制限（バッキングに固定されたタイミング、ハミングのプロキシとしての歌唱）は引き続き適用されます。

出力先 `data/m0p_t2_damp_work/`:

```text
m0p_t2_damp_per_path.csv          (演奏, パス)ごとの各行: F / CMLt / AMLt / RT
m0p_t2_damp_per_kind.csv          特徴量種類ごとの平均 (chroma vs pitch)
m0p_t2_damp_per_arrangement.csv   arrangement_idごとの平均
m0p_t2_damp_overall.csv           全体の平均
<arr>__<rendition>__<kind>.png    クエリボーカル ＋ GTと復元された拍 ＋ ワープパス
```

**数値を読む際の限界** (仕様書 §9.x DAMP にも記載):
- DAMPの演奏はバッキングに*合わせて*歌われているため、タイミングはほぼ固定されており、完全に自由なテンポの独立したパフォーマンスよりも簡単です。
- ピッチパスは**本物の歌唱をハミングのプロキシとして使用**します。実際のパフォーマーのF0（自己派生ではない）でF0のみのDTWパスをテストしますが、ピッチが不安定になりがちな真のハミングに対しては楽観的かもしれません。

## ~~M0~~ — ビートトラッキングのリアリティチェック *(保留されたフォールバック、盲目パス)*

> `docs/SYSTEM_SPEC.md` §14.3 に従って保留中（madmom 0.16.1 が Colab の Python 3.12 + numpy 2.x スタックでビルドできないため）。このコードは、未知の曲や即興のハミング用のフォールバックパスとして残ります。依存関係の状況が解決され次第、数値を再度有効にします。

目的: 自分のアカペラとハミングに対して本物の歌唱用ビートトラッカーを実行し、どこで破綻するかを確認し、指摘できる数値を得ること。トラッカーは [BeatNet](https://github.com/mjhydri/BeatNet)（オンライン/パーティクルフィルタモード、NFR-2 因果的）です。ハーネスは `tools/eval_beat.py` です。

**録音プロトコル (クリック同期のグラウンドトゥルース)**

1. イヤホンを装着し、既知のBPMでメトロノームを再生します。
2. アカペラ → ハミング → アカペラ の短いテイク（各約30秒）を録音します（マイク＋イヤホンの音声は耳にのみルーティングし、マイクには入れない）。WAVとして保存します。
3. メトロノームのBPMがグラウンドトゥルース（正解）です。ハーネスは `--bpm` からクリックグリッドを構築します。

WAVを `data/`（git-ignored）の下に保存します。

**実行環境**

- **ローカル** (WindowsノートPC): ハーネスの合成WAVスモークテストのみがここで実行されます。BeatNet自体はWindowsでは重いため（torch + madmomを引き込む）、ローカルにはインストールしません — `tests/test_beat_tracker.py::*_real_beatnet_*` はスキップされます。
- **Colab / Kaggle**: BeatNetはここで実行されます。`!pip install -r requirements-experiments.txt` でインストールし、同じ `tools.eval_beat` CLIを呼び出します。

**実行**

```bash
# ローカル — 合成スモーク (BeatNet不要):
python -m tools.eval_beat synth --out data/click_120.wav --bpm 120 --seconds 8 --with-vocal

# BeatNetがインストールされている場所 (Colab/Kaggle、またはpip install後):
python -m tools.eval_beat eval --wav data/singing_120.wav --bpm 120 \
       --beats-per-bar 4 --out data/singing_120.png
python -m tools.eval_beat eval --wav data/humming_120.wav --bpm 120 \
       --beats-per-bar 4 --out data/humming_120.png --json > data/humming_120.json
```

**結果表 — 実際の録音が存在したら記入**

| トラック | BPM | F | CMLt | AMLt | RT-factor |
|---|---:|---:|---:|---:|---:|
| a-cappella_120bpm.wav | 120 | _ | _ | _ | _ |
| humming_120bpm.wav    | 120 | _ | _ | _ | _ |
| a-cappella_90bpm.wav  |  90 | _ | _ | _ | _ |
| humming_90bpm.wav     |  90 | _ | _ | _ | _ |

指標の読み方:
- **F** — 近接F値（70msウィンドウ）。クリックの近くに着地した検出拍の数をカウントします。
- **CMLt** — 正確なメトリカルレベル（テンポ固定）。トラッカーが間違ったテンポグリッド上にある場合は0に低下します。
- **AMLt** — 許容されるメトリカルレベル。テンポの半減/倍増を許容します。CMLtとAMLtのギャップは、ハミングで気にするべき「オクターブエラー」シグナルです。
- **RT-factor** = 処理秒数 / 音声秒数。1.0以下の場合はリアルタイム対応を意味します。

### 公開データ評価 (録音不要; SOTAと比較可能)

ハーネスは `--beats <FILE>`（1行に1つの拍の秒数）を介してアノテーションファイルも受け入れます。ビートアノテーション付きの公開データセットからボーカル分離された音声と組み合わせることで、自作の録音が存在する前に、SOTA互換のグラウンドトゥルースに対して同じ評価を実行できます（仕様書 §10.2）。

**推奨される公開データセット**

- *拍注釈付きポップス/ロック* (メイン — SOTAと直接比較可能):
  - **GTZAN-Rhythm** (Marchand & Peeters 2015) — 1000トラック、10ジャンル
  - **Ballroom** (Gouyon et al. 2006) — 698トラック、安定したテンポ
  - **Hainsworth** (Hainsworth & Macleod 2004) — 222の難しいトラック
  - **Isophonics / Beatles** (Mauch et al. 2009) — 拍＋コード＋構造
  - **RWC Popular** (Goto et al. 2002, AIST annotations) — 100のJ-Popトラック
- *実際のアカペラ* (対照 — Demucs不要):
  - **Dagstuhl ChoirSet** (Rosenzweig et al. 2020)
  - **Choral Singing Dataset** (Cuesta et al. 2018)
- *ハミングコーパス* (補助 — メロディのみのGT。F値ではなく、テンポのみ/拍ありの分析用):
  - **MIR-QBSH** (Jang & Lee 2008), **MTG-QBH** (Salamon et al. 2013)

**パイプライン (Colab/Kaggle — DemucsとBeatNetが配置されている場所)**

```bash
# (1) データセットのアノテーションを当プロジェクトの --beats 形式に変換。
#     どこでも実行可能。重い依存関係なし。
python -m tools.prep_dataset convert-ann --dataset ballroom \
       --input  raw/Ballroom/annotations/Albums-AnaBelen_Veneo-11.beats \
       --output data/beats/AnaBelen_Veneo-11.beats

# (2) Demucsでボーカルを分離。Colab/Kaggleのみ。
#     !pip install -r requirements-experiments.txt
python -m tools.prep_dataset separate \
       --wav     raw/Ballroom/audio/AnaBelen_Veneo-11.wav \
       --out-dir data/vocal

# (3) クリック同期録音と同じように評価しますが、--beatsを使用します:
python -m tools.eval_beat eval \
       --wav   data/vocal/AnaBelen_Veneo-11.wav \
       --beats data/beats/AnaBelen_Veneo-11.beats \
       --out   data/eval/AnaBelen_Veneo-11.png
```

ローカルでは、**(1)** のみが実行されます — パーサーは純粋なPythonです（`tests/test_prep_dataset.py`でカバー済み）。ステップ **(2)** と **(3)** は、Demucs / BeatNetが見つからない場合、「Colab/Kaggleにインストールされています」という明確なエラーを発生させます。

### ターンキーColabノートブック — GTZAN-Rhythm スイープ

`notebooks/m0_gtzan_eval.ipynb` は、新しいColab GPUランタイム上でパイプライン全体（クローン、インストール、GTZAN音声＋Marchand–Peetersアノテーションの取得、Demucsボーカル分離、BeatNet評価、ジャンル別表＋CSV）を実行します。

**Colabで開く:**
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yukijames25/groovebot/blob/master/notebooks/m0_gtzan_eval.ipynb)

またはブラウザにこのURLを貼り付けてください:
`https://colab.research.google.com/github/yukijames25/groovebot/blob/master/notebooks/m0_gtzan_eval.ipynb`

ノートブックの詳細:
- **ボーカルが多い** GTZANジャンルのみを評価します（`pop`, `rock`, `hiphop`, `reggae`, `blues`, `country`, `disco`）。クラシック / ジャズ / メタルは除外。
- BeatNetの `online` モードのみを使用します（NFR-2）。
- `PER_GENRE_LIMIT` ユーザー設定があります（デフォルト5 → 合計35トラック; T4で約20〜30分）。
- GitHubミラーを優先し、必要に応じてKaggleミラーにフォールバックします（`kaggle.json`が必要）。
- `data/m0_work/{beats,vocal,eval}/` ＋ トラック別およびジャンル別のCSVを出力します。

実際のループ / 集約は `experiments/run_gtzan_eval.py` にあり、`tests/test_run_gtzan_eval.py` によってカバーされています（ローカルではDemucs / BeatNet / torch不要）。

**ローカルでのノートブック編集:** ノートブックは `tools/_build_m0_notebook.py` から生成されます。このスクリプトを編集し、`python -m tools._build_m0_notebook` を実行して `notebooks/m0_gtzan_eval.ipynb` を再生成してください。構造とCLIフラグの一貫性は `tests/test_notebook.py` によってチェックされます。


## GrooveStyleSelector — 起動時のスタイルピッカー (v1, 並行トラック)

上記のアライメント/ビート処理は、ロボットが**いつ（When）**動くかを決定します。`groovebot.style` パッケージは、曲のキャラクターに合わせて、ヘドバン、揺れ、ペンライト振りなど、どのような種類のノリを、どれくらいの強さで（**どのように / How**）行うかを決定します。
これは意図的にタイミングトラッキングとは別の独立した領域として設計されています。スタイルの決定は起動時ウィンドウ（曲の最初の約5〜10秒の音声）で一度だけ行われるため、DAMPデータの利用可能性やオンラインアライメントの進捗がブロックされることはありません。

（契約の詳細については、仕様書 `docs/SYSTEM_SPEC.md` §14のモジュールノートを参照してください。）

### 設計

```text
 音声 (5-10 秒の起動ウィンドウ)
   ├── features.log_mel_spectrogram ─► StyleCNN ─► (genre_probs, mood_probs)
   ├── attributes.estimate_tempo    ─► tempo BPM
   └── attributes.estimate_arousal  ─► arousal 0..1 ─► arousal_bucket
                                                            │
                              table.select_move(genre, arousal_bucket, mood_probs)
                                                            │
                                                            ▼
                                                      GrooveStyle
                                                  (動作 + 強度のテキスト)
```

- **features.py** — 対数メルスペクトログラム (librosa, n_mels=64, hop=512, 22050 Hzにリサンプリング)。時間次元は可変。モデルは適応的平均プーリングを介して処理します。
- **model.py** — 小型の4ブロックCNN + 各ヘッドの `ModuleDict`。v1には `genre` (10クラス、GTZAN語彙) と `mood` (6クラス) が含まれます。テンポとアローザルは今のところヒューリスティックで計算されます（後で学習済みヘッドに昇格可能）。
- **attributes.py** — `estimate_tempo` (librosa.beat.beat_track) と `estimate_arousal` (正規化されたRMS音量とオンセット密度の幾何平均)。
- **table.py** — Yukiの手作りのノリのルックアップテーブル: `(genre, arousal_bucket, mood) → (move, intensity)`。moodはargmaxではなくソフトな確率分布として入力されます。
- **select.py** — `GrooveStyleSelector.select(audio, sr)` は、`move`、`intensity`、`tempo_bpm`、`arousal`、および完全なsoftmax確率辞書を持つ `GrooveStyle` データクラスを返します。

### 上半身で実現可能な動作語彙 (10 DOF, 脚なし)

`headbang`, `bob_nod`, `sway`, `rock`, `fist_pump`, `clap`, `penlight_wave`, `quiet_listen`。動作の意味論は `groovebot/style/table.py` に文書化されています。v1はテキストラベルのみを出力します。

### データ

- **Genre** (リアル): GTZAN, 10クラス。`data/raw/gtzan_mini/genres/<genre>/<file>.wav` でパイプラインのスモークテスト（動作確認）が可能です。意味のある精度を得るにはフルGTZAN（約1000ファイル）が必要です。
- **Mood** (スタブ): CCのムードタグ付き音声はまだ接続されていません。`experiments/train_style.py` は、エンドツーエンドの学習ループが実行できるように、決定論的なジャンル→ムードの疑似マップ（`_STUB_MOOD`）を使用します。
- **TODO**: `_STUB_MOOD` を **MTG-Jamendo moodサブセット** (約1万4000のCCクリップ、自動タグ付けムードラベル) を使ったローダーに置き換えること。
- **絶対ルール**: ディスク上やコミットに、著作権のあるJ-Popの音声や動画を含めないこと。スタイルパイプライン全体は、CC / 研究用ライセンスの素材のみを取り込みます（データは `data/` の下に置かれ、gitignoredされます）。

### 実行方法

インストール（M0'プロファイル上にさらに1つ追加するだけです）:
```bash
pip install torch  # v1はCPUで十分です; 小さなCNNは数分で学習できます
```

`gtzan_mini` でのスモークトレーニング:
```bash
python -m experiments.train_style \
    --gtzan-root data/raw/gtzan_mini/genres \
    --out-dir data/style_smoke \
    --epochs 20 --batch-size 8
```

出力:
- `data/style_smoke/style_cnn.pt` — チェックポイント
- `data/style_smoke/report.json` — 学習/検証の履歴、代表的なジャンルごとの `GrooveStyle` ラベル、テンポ推定など

Pythonからの推論:
```python
import soundfile as sf
from groovebot.style import GrooveStyleSelector

selector = GrooveStyleSelector()                     # ランダムな重み
# selector.model.load_state_dict(torch.load("data/style_smoke/style_cnn.pt")["state_dict"])
audio, sr = sf.read("some_song.wav", dtype="float32", always_2d=False)
style = selector.select(audio[: 10 * sr], sr)
print(style.as_text())
# -> "headbang@0.95 (metal/aggressive, 145BPM, arousal=0.81/high)"
```

### 検証

`tests/test_style_*.py` でカバーされています（CPUのみ、実際の音声データセット不要）。
`pytest -q tests/test_style_*.py` を実行すると、すべてグリーンになります。GPUは不要です。

### v1 の正直な限界
- ムードヘッドはフェイク（ジャンルからの決定論的マップ）。
- gtzan_miniはジャンルあたり10ファイルしかないため、精度は低い。
- アローザル（覚醒度）のヒューリスティックは信号レベルであり、知覚的ではない（将来的にAVECスタイルのラベルで学習させる必要がある）。
- テンポ推定はオクターブエラー（倍・半分のテンポ）を起こしやすい。
- まだテキストラベルを出力するだけ。

### v2 — ジャンルヘッド、本物の数字 (フルGTZAN、2つのスプリット)
v2では、ジャンルヘッドをフルGTZAN（1000トラック）で学習させます。フォールトフィルタリングおよびアーティスト非重複の分割（正直な数字）と、ナイーブな層化分割（楽観的なベースライン）の両方を報告します。
（データ準備と学習のコマンド詳細は原本を参照してください）

### v3 — 転移学習 (凍結された PANNs CNN14) ＋ 本物のムードヘッド
v3では、v2のスクラッチからの小さなStyleCNNバックボーンを、**凍結された学習済みの PANNs CNN14**（AudioSetで学習）に置き換え、その2048次元のクリップ埋め込みの上にMLP `StyleHead` を配置します。
また、MTG-Jamendo `autotagging_moodtheme` コーパスを接続し、ムードヘッドも本物のデータで学習させます。

（データ取得、セレクタAPIの利用、ジャンルとムードの転移学習の詳細な結果と考察、およびv3.1のDEAMデータセットを使用したアローザル/バレンスの統合については原本を参照してください。）

## JointCommandブリッジ v1 (スタイル条件付きMuJoCoグルーヴ)

`groovebot/groove_style.py` は、検証済みのスタイル属性（動作、強度、テンポ、アローザル）が実際に身体を駆動するように、各ティックで `GrooveStyle` を `JointCommand` に変換します。

```text
 GrooveStyleSelector.select(audio_window)          # 低速 (5-10 s)
                   └─► GrooveStyle  ─► StyleGrooveGenerator.set_style()
                                                 ▲
 Orchestrator (30-50 Hz)  ─► generate(ctx)  ─────┘  ─► JointCommand
                               │ beat_pos = MetronomePerception(style.tempo_bpm)
                               ▼
                        URDF clamp (NFR-4)  ─►  MuJoCoBackend
```

**プリミティブライブラリ** (`MOVE_PRIMITIVES`): `table.MOVES` の各動作に対する1つの関数。各プリミティブは `intensity` で線形スケーリングされ、振幅はURDFの制限内にあるソフトシーリング（`SOFT_AMP`）に収まり、さらにハードな安全ガードとしてOrchestratorの `clamp_command` が機能します。

**試す:**

```bash
# 全ての動作プリミティブ → それぞれGIF化 (data/renders/<move>.gif)
python -m experiments.render_groove --all-moves --seconds 4

# 単一の動作、高速テンポ
python -m experiments.render_groove --style headbang --bpm 140 --seconds 6

# セレクタパス: クリップが着地したスタイルを転送する
python -m experiments.render_groove --audio data/raw/clip.wav --seconds 8

# v1.1 ナレーション: "perception → decision → action" を標準出力に表示
python -m experiments.render_groove --style headbang --seconds 2 --skip-render --narrate
```

### v1.1 — ナレーションレイヤー (Observability / 観測可能性)

本番環境の出力は依然として `JointCommand` です。`groovebot/style/narrate.py` は、CSVが記録する同じティックごとの状態（`GrooveStyle` ＋ 関節ターゲット辞書）を人間が読める要約に変換します。これは完全に**読み取り専用（read-only）**であり、30-50 Hzの制御パスの外側にあります。

```text
[0.00-2.00s, 50Hz]
  知覚: genre=rock, mood=aggressive, arousal=0.85/mid, tempo=120BPM
  判断: GrooveStyle=headbang@0.85 (rock × mid × aggressive(1.00))
  動作: 首pitch（縦ノリ）、1拍/サイクル, 主関節=neck_pitch、設計上限|θ|≈0.55 rad
  beat#0 [t=0.000s] headbang: neck_pitch peak=-0.466 rad @ b+0.48 (off-beat)
  ...
```

### v1 ブリッジの正直な限界

- **シミュレータ ≠ 実際のサーボ**。MuJoCoはキネマティックな再生を行いますが、実際のサーボはオーバーシュートや遅れを示します。
- **ビート源はBPMメトロノーム**。実際のユーザーの歌声のタイミングではありません（M2で実際の ReferenceAligner に置き換わります）。
- **プリミティブは手作業で作成されている**。M3で学習済みの生成モデル（AIST++を使用）に置き換える予定です。

## ディレクトリ構成

```text
robot/groovebot.urdf          身体の契約（10自由度の上半身）
groovebot/
  backend.py                  RobotBackendインターフェース + MuJoCo / PyBullet / RealServo
  types.py                    GrooveContext / JointCommand
  groove.py                   RuleGrooveGenerator — 手作業で作成されたグルーヴ (M1)
  groove_style.py             StyleGrooveGenerator — GrooveStyle → JointCommand (v1ブリッジ)
  orchestrator.py             30-50 Hzの固定制御ループ + MetronomePerceptionスタブ
  limits.py                   URDF由来の関節制限とクランプヘルパー
  perception/                 知覚モジュール（BeatTracker等）
  align/                      M0' オフライン参照アライメント（ローカル、librosaのみ）
  style/                      GrooveStyleSelector (起動時のスタイル選択、並行トラック)
tools/                        評価、データセット準備、その他ユーティリティスクリプト
experiments/                  各種実験・評価用スクリプト
notebooks/                    Colabノートブック（GTZAN評価用など）
demo_groove.py                Orchestratorによって駆動されるエンドツーエンドループのデモ
tests/                        pytestによる自動テスト群
docs/                         仕様書（SYSTEM_SPEC.md）、ポートフォリオ、進捗ログ等の詳細ドキュメント
requirements.txt              コア依存関係（mujoco, mir_eval, soundfile, matplotlib, pytestなど）
```

> **Note**: 学習プロセスの詳細や、実験（M0, M0', v1/v2/v3 の転移学習など）の実行方法、公開データセットを使用した評価などについては、英語の元READMEまたは `docs/` 内の各ドキュメントを参照してください。
