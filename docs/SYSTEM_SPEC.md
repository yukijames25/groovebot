# GrooveBot ソフトウェアシステム仕様書

**版**: v0.1 ／ **種別**: 要求仕様 (SRS) ＋ ソフトウェア設計 (SDD) ／ **対象読者**: 開発者・指導教員

---

## 1. はじめに

### 1.1 本書の目的
アカペラ（鼻歌を含む）の歌声に対して、テンポと感情（テンション）にリアルタイム同期して
「ノる」ロボットを実現するための、**ソフトウェアの要求と設計**を定義する。ハードウェア
（自作機 / Pepper / 小型人型）に依存しない形で記述し、身体は差し替え可能な部品として扱う。

### 1.2 対象範囲 (Scope)
- **本書が扱う**: 音声入力 → 知覚 → ノリ生成 → 関節駆動・画面フィードバック の一連のソフトウェア、
  そのモジュール分割・インターフェース・リアルタイム設計・学習パイプライン・評価方法。
- **本書が扱わない**: 機構設計・電装（別途「プロジェクト全体仕様書」で扱う）。

### 1.3 スコープ外（将来課題）
- 高齢者向けの「穏やかな同調」スタイル（卒研等で別途）。
- 明示的なジャンル分類（当面は arousal＋tempo で代替）。
- カメラによるユーザ表情認識（オプションの視覚チャンネル）。

---

## 2. システム概要

### 2.1 コンセプト
ミラー型。歌声が激しく高テンションなら激しく（縦ノリ・ヘドバン）、静かなら穏やかに（左右の揺れ）。
ターゲットのスタイルは「カラオケで一緒に盛り上がる楽しい系」。

### 2.2 アクター
| アクター | 役割 |
|---|---|
| ユーザ（歌い手） | アカペラ/鼻歌で歌う。ロボのノリを受け取る |
| ロボット（身体） | 関節を駆動し、顔・胸画面で感情を表示する |
| 研究者 | システムを構築・学習・評価する |

### 2.3 主要ユースケース
1. **曲選択 → 参照ロード → 同期**: ユーザがカラオケ要領で曲を選ぶ →
   その曲の参照（拍時刻・小節頭・曲構造・アライメント用特徴）を読み込む →
   ユーザがアカペラ/鼻歌で歌う → ロボがオンラインで歌声を参照タイムラインへ
   整合し、参照側の拍グリッドに同期して動く。
2. テンションが上がる → 動きが大きく速くなる、画面に「Foo!!」等を表示。
3. 評価実験 → 同期条件 vs 非同期条件で、客観精度と主観評価を取得。

> **主軸**: タイミングは「ユーザが選んだ曲の参照拍グリッド」が支配する（§14）。
> 弱い歌声から盲目に拍を再抽出するのではなく、信頼できる外部タイムラインに整合する。
> 盲目オンラインビート追跡（BeatTracker 系）は未知曲・即興鼻歌向けの**任意のフォールバック**である（§14.3）。

---

## 3. 要求仕様

### 3.1 機能要求 (Functional Requirements)
| ID | 要求 |
|---|---|
| FR-1 | マイク/ファイルから歌声を取り込む |
| FR-2 | （フォールバック、§14.3）未知曲・即興鼻歌に対しては歌声から盲目にオンライン（causal）拍位相・ダウンビート・テンポを推定する |
| FR-3 | 歌声から覚醒度(arousal)・感情価(valence)・エネルギーを推定する |
| FR-4 | (M3) 歌声から自己教師あり埋め込み (voice embedding) を抽出する |
| FR-5 | 上記の条件からロボットの関節目標値（ノリ）を生成する |
| FR-6 | 関節目標値を身体（シム/実機）に送り駆動する |
| FR-7 | 顔の感情表示・胸画面（波形/絵文字/テキスト）を描画する |
| FR-8 | 身体を設定で差し替えられる（MuJoCo/PyBullet/Pepper/NAO/自作機） |
| FR-9 | 評価ハーネスは定常BPM（`--bpm`、合成クリック/被験者クリック同録）に加え、拍時刻列の注釈ファイル（`--beats`、1行1拍時刻[秒]）を受け付ける。これにより公開拍注釈データセット（§10.2）が同一指標で評価できる |
| FR-10 | ユーザは曲を選択でき、その曲の `SongReference`（拍時刻・小節頭・曲構造・アライメント用参照特徴）をロードできる |
| FR-11 | 生の歌声フレームを `SongReference` のタイムラインへオンライン（causal）に整合し、参照内位置から `beat_pos / downbeat / tempo / section` を `GrooveContext` に流す（主軸、§14） |

### 3.2 非機能要求 (Non-Functional Requirements)
| ID | 要求 | 目標 |
|---|---|---|
| NFR-1 | リアルタイム性（声→動作開始の遅延） | 端到端 **150 ms 以下**を目標 |
| NFR-2 | 因果処理 | 未来情報を使わない。過去の出力を訂正しない |
| NFR-3 | 身体非依存性 | 脳側はシム/実機APIを import しない |
| NFR-4 | 安全性 | 出力は必ず URDF の関節可動域内にクランプ |
| NFR-5 | 無料計算資源で学習可能 | Kaggle 等の無料GPUで学習が回ること |
| NFR-6 | モジュール差し替え性 | 各知覚・生成器を単体で交換できること |
| NFR-7 | 制御レート | 30–50 Hz で関節指令を出力 |

---

## 4. システムアーキテクチャ

### 4.1 全体構成
「脳（身体非依存）」と「身体（差し替え可能）」を **ports & adapters** で分離する。
脳は `RobotBackend` インターフェースにのみ依存する。

### 4.2 データフロー
```
 user picks song ──► SongReference  ───────────────────────────────┐
                     (beats, downbeats,                            │
                      sections, align_features)                    │
                                                                   ▼
 mic/file ──► AudioInput ──► (ring buffer) ──► ReferenceAligner ─► beat_pos,
                                  │            (主軸 §14)          downbeat,
                                  │                                tempo,
                                  │                                section
                                  ├──► ArousalEstimator ──► arousal, valence, energy
                                  └──► VoiceEncoder(M3) ──► embedding
                                                │
                                                ▼
                                          GrooveContext ──► GrooveGenerator ──► JointCommand
                                                                                       │
                                                       ┌───────────────────────────────┼────────────┐
                                                       ▼                               ▼            ▼
                                                 RobotBackend             FeedbackRenderer       (logging)
                                                (MuJoCo/…/RealServo)      (face/screen)

 (フォールバック経路: SongReference が無い／未知曲・即興鼻歌の時のみ、
  AudioInput ──► BeatTracker ──► beat_pos, downbeat, tempo  を使用。§14.3)
```

### 4.3 コンポーネント一覧
| コンポーネント | 責務 | 入力 → 出力 | フェーズ | 実装候補 |
|---|---|---|---|---|
| AudioInput | 音声取得・バッファリング | mic/file → frames | M0 | sounddevice / soundfile |
| ReferenceAligner | **オンライン参照アライメント（主軸）** | (frames, SongReference) → beat_pos, downbeat, tempo, section | M0'/M2 | online DTW (MATCH/Dixon系) / score following (Antescofo系) |
| BeatTracker | （フォールバック）盲目オンライン拍追跡 | frames → beat_pos, downbeat, tempo | 任意 | SingNet / mjhydri / BeatNet（棚上げ、§10.2 副次） |
| ArousalEstimator | テンション推定 (B-2) | frames → arousal, valence, energy | M2 | MER head（音源分離転移） |
| VoiceEncoder | SSL 埋め込み | frames → embedding | M3 | WavLM / HuBERT |
| GrooveGenerator | ノリ生成 | GrooveContext → JointCommand | M1→M3 | 規則(M1) → VQ-VAE+Transformer(M3) |
| RobotBackend | 身体駆動 | JointCommand → (物理) | M1 | MuJoCo/PyBullet/Pepper/NAO/RealServo |
| FeedbackRenderer | 顔・画面表示 | arousal,energy,waveform → 描画 | M2 | 画面UI（LCD/タブレット） |
| Orchestrator | リアルタイムループ統括 | — | M1 | 自前 |

---

## 5. インターフェース設計

脳と各部品の「契約」を型で固定する。実装が変わっても契約は不変。

### 5.1 データ型
```python
from dataclasses import dataclass

@dataclass
class GrooveContext:
    beat_pos: float        # musical position in beats (整数=拍数, 小数=拍内位相)
    downbeat: bool         # 小節頭か
    tempo: float           # BPM
    arousal: float         # 0..1  テンション（B-2）
    valence: float         # -1..1 感情価（B-2）
    energy: float          # 0..1  瞬間音量エンベロープ
    section: str | None = None              # 例: "verse" / "chorus" / "bridge"（参照モード時）
    embedding: "np.ndarray | None" = None   # M3 用 voice embedding

@dataclass
class JointCommand:
    targets: dict[str, float]   # joint_name -> radians（URDF の JOINT_NAMES）

@dataclass
class SongReference:
    """ユーザが選んだ曲の参照タイムライン（§14 主軸入力）。"""
    beats: "np.ndarray"          # shape (N,), 拍時刻 [秒]
    downbeats: "np.ndarray"      # shape (M,), 小節頭時刻 [秒]
    sections: list[tuple[float, float, str]]   # [(start_sec, end_sec, label), ...]
    align_features: "np.ndarray"  # 参照側のアライメント用特徴系列（例: chroma / pYIN melody / vocal SSL）
    sample_rate: int              # align_features のフレームレート [Hz]
    tempo: float                  # 名目 BPM（曲全体のミドル値で可）
```

### 5.2 モジュールインターフェース
```python
from typing import Protocol

class ReferenceAligner(Protocol):
    """主軸: ユーザ選択曲の参照タイムラインへ歌声を online 整合させる（§14）。"""
    def load(self, ref: SongReference) -> None: ...
    def update(self, frames) -> tuple[float, bool, float, str | None]:
        ...   # beat_pos, downbeat, tempo, section_label（参照位置に追随）

class BeatTracker(Protocol):
    """フォールバック: 参照が無い／未知曲・即興鼻歌でのみ使用（§14.3）。"""
    def update(self, frames) -> tuple[float, bool, float]: ...   # beat_pos, downbeat, tempo

class ArousalEstimator(Protocol):
    def update(self, frames) -> tuple[float, float, float]: ...  # arousal, valence, energy

class VoiceEncoder(Protocol):
    def update(self, frames) -> "np.ndarray": ...

class GrooveGenerator(Protocol):
    def generate(self, ctx: GrooveContext) -> JointCommand: ...  # M1規則もM3モデルも同一契約

class RobotBackend(Protocol):
    def load(self, urdf_path: str) -> None: ...
    def set_joint_targets(self, targets: dict[str, float]) -> None: ...
    def step(self, dt: float) -> None: ...
    def get_joint_states(self) -> dict[str, float]: ...
    def close(self) -> None: ...

class FeedbackRenderer(Protocol):
    def render(self, ctx: GrooveContext, waveform) -> None: ...
```
> 注:
> - 主軸の `ReferenceAligner.update` は **causal / online** に限る（NFR-2）。
>   offline DTW は禁止。online DTW (MATCH/Dixon) / score following 系のみ。
> - `ReferenceAligner` も `BeatTracker` も下流の契約（`GrooveContext` 構築）に対しては
>   交換可能。Orchestrator は主軸→フォールバックの順で選ぶ。
> - 現行コードの `GrooveController.compute(beat_pos, energy)` は M1 簡易版。
>   M3 へは `generate(ctx: GrooveContext)` に一般化する（このファイルだけ差し替え）。
> - 全ての `JointCommand.targets` は出力直前に URDF 可動域へクランプする（NFR-4）。

---

## 6. リアルタイム・並行設計

### 6.1 スレッド構成
| スレッド | 周期 | 役割 |
|---|---|---|
| Audio | コールバック | マイク取得 → ring buffer へ書き込み |
| Perception | 可変（重い） | ReferenceAligner（主軸） / ArousalEstimator / VoiceEncoder。参照が無い時のみ BeatTracker（フォールバック） |
| Control loop | 固定 30–50 Hz | 最新の GrooveContext で生成 → backend へ送出 |

知覚（重い・可変遅延）と制御（軽い・固定レート）を**疎結合**にし、制御ループは常に
「最新の推定値」を使って滑らかに補間する。これで知覚の遅延がノリのガタつきに直結しない。

### 6.2 レイテンシ予算（NFR-1: 合計 ≤150 ms）
| 区間 | 目標 |
|---|---|
| 音声バッファ | ≤ 40 ms |
| 拍/テンション推定 | ≤ 60 ms |
| ノリ生成 | ≤ 20 ms |
| 送出・サーボ整定 | ≤ 30 ms |

### 6.3 同期方針
タイミングは **ReferenceAligner**（ユーザ選択曲の参照拍グリッド）が支配し、
GrooveGenerator はスタイル・質感のみを担う（設計リスク#2への対策、§14）。
参照が無い場合に限り BeatTracker（盲目フォールバック）がタイミングを担う。

---

## 7. データ設計・学習パイプライン

### 7.1 学習データ生成（オフライン・無料）
1. AIST++（曲↔3Dダンス、SMPL＋音声。巨大な多視点動画は不要）。
2. Demucs で各曲のボーカルを分離 → ダンス動作に「ボーカル＋元曲の拍ラベル」が付く。
3. ボーカルで条件付けして学習（フル音源ではない）。

### 7.2 リターゲット（MLの実工程の山場）
AIST++ の上半身（脊椎・首・肩・肘）→ URDF の 10 関節へ写像。URDF 可動域でクランプ。

### 7.3 モデル成果物
VQ-VAE のノリ・コードブック ＋ 条件付き Transformer（beat_phase / arousal / valence / embedding）。
チェックポイントは Kaggle Datasets / Drive に保存（セッションでディスクが消えるため）。

---

## 8. 技術スタック
- 言語: Python
- ML: PyTorch、HuggingFace（WavLM/HuBERT）、Demucs、AIST++
- 拍追跡: SingNet / mjhydri 系
- シム: MuJoCo（既定）/ PyBullet、抽象は `RobotBackend`
- 実機: NAOqi（Pepper/NAO）または自作機のサーボ制御（配属後）
- 計算: Kaggle 無料GPU（プロトタイプ）→ 研究室GPU（本番）
- 任意: ROS2（`/joint_commands` トピックで身体を完全に疎結合化）

---

## 9. 開発フェーズとモジュール対応
| フェーズ | 内容 | 主に触るモジュール |
|---|---|---|
| ~~M0~~ | （棚上げ）盲目オンライン拍追跡（BeatNet/madmom）の評価。Colab/madmom 互換問題で死に筋、§10.2 副次へ降格 | （棚上げ：`tools/eval_beat.py`, `experiments/run_gtzan_eval.py` は残置） |
| **M0'** | **小規模な参照曲セットの作成 + オフラインで歌唱/鼻歌を参照拍グリッドへ整合できるかの実現性検証**。既知曲に対して歌唱/鼻歌をどれだけ整合させて拍グリッドを復元できるかを `mir_eval` ハーネス（F/CMLt/AMLt、流用）で測る。madmom 不要 | SongReference 作成スクリプト, オフライン整合プロトタイプ（DTW で可）, `tools/eval_beat.py`（指標流用） |
| M1 | メトロノーム＋手付けノリで端到端を通す（**完了**） | Orchestrator, GrooveGenerator(規則), RobotBackend |
| **M2（必達）** | オンライン `ReferenceAligner` を Orchestrator の Perception に接続 + arousal 推定 + 顔/画面フィードバック | ReferenceAligner, ArousalEstimator, FeedbackRenderer |
| **M3（目標）** | 拍/小節頭/**曲構造**/arousal/voice-embedding で条件付けした学習 groove 生成（AIST++ vocal 分離）。曲構造から「次サビ来る」を予期した先読みのノリ | VoiceEncoder, GrooveGenerator(モデル) |

### 9.x M0' 詳細: 参照アライメントの実現性検証
目的: 既知曲の参照に、その曲の歌唱/鼻歌をオフラインで整合させ、参照の拍グリッドを
どれだけ復元できるかを測る。madmom不要・GPU不要・CPUで完結(ローカル実行可)。

特徴量: 歌唱=クロマ(chromagram)既定／鼻歌=音高コンター(F0; pyin)既定(単旋律のため)。
  feature_fn は差し替え可能。
アライメント: オフラインDTW(librosa.sequence.dtw)で query↔reference の特徴量列を整合 →
  ワーピングパス → 参照拍を query タイムラインへ写像し復元拍を得る。
  (オンライン化(OLTW/score-following)は M2。NFR-2 はオンライン要件で、M0' はオフライン検証)
評価: 復元拍を既存 mir_eval ハーネス(F/CMLt/AMLt)で正解グリッドと照合。

Tier 1(最初・完全クリーン・新規DLや録音なし):
  参照音声+拍グリッドに既知のテンポ変動(time-stretch; 一定倍率集合→任意で区分的)をかけ query を生成 →
  DTWで整合 → 既知の正解ワープに対し拍復元精度を採点。アライメント機構とテンポ耐性を検証。
  注: 自己ワープのため楽観値(機構検証であり別人歌唱の難しさは測れない)。
  データ: 手持ちGTZAN-Rhythmのボーカル多めジャンルを数曲(全曲そのまま、または一度だけDemucsでボーカル分離)。MUSDB18(CC)も可。
Tier 2(Tier 1通過後・本物の頑健性):
  同曲の別演奏(別人の歌唱/鼻歌)を参照に整合。データはライセンス済みカラオケ/カバー研究データを優先。
著作権/規約: 音源は data/(gitignore済)にローカル保持。再配布(commit/push)しない。YouTube等を使う場合も
  data/ ローカル限定・分析専用・公開リポジトリへ絶対コミットしない。

### 9.x M0' Tier 2: 別演奏での頑健性
目的: 既知曲の参照に、その曲の「別演奏」(別録音の歌唱・鼻歌、アカペラ)を整合させ拍グリッドを
どれだけ復元できるか測る。Tier 1の自己ワープと違い、音色・表現・微小タイミングが異なる本物の試金石。
特に鼻歌(単旋律・和声なし・子音なし)が核心。

参照の作り方(重要): フルミックスではなく声/旋律で整合する。
  - reference vocal: 原曲を Demucs でボーカル分離。
  - reference melody: 参照ボーカルに pyin で F0 → 旋律特徴(鼻歌整合用)。
  - reference beats: 原曲フルミックスのビート追跡(librosa.beat; フルミックスは盲目でも信頼できる)。
特徴量: 歌唱=クロマ(声 vs 参照ボーカル)／鼻歌=音高コンター(query F0 vs 参照旋律)。
アライメント: オフライン DTW(Tier 1 の OfflineDTWAligner 流用) → 復元拍。
評価: 復元拍を演奏の正解拍(GT)に mir_eval(F/CMLt/AMLt)で採点。GT取得は演奏データの作り方に依存。

データ(演奏 query と GT 拍):
  - 推奨(クリーン・鼻歌を確実に含む): 既知曲 2-3 を、原曲をイヤホンで流しつつアカペラで歌唱/鼻歌録音。
    GT拍 = 原曲の拍グリッド(声を原曲に合わせ timeline 共有)。自分の声なので著作権無関係。
  - 代替: ライセンス済みカラオケ/カバー研究データ。
著作権/規約: 原曲・演奏音源は data/(gitignore)にローカル保持、commit/push しない。

### 9.x M0' Tier 2 (録音なし版): DAMP-VSEP による実演奏スケール評価
動機: 自前録音なしで、実アマチュア歌手の多数の独立演奏でアライメントを評価する。
データ: DAMP-VSEP (各セグメントが vocal v / backing b / mixture x を提供、6456歌手・41kセグメント)
  または DAMP-S-AG (同一曲 Amazing Grace の17,582独立演奏)。
  ライセンス: Smule Research Data License。要申請・非商用・再配布禁止 → data/(gitignore)ローカルのみ。
  ※ステム分離済みなので Demucs 不要。

参照(SongReference, アレンジ単位):
  - beats: backing(器楽)を librosa.beat でビート追跡 → 拍グリッド。
    (これは「器楽・参照側・オフライン」のビート追跡で信頼できる。放棄した『生アカペラの盲目追跡』とは別物。madmom不要)
  - chroma参照: backing のクロマ。
  - melody参照: 同一アレンジの別renditionのF0(既定: クエリと別の指定参照rendition、任意: 複数renditionのF0コンセンサス/leave-one-out)。
特徴量経路(両方走らせる):
  - chroma経路(歌唱): query vocal のクロマ → backing クロマ に整合。
  - pitch経路(鼻歌の代理): query vocal の F0 → melody参照 に整合。
    根拠: 同一旋律なら歌唱のF0と鼻歌のF0はともに旋律F0軌跡に収束するため、実歌声のF0で鼻歌経路を代理検証できる。
    queryのF0は実演者由来であり自己派生(出来レース)ではない。
GT拍: backing の拍グリッド(renditionはbackingに合わせて歌うため timeline 共有)。
評価: 両経路の復元拍を eval_beat.score_beats(Tier1と同一)で採点 → chroma/pitch別・アレンジ別・全体。
限界(正直に): (1) renditionはbackingに同期＝タイミングは概ね固定(自由テンポの独立演奏より易しい。ただし
  「選曲して合わせて歌う/鼻歌る」製品の主用途に一致)。(2) pitch経路は実歌声を鼻歌代理とするため真の鼻歌
  (音程が甘い場合あり)をやや楽観視しうる。真鼻歌の最終確認は QBH コーパス(MIR-QBSH等)を旋律アライメント指標で(別途・任意)。

DAMP-S-AG 向け補足 — reference=MIDI ルート:
  DAMP-S-AG には参照 MIDI ファイル(amazing_grace.midi、標準 MIDI)が同梱されており、
  これを使えば backing 音声・ffmpeg・librosa.beat を一切経由せずに grid / melody / chroma の
  すべてを「楽譜側」から取れる。実装は groovebot/align/midi_ref.load_reference_from_midi:
    - beats: pretty_midi.PrettyMIDI.get_beats() / get_downbeats()
    - melody: ノート列を (12, T) one-hot dominant pitch class chroma にラスタライズ
    - chroma_template: 同じノート列を column-L2 正規化した chroma 風テンプレート
  pitch 経路(query F0 → MIDI melody)は鼻歌想定の主軸、chroma 経路(query chroma → MIDI chroma)は
  従。MIDI モードは designated/consensus melody 切替が無く全 rendition がクエリ対象。
  運用: tools/ingest_damp damp-s-ag で tar から少数の rendition + MIDI だけストリーム抽出
  → experiments/run_m0p_t2_damp --reference-source midi で評価。raw tarball は不変。
  
---

## 10. テスト・評価

### 10.1 ソフトウェアテスト
- 単体: GrooveGenerator 出力が全フレームで可動域内（プロパティテスト）。BeatTracker を
  注釈付き音源に対して F値で検証。
- 結合: 端到端ループがレイテンシ予算内で動作。
- 身体非依存性: 同一の脳が MuJoCo / PyBullet 双方で動く回帰テスト。

### 10.2 研究評価
#### 主指標 — アライメント精度（主軸）
**「`ReferenceAligner` が参照拍グリッドをどれだけ復元できるか」** を中心に置く。
参照側の拍注釈を正解、`ReferenceAligner.update()` が返す `beat_pos` から再構成した
拍時刻列を推定値として F値 / CMLt / AMLt を計算する（指標と計算ハーネスは既存の
`tools/eval_beat.py --beats` を流用）。RT-factor（壁時計処理時間 / 音源時間、≤1.0 が
realtime 可）も同じハーネスで取得する。

- **客観（同期精度）**: 被験者にイヤホンでクリックを聴かせて歌わせ、その拍グリッドを
  正解として、`ReferenceAligner` が追従した参照拍グリッドのズレを測定。
  （`tools/eval_beat.py --bpm` ／ `--beats`）。
- **公開データ方式（SOTA 比較可能・録音前から走らせられる）**: 拍注釈付き公開データを
  **そのまま `SongReference` として使う**。歌唱/鼻歌（任意音源）を整合し、参照拍グリッドを
  正解として整合精度を測る（録音が無くても評価ループが回せる）。
  - **拍注釈あり楽曲データセット**（一次評価。SOTA 値と直接比較）
      - **GTZAN-Rhythm (Marchand & Peeters, 2015)**: 1000曲・10ジャンル、拍＋ダウンビート注釈
      - **Ballroom (Gouyon et al. 2006; ISMIR LB extended)**: 698曲、拍注釈、テンポが安定
      - **Hainsworth (Hainsworth & Macleod, 2004)**: 222曲、多ジャンル、難曲が多い
      - **Isophonics / Beatles (Mauch et al. 2009)**: ビートルズ等、拍＋コード＋**構造**注釈
      - **RWC Popular (Goto et al. 2002)**: J-Pop 100曲、AIST 配布
  - **本物アカペラ（control）**
      - **Dagstuhl ChoirSet (Rosenzweig et al. 2020)**: 多声合唱の単声ボーカル＋拍ラベル
      - **Choral Singing Dataset (Cuesta et al. 2018)**: 単声録音
  - **鼻歌コーパス（アライメント側の頑健性検証）**
      - **MIR-QBSH (Jang & Lee, 2008)**: 4431件の humming/singing、旋律 MIDI GT のみ
      - **MTG-QBH (Salamon et al. 2013)**: ~120件、旋律 GT、Creative Commons
  - **指標**（公開データ・録音とも同一）: F値（70ms 窓）、CMLt、AMLt、RT-factor。
  - **手順サマリ**: ① 公開データを取得 → ② 拍注釈を `SongReference.beats` として読み込み、
    参照側 align_features（chroma 等）を抽出 → ③ 歌唱/鼻歌を `ReferenceAligner.update()` で
    online 整合 → ④ 整合結果の拍時刻列を `--beats` 形式に出力 → ⑤ `eval_beat.py` で表に積む。

- **M0' オフライン採点パス（実装）**: §9.x の M0' Tier 1 では、`groovebot/align/dtw_align.py`
  のオフライン DTW が出した復元拍を `tools/eval_beat.py` の `score_beats()`（mir_eval ベース、
  F値 / CMLt / AMLt）にそのまま渡して採点する。スコアラは盲目モードと共有なので、参照
  アライメントと盲目モードを**同一指標**で並べられる。M0' の集計 runner は
  `experiments/run_m0p_align.py`。Tier 1 は合成 time-stretch（既知ワープ）を query 側に
  かけるため、機構検証としては楽観値（§9.x 注意書き参照）。
- **Tier 2 も同一スコアラ**: §9.x の M0' Tier 2（別演奏 query）も復元拍 → 演奏 GT 拍を
  同じ `tools/eval_beat.py::score_beats()` に通す。Tier 2 集計は
  `experiments/run_m0p_t2.py`、Tier 1 と Tier 2 の数字は同一指標で並べて比較できる。
- **Tier 2 DAMP は chroma / pitch 両経路を同一スコアラで採点**: §9.x の DAMP ルート
  （`experiments/run_m0p_t2_damp.py`）は、各クエリ rendition に対して chroma 経路（query 声
  vs backing クロマ）と pitch 経路（query F0 vs melody 参照）の両方を走らせ、それぞれの
  復元拍を同じ `score_beats()` に通す。CSV では `feature_kind ∈ {chroma, pitch}` で分離集計し、
  Tier 1 / Tier 2 録音版 / Tier 2 DAMP のすべての数字を一つの mir_eval 指標系で並べて比較できる。

#### 副次・棚上げ — 盲目オンラインビート追跡
盲目モード（`BeatTracker`）の同一指標評価は副次扱い。既存の Colab パイプライン
（`notebooks/m0_gtzan_eval.ipynb` ＋ `experiments/run_gtzan_eval.py` ＋ Demucs vocal 分離 ＋
BeatNet/madmom）はコードとして残置するが、現時点では **棚上げ**（madmom が Python 3.10
縛りで Colab の運用環境に乗らないため、数値化は据え置き）。フォールバック品質の参考値が
必要になった段階で再開する。

#### 主観（HRI）
同期条件 vs 非同期条件（わざと拍を外す）で、楽しさ・「合ってる感」・エンゲージメントを
比較。Keepon 型の対照実験設計。

---

## 11. リスクと対策
| リスク | 対策 |
|---|---|
| スタイルのドメインギャップ | ターゲットを「楽しい系」に固定し AIST++ と一致させた（解消済み） |
| 声条件 vs 音楽条件のズレ | タイミングは参照アライメントに任せ、モデルは質感のみ（§6.3, §14） |
| **未知曲・即興鼻歌でアライメントが効かない** | **主軸はカラオケ（既知曲）に固定。未知曲・即興鼻歌は弱フォールバック（盲目 `BeatTracker`、§14.3）のみで、性能は保証しない。HRI 評価でも既知曲条件のみを主軸とする** |
| 鼻歌が音声SSLの土俵外 | 歌唱 vs 鼻歌のアブレーション、音高/エネルギー特徴を併用 |
| ハードが間に合わない | 身体非依存設計。評価は既存ロボ（Pepper等）で代替可能 |
| 盲目拍追跡の依存（madmom）が壊れる | フォールバック扱いに降格済（§14.3）。本系の品質は影響を受けない |

---

## 12. リポジトリ構成（拡張案）
```
robot/groovebot.urdf        身体の契約（10 DOF）
groovebot/
  backend.py                RobotBackend と各実装
  groove.py                 GrooveGenerator（M1規則→M3モデル）
  perception/               BeatTracker / ArousalEstimator / VoiceEncoder
  feedback/                 FeedbackRenderer（顔・画面）
  orchestrator.py           リアルタイムループ
demo_groove.py              端到端デモ（M1）
train/                      学習パイプライン（AIST++→codebook）
docs/SYSTEM_SPEC.md         本書
```

---

## 13. 用語集
- **arousal/valence**: 感情の覚醒度・快不快の2軸。「テンション」は概ね arousal。
- **causal/online**: 未来を見ずに逐次処理すること（リアルタイムの必須条件）。
- **retarget**: 人体骨格の動きをロボの関節構成へ写し替えること。
- **ports & adapters**: 中核ロジックを外部実装から隔離する設計（脳/身体分離）。
- **ReferenceAligner**: 本システム主軸の知覚モジュール。ユーザ選択曲の参照タイムラインへ
  歌声を online 整合し、参照側の `beat_pos / downbeat / tempo / section` を出力する（§5.2, §14）。
- **SongReference**: 曲ごとの参照データ（拍時刻列・小節頭・曲構造ラベル・アライメント用
  参照特徴）。`ReferenceAligner.load()` の入力（§5.1）。
- **online alignment**: 過去入力のみから参照タイムラインへ追従する整合。online DTW
  (MATCH/Dixon) / score following (Antescofo, Music Plus One) が代表（§14）。
- **盲目（blind）ビート追跡**: 参照なしで音声のみから拍を抽出する古典手法。本書では
  未知曲・即興鼻歌向けの**フォールバック**扱い（§14.3）。

---

## 14. 参照情報つきアライメント（主軸 / Reference-informed alignment）

### 14.1 位置付け
本システムの**主軸の知覚**である。ターゲットはカラオケ＝既知曲なので、曲ごとに用意した
参照タイムライン（`SongReference`）へ歌声を online 整合し、参照側の信頼できる拍グリッド・
小節頭・曲構造を継承する。タイミングを「弱い歌声から拍を再抽出する」ではなく
「外部の確かな参照から得る」設計に倒すことで、鼻歌で拍が立っていなくても同期できる。

> 旧版（v0.1 初版）はこの節を「将来拡張」として扱い、M0 で盲目オンライン拍追跡
> （BeatTracker / BeatNet）を主軸ベースラインとしていた。本版で主従を入れ替えた。
> 理由: (a) madmom（BeatNet の必須依存）が Python 3.10 縛りで Colab/Kaggle
> （≥3.11）に乗らず実運用が困難、(b) カラオケ標的では参照アライメントの方が
> 堅牢かつ「楽曲構造を使った先読みのノリ」など下流の表現が豊か。

### 14.2 構成（`ReferenceAligner` 層の内部で完結。§5.2 の下流契約は不変）
1. ユーザが曲名／曲 ID で選曲（カラオケ的 UX）。
2. その曲の `SongReference`（拍時刻・小節頭・曲構造ラベル・アライメント用参照特徴系列）
   を事前計算データから読み込む。
3. マイクの生フレームから同種の特徴（chroma / pYIN melody / vocal SSL など）を online に
   抽出し、online DTW（MATCH/Dixon 系）または score following（Antescofo 系）で参照
   タイムライン上の現在位置を更新する。
4. 現在位置 → `beat_pos / downbeat / tempo / section` を `GrooveContext` へ流す。

### 14.3 フォールバック（盲目オンラインビート追跡）
参照が無い場合 — 未知曲・即興鼻歌・参照ロード未完 — のみ、`BeatTracker`（SingNet /
mjhydri / BeatNet 系）で歌声から盲目に拍を推定する。
- 性能は保証しない（リスク §11）。
- 既存の Colab/Kaggle 評価パイプライン（`tools/eval_beat.py`, `experiments/run_gtzan_eval.py`,
  `notebooks/m0_gtzan_eval.ipynb`、Demucs vocal 分離 ＋ BeatNet）はコードとして残置するが、
  madmom 互換問題により**棚上げ**（§10.2 副次）。フォールバック品質の参考値が必要に
  なった段階で再開する。
- Orchestrator は参照モード優先、参照が無い／信頼度低の時のみフォールバックに切替。

### 14.4 簡略化（曲選択 UX の効用）
- **自動曲識別（QBH / 旋律照合）は当面不要**。ユーザが曲を選ぶ前提なので、識別の不確実性と
  「自信満々に誤った拍」リスクを丸ごと回避できる。将来 UX 上必要になった段階で追加検討。
- 信頼度ゲートも、当面は「ユーザ選曲モード」vs「フォールバック」の二択でよい（細かい
  曲内チャネル切替は後段）。

### 14.5 制約・留意
- **NFR-2 を維持**: offline DTW は禁止、online DTW / score following のみ。過去出力を
  訂正しない。
- クラシック等（rubato 強い・レパートリー曖昧）は弱点。カラオケ用途ではスコープ外として許容。
- 参照側の事前計算（拍注釈・align_features の抽出）はオフラインで自由に重い処理を使ってよい。
  リアルタイム制約は live 側のみ。

### 14.6 副産物
`SongReference.sections` から「次サビが来る」を予期した先読みのノリ
（anticipatory grooving）が可能になる（M3）。盲目追跡では不可能な音楽的表現。

### 14.7 段取り
**M0'**: 小規模な参照曲セット作成 + オフライン整合の実現性検証（既知曲に対して歌唱/鼻歌を
DTW で整合し、参照拍グリッドの復元率を `mir_eval` で測る、§10.2 主指標）。→
**M2**: オンライン `ReferenceAligner` を Orchestrator の Perception スレッドに接続。→
**M3**: 拍/構造/arousal/voice-embedding で条件付けした学習 groove 生成。
関連先行研究: online DTW（MATCH/Dixon）, score following / 自動伴奏（Antescofo, Music Plus One）。

---

### モジュール: GrooveStyleSelector（曲調→ノり方スタイル）
役割: 起動ウィンドウ(歌い始め5–10s)の音声から曲調属性を推定し、ルール表で「ノり方の種類＋強さ」を選ぶ。
  「どう動くか(スタイル)」担当で、「いつ動くか(タイミング)」の整合/拍系とは独立。出力は当面テキストラベル。
  タイミング系(M0'/M2)とは別トラックで、DAMP-VSEPのアクセス可否に依存せず開発可能。
入力: 起動ウィンドウ音声 → log-melスペクトログラム(画像認識スタイルのCNN入力)。
属性(マルチタスク):
  - ジャンル: 分類(ノリの語彙を決める)
  - テンポ: librosa.beatで計算(学習不要)
  - arousal/エネルギー: 連続(v1は特徴量ヒューリスティック) → 動きの強さ ＋ 激しい↔ゆっくりのバケット
  - ムード: 6クラス離散分類 {aggressive, happy, sad, calm, dark, epic}(暫定・差し替え可)
ルール表: (ジャンル × arousalバケット × ムード) → ノり方の種類(離散)＋強さ。ムードはargmaxでなく確率分布(ソフト)で重み付け。
ロボット実現可能クラス(上半身~10DOF・脚なし): ヘドバン/ボビング・頷き/左右揺れ/前後揺れ/拳上げ・フィストパンプ/拍手/腕振り(ペンライト風)/静聴。
学習データ: CCの音楽タグ(FMAジャンル, MTG-Jamendoムード等)。著作権付きJ-pop音源・映像は使わない。
検証: ジャンル/ムードの分類精度(保留)、テンポ誤差、代表クリップの出力ラベル妥当性。

---

### GrooveStyleSelector v3: 転移学習＋MTG-Jamendo
動機: v2(GTZAN, from-scratch, 443訓練曲)は fault test 43%で頭打ち。律速は「少データ×ゼロ学習」。
転移学習＋大きく綺麗なマルチタグデータへ移行。骨組み(table/select/出力契約)は不変、モデル中身とデータのみ差替え。
バックボーン: 事前学習済み音声(PyTorch: PANNs/CNN14)を凍結し、曲ごと埋め込みを1度抽出しキャッシュ → 小ヘッドのみ訓練。
ヘッド: 埋め込み→genre(分類)/mood(6クラス分類)。tempo=librosa計算、arousal=v2ヒューリスティック継続。
データ:
  - genre(即時・手持ち): GTZAN を PANNs 埋め込みで再学習(fault split) → v2の43%と比較し転移効果を見る。将来FMA。
  - mood(要DL・有界): MTG autotagging_moodtheme サブセット(18,486曲/57タグ,CC)。全DL回避(low audioでも46GB)、
    download.pyで一部アーカイブのみ・数千曲に限定。57タグ→6クラス{aggressive,happy,sad,calm,dark,epic}へ写像
    (themeのみは除外、複数mood競合は解決規則)。PANNs埋め込み→moodヘッド学習でSTUB置換。
検証: genre/mood精度(artist非重複split)、v2比較、混同行列。データはCC・非コミット。

---

### GrooveStyleSelector: arousal を DEAM で本物にする
動機: arousal(動きの強度＋「激しい↔ゆっくり」主軸を駆動)は最重要かつMERで予測しやすい軸だが、v1以降ヒューリスティック(RMS×onset密度)のまま未検証。DEAMの正解で検証・強化。
データ: DEAM(1,802曲, valence+arousalの静的/動的注釈, CC BY-NC)。公式 cvml.unige.ch/databases/DEAM/ から直接curl(`DEAM_audio.zip` 1.25GB + `DEAM_Annotations.zip` 4.5MB)。`data/raw/deam/` 配下・gitignore・非コミット。Kaggle 経路は不採用(認証不要にした)。
方式: 凍結PANNs埋め込み(既存)→ `StyleRegressionHead`(共有 MLP + 属性別 Linear、`arousal`/`valence`)。凍結backbone＋属性別ヘッドの既存パターン踏襲。学習出力は DEAM の SAM 1..9 スケール、推論時に `deam.sam_to_unit` で 0..1 へ較正して `arousal_bucket()` へ。
ローダ: `groovebot.style.deam.read_static_annotations_many` が CSV ペア(`_1_2000.csv` + `_2000_2058.csv`、列頭 leading space を正規化)を読み、`MEMD_audio/<song_id>.mp3` を解決して `DeamRecord(audio_path, song_id, arousal, valence)` を返す。曲単位 split は DEAM に artist/album メタが無いため song_id 直接(seed 固定)。

#### v3 arousal 本物の数字 (2026-06-15、DEAM 1802 曲、CPU、song-disjoint 1262/270/270)

|         | val R²  | val RMSE | val r  | test R² | test RMSE | test r | 文献目安  |
|---------|--------:|---------:|-------:|--------:|----------:|-------:|----------:|
| arousal |  0.549  |   0.901  |  0.742 |  0.522  |    0.886  |  0.723 | R²≈0.6    |
| valence |  0.398  |   0.898  |  0.653 |  0.451  |    0.875  |  0.675 | R²≈0.4    |

(RMSE は SAM 1..9 スケール。pearson r は test 270 曲。`experiments/train_arousal_tl.py`、early stop @ ep26 patience 15。)

**ヒューリスティック vs DEAM 真値 (test, n=270)**
- 現行 `estimate_arousal()`(RMS×onset 密度、0..1)と DEAM 正解 arousal(SAM 1..9)の **Pearson r = 0.422**(unit↔unit でも同値、線形写像のため)。
- unit 尺度での RMSE = 0.247。ヒューリスティック平均 0.635 / 真値平均 4.76 (SAM ≈ 0.47 unit)。

判定:
- **ヒューリスティックは方向性は正しいが弱い**(r=0.42 → 説明分散率 ~18%)。0..1 内で +0.16 ほど過大に偏る傾向(0.635 vs 真値 0.47)。
- 学習ヘッドは r=0.723(説明分散率 ~52%)で **+34.4 pp の R² ゲイン**。
- 結論: ヒューリスティックは消さない(PANNs 無し / 軽量ループ用の default)、learned arousal_fn を opt-in の上位経路として `GrooveStyleSelector(arousal_fn=...)` 経由で差し替え可能にした(`make_panns_arousal_fn(backbone, head)` ヘルパ)。

較正: `deam.sam_to_unit(value, 1, 9) = (value - 1) / 8`(範囲外はクランプ)。これを通って `arousal_bucket()` に入る。DEAM 真値の SAM 平均が 4.76 ≈ unit 0.47 で `arousal_bucket()` の "mid"(0.33-0.66)範囲内、分布の歪みは小さく区分的較正は不要。

限界:
- 1802 曲・凍結 PANNs・ヘッド肥大なしで R²=0.52(arousal)/0.45(valence)。文献の arousal R²≈0.6 から **-0.08** は backbone fine-tune / Music Tagging Transformer 系で埋まる余地だが今は範囲外。
- DEAM 静的アノテは曲全体 1 ラベル。動的(秒単位)アノテは別軸で M2 以降の online arousal estimator にとって有用だが今回は静的のみ。
- AudioSet pretrain(PANNs CNN14)と DEAM の overlap は MTG/GTZAN より小さい想定だが厳密には未測。R² の絶対値は「凍結 embedding + 小 MLP」の上限としてのみ解釈。
- 配線: select は `arousal_fn: (audio, sr) -> 0..1` を受ける。デフォルトはヒューリスティック維持(後方互換)。

波及(任意): DEAM valence は test R²=0.451 で MTG 離散 mood (test 0.350) と直接比較できないが、連続感情価としては実用域。M3 で mood ヘッドを valence-soft へ置換する選択肢が再浮上(今回はスコープ外)。