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
1. ユーザが歌う → ロボがリアルタイムで拍に同期して動く。
2. テンションが上がる → 動きが大きく速くなる、画面に「Foo!!」等を表示。
3. 評価実験 → 同期条件 vs 非同期条件で、客観精度と主観評価を取得。

---

## 3. 要求仕様

### 3.1 機能要求 (Functional Requirements)
| ID | 要求 |
|---|---|
| FR-1 | マイク/ファイルから歌声を取り込む |
| FR-2 | 歌声からオンライン（causal）に拍位相・ダウンビート・テンポを推定する |
| FR-3 | 歌声から覚醒度(arousal)・感情価(valence)・エネルギーを推定する |
| FR-4 | (M3) 歌声から自己教師あり埋め込み (voice embedding) を抽出する |
| FR-5 | 上記の条件からロボットの関節目標値（ノリ）を生成する |
| FR-6 | 関節目標値を身体（シム/実機）に送り駆動する |
| FR-7 | 顔の感情表示・胸画面（波形/絵文字/テキスト）を描画する |
| FR-8 | 身体を設定で差し替えられる（MuJoCo/PyBullet/Pepper/NAO/自作機） |

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
 mic/file ─► AudioInput ─► (ring buffer)
                               │
            ┌──────────────────┼───────────────────┐
            ▼                  ▼                   ▼
       BeatTracker       ArousalEstimator     VoiceEncoder(M3)
       beat_pos,         arousal, valence,    embedding
       downbeat, tempo   energy
            └──────────────────┼───────────────────┘
                               ▼
                         GrooveContext  ──►  GrooveGenerator ──► JointCommand
                                                                     │
                                          ┌──────────────────────────┼────────────┐
                                          ▼                          ▼            ▼
                                    RobotBackend            FeedbackRenderer   (logging)
                                  (MuJoCo/…/RealServo)      (face/screen)
```

### 4.3 コンポーネント一覧
| コンポーネント | 責務 | 入力 → 出力 | フェーズ | 実装候補 |
|---|---|---|---|---|
| AudioInput | 音声取得・バッファリング | mic/file → frames | M0 | sounddevice / soundfile |
| BeatTracker | オンライン拍追跡 | frames → beat_pos, downbeat, tempo | M0/M1 | SingNet / mjhydri |
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
    embedding: "np.ndarray | None" = None   # M3 用 voice embedding

@dataclass
class JointCommand:
    targets: dict[str, float]   # joint_name -> radians（URDF の JOINT_NAMES）
```

### 5.2 モジュールインターフェース
```python
from typing import Protocol

class BeatTracker(Protocol):
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
> 注: 現行コードの `GrooveController.compute(beat_pos, energy)` は M1 簡易版。
> M3 へは `generate(ctx: GrooveContext)` に一般化する（このファイルだけ差し替え）。
> 全ての `JointCommand.targets` は出力直前に URDF 可動域へクランプする（NFR-4）。

---

## 6. リアルタイム・並行設計

### 6.1 スレッド構成
| スレッド | 周期 | 役割 |
|---|---|---|
| Audio | コールバック | マイク取得 → ring buffer へ書き込み |
| Perception | 可変（重い） | BeatTracker / ArousalEstimator / VoiceEncoder |
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
タイミングは BeatTracker（声から頑健に取れる拍位相）が支配し、GrooveGenerator は
スタイル・質感のみを担う（設計リスク#2への対策）。

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
| M0 | 歌声/鼻歌で拍追跡が壊れる箇所を把握 | AudioInput, BeatTracker |
| M1 | メトロノーム＋手付けノリで端到端を通す（**現行コード**） | Orchestrator, GrooveGenerator(規則), RobotBackend |
| M2（必達） | 声→テンション、画面フィードバック | ArousalEstimator, FeedbackRenderer |
| M3（目標） | 学習モデルでノリ生成 | VoiceEncoder, GrooveGenerator(モデル) |

---

## 10. テスト・評価

### 10.1 ソフトウェアテスト
- 単体: GrooveGenerator 出力が全フレームで可動域内（プロパティテスト）。BeatTracker を
  注釈付き音源に対して F値で検証。
- 結合: 端到端ループがレイテンシ予算内で動作。
- 身体非依存性: 同一の脳が MuJoCo / PyBullet 双方で動く回帰テスト。

### 10.2 研究評価
- **客観（同期精度）**: 被験者にイヤホンでクリックを聴かせて歌わせ、その拍グリッドを正解として
  ロボ動作のズレを F値 / CMLt / AMLt で測定。
- **主観（HRI）**: 同期条件 vs 非同期条件（わざと拍を外す）で、楽しさ・「合ってる感」・
  エンゲージメントを比較。Keepon 型の対照実験設計。

---

## 11. リスクと対策
| リスク | 対策 |
|---|---|
| スタイルのドメインギャップ | ターゲットを「楽しい系」に固定し AIST++ と一致させた（解消済み） |
| 声条件 vs 音楽条件のズレ | タイミングは拍チャンネルに任せ、モデルは質感のみ（§6.3） |
| 鼻歌が音声SSLの土俵外 | 歌唱 vs 鼻歌のアブレーション、音高/エネルギー特徴を併用 |
| ハードが間に合わない | 身体非依存設計。評価は既存ロボ（Pepper等）で代替可能 |

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
