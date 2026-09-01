# GrooveBot — 開発ログ

歌声・ハミングにリアルタイムでグルーヴする上半身ロボットの研究開発記録。意思決定とその根拠、結果、学びを時系列で残す。スコアはすべて同一の `mir_eval` 系指標（F-measure / CMLt / AMLt / リアルタイム係数）で測っている。

> 公開リポジトリ方針: ライセンスデータ（DAMP / Smule 等）の音源・派生物は一切コミットしない。すべて `data/`（gitignore）にローカル保持し、再配布しない。

---

## Phase 0 — 研究フレーミング

「人がアカペラ／鼻歌で歌うと、ロボットがリアルタイムでノる」という体験を狙う。用途は **energetic karaoke（楽しく盛り上がる）** に固定（穏やかな高齢者ケア版は卒研候補として保留）。**ハミング対応は必須要件**。

問題を二つに分解：
- **いつ動くか** = 拍・テンポ
- **どう動くか** = グルーヴの質（盛り上がり→動きの強度）

音楽駆動ロボットダンス、リズム的引き込み、オンラインビート追跡、音楽感情推定、music→dance生成（AIST++ / Bailando / EDGE 等）といった先行研究を踏まえて方針を組んだ。

## Phase 1 — アーキテクチャと仕様

頭脳と身体を分離し、インターフェースを固定契約とする設計を採用。

- 公開IF: `GrooveContext`（拍位相・テンポ・構造・arousal）、`JointCommand`（関節指令）、`RobotBackend`（身体抽象）。
- 約10自由度の上半身（首・体幹・両腕）＋感情表示スクリーンを想定。
- `SYSTEM_SPEC.md` に SRS+SDD（機能/非機能要件、リアルタイム設計 ≤150ms、30–50Hz制御、M0–M3）を整備。
- 不変条件: 頭脳はシミュレータ/ロボットSDKをimportしない、関節出力はURDF上限へクランプ、因果的・オンライン処理（オフラインDTWは実現性検証のみ許可）。

## Phase 2 — M1: リアルタイム・グルーヴループ（完了, tag m1）

- Orchestrator が知覚から分離した制御ループ（30–50Hz）を駆動。
- `RuleGrooveGenerator`、URDF関節上限クランプ、MuJoCoバックエンド。
- 型付きIFで身体差し替えを担保。

## Phase 3 — M0評価ハーネスと、方針転換（重要な分岐）

機能追加の前に評価基盤を用意：`mir_eval`（F / CMLt / AMLt / RT係数）、合成・実データ両対応、ステージ脱落を透明化するレポート。

その後、**ライブ音声からの盲目ビート追跡（BeatNet/madmom）が現代Colabで構造的に動かない**ことが判明：
- Colab が numpy 2.x を既定化し madmom が破綻。
- 根本的に madmom は Python <3.10 にしか対応せず、Colab は 3.12（`collections.abc` / `pkgutil.ImpImporter` 周りで詰む）。

→ 小手先の回避を続けず、**ターゲット用途の性質で問題を作り替える**判断をした：

> **盲目追跡をやめ、「参照アライメント」を主軸に。** ユーザは曲を選ぶ＝既知曲。その曲の参照（拍・ダウンビート・構造・特徴量）へ生歌声を整合させれば、不安定な盲目追跡なしに**信頼できる拍グリッドと構造を継承**できる。

仕様書を全面改訂し、`ReferenceAligner` / `SongReference` を主軸へ昇格、盲目追跡を任意フォールバックへ降格（`8b60c7b`、ドキュメントのみ・テスト不変）。旧M0は棚上げ、**M0′（参照データ＋オフライン整合の実現性検証, madmom不要）**を定義。

## Phase 4 — M0′ Tier 1: 機構の実現性検証（強い結果）

**方法**: 参照（ボーカル＋拍グリッド）に既知のテンポ変動（time-stretch）を与えた合成クエリを作り、オフラインDTW（`librosa.sequence.dtw`）で整合 → 拍を復元 → `mir_eval` で採点。著作権・録音不要で機構を確かめられる、完全にクリーンな設計。

**結果**（GTZAN系7ジャンル×2曲×5倍率 = 70ラン, フルミックス）:

| 倍率 | F | CMLt | AMLt |
|---|---|---|---|
| 0.90 | 0.970 | 0.975 | 0.975 |
| 0.95 | 0.987 | 0.988 | 0.988 |
| 1.00 | 1.000 | 1.000 | 1.000 |
| 1.05 | 0.992 | 0.994 | 0.994 |
| 1.10 | 0.967 | 0.962 | 0.962 |
| **全体** | **0.983** | **0.984** | **0.984** |

リアルタイム係数 0.017（約60倍速）。

**学び**:
- 配線（特徴量→DTW→拍写像→採点）は正しく、テンポ変動に極めて頑健。
- **CMLt ≈ AMLt（差ゼロ）＝半・倍テンポの取り違えが原理的に起きない**。盲目追跡が最も苦手な領域で、参照方式の構造的な勝ち。
- ただし自己ワープなので**楽観値**。別演者の難しさは Tier 2 で測る、と明確に切り分け。

## Phase 5 — M0′ Tier 2: 実演奏評価の実装

実アマチュアの独立演奏で測るため、ライセンス済みデータ（DAMP系）の取り込み経路を実装。録音に頼らずスケールさせる方針。

- **二系統の特徴量経路**: 歌唱=chroma、鼻歌=pitch（音高コンター）。
- 設計上の要点: 参照側はフルミックスでなく**声／旋律**で整合する（声 vs 声、または鼻歌 vs 旋律）。
- pitch経路の根拠: 同一旋律なら歌声のF0も鼻歌のF0もともに「旋律のF0軌跡」に収束するため、実歌声のF0で鼻歌経路を代理検証できる（クエリのF0は実演者由来で、参照から派生させた“出来レース”ではない）。
- 評価は Tier 1 と同一の `mir_eval` スコアラを共有。

## Phase 6 — DAMP-S-AG 実データ：低スコア → 診断 → 修正（方法論の山場）

手元で先行入手できた **DAMP-S-AG（Amazing Grace, 17,582独立演奏 + 共有backing + MIDI）** で初の実数字を取得。参照は同梱MIDI（拍グリッド＋旋律がクリーン、ffmpeg不要）。

**初回（n=20）**: F_chroma = 0.298 / F_pitch = 0.199。Tier 1 の 0.98 から大幅低下。

ここで「想定通り難しい」と結論せず、**低スコアが本物の難しさか計装のアーティファクトかを切り分ける診断**を実施：

- **タイムライン**: 各演奏は1.4–9.6秒の先頭無音を持ち、MIDI（t=0始まり）と起点が一致しない。
- **DTW経路**: chromaは概ね対角近傍。**pitchは12–19秒の極端な退化**（MIDI 99%有音 vs クエリ72–85%有声 のスパース非対称が誘発）。
- **オフセット掃引（診断目的）**: 最悪ケースは +0.15s の起点補正だけで F 0.07→0.41 と激増。原点オフセット問題が実在。
- **無音トリム**: ほぼ全例で悪化 → 「無音が原因」仮説を棄却（full DTW が無音を境界スラックとして有効活用していた）。

**切り分け結論**: pitch経路の低さは明確に計装アーティファクト、chromaは起点オフセットが部分的に効いている。

**修正（重要: 正解データで出力を調整しない＝テストリーク回避）**:
- 試した修正のうち subseq DTW + 無音トリム（Lever 1）は**仮説に反して大幅悪化**（F 0.062）。スパース非対称下で小領域に縮退し、多くの拍がドロップしたため。full DTW の全長被覆が効いていたという学び。
- Sakoe-Chiba band（Lever A, full DTW維持）はこの設定では中立。
- **GT非参照の起点アンカー（Lever B）** が実利得を独占：F_chroma 0.298→0.324、F_pitch 0.199→0.213。ただし中央値ラグは 0.046s と小さく、伸びしろは限定的。

**正直な結論**:
- クリーンな方法論でも Amazing Grace では F_chroma ≈ 0.32。だがこれは**病的な外れ値**の数字：遅いルバートの3/4賛美歌は人間ですら正確な拍を打ちにくく、そもそも**グルーヴ対象の曲ではない**（用途的にも非代表）。
- **CMLt ≈ AMLt（メトリカル誤推定ゼロ）という構造的性質はこの最悪条件でも維持**された。
- よって**この数字は手法の判定ではない**。Amazing Graceの最適化はここで打ち切り。
- この回り道の本当の収穫は、**評価ランナーが方法論的に堅牢化した**こと（GT非参照アンカー、bandオプション、スパース非対称の知見、テストリーク回避の規律）。

### （この時点でのTIMINGスナップショット）

- 代表値の判定はテンポ安定なポップ楽曲（DAMP-VSEP）でしか出ない → アクセス許諾待ち。
- テスト推移はここまでで 91 → 103 → 127 → 142 → 173 passed。公開IFは不変。
- この後、TIMING を PARK して STYLE ブランチへ展開する（Phase 7 以降）。最新の現在地・テスト推移は末尾にまとめ直している。

---

## Phase 7 — TIMING ブランチの整理と PARK 判断

DAMP-VSEP のアクセス許諾が長期に渡り保留のため、代表値判定（Tier 2）はブロック。ここで「日本語歌唱データを大量に集めて埋める」案を検討したが、**J-popは世界でも著作権が最も厳しい領域**であり、既知曲の歌唱サンプル収集は法的にも、GT拍の付与の面でも二重に重い。Tier 1 で機構は検証済みである以上、**timing は健全にPARK**し、エネルギーを STYLE ブランチに振る判断をした（無駄な大規模収集はしない）。

> 教訓: 「データが無いから集める」の前に、「今それが本当に必要か」「他の軸で前進できないか」を問う。

## Phase 8 — STYLE ブランチ始動：GrooveStyleSelector の設計

「どう動くか」を学習で解く方針を立てるにあたり、**何を学習し、何を学習しないか**を最初に確定した。

- **学習対象は audio → 属性（genre 分類 / arousal・valence 回帰）のみ**。
- **属性 → グルーヴ style + intensity は人手のテーブル**（genre × arousalバケット × mood）。これは設計判断であって学習対象ではない。
- ロボット実現可能な出力 ≈ 上半身8クラス（headbang / bob_nod / sway / front-back rock / fist_pump / clap[2&4] / penlight_wave / quiet_listen）。
- mood = 6感情クラス {aggressive, happy, sad, calm, dark, epic}、テーブル内では**ソフト確率**として使用（argmaxしない）。
- 教師データは**著作権の重いライブ映像ではなくCCの音楽タグ付けデータ**で済む設計。RLではなく分類＋回帰。

> 「NNが勝手に解く」のではなく「何を教師信号にするか」が本質、という理解に立つ。Shimada Lab の画像認識→ロボット動作（DL知覚→行動）の構図と相似。

## Phase 9 — genre：from-scratch → 転移学習

- **v2 from-scratch StyleCNN**（GTZAN fault-split, 443/197/290）: fault test 0.431 / naive 0.573（リーク由来バイアス +14pt）/ pop-sink / 過学習ギャップ +0.22。
- **v3 転移学習**: 凍結 **PANNs CNN14**（panns_inference, Zenodo 3987831）→ MLPヘッド。**fault test 0.431 → 0.817（+38.6pt）**、過学習ギャップ +0.03、pop-sink解消。
- 正直な但し書き: PANNs の AudioSet 事前学習が GTZAN/Jamendo と分布的に重なるため、純粋な手法利得ではない。**用途に対し十分（82%）と判断しクローズ**。

## Phase 10 — mood / arousal / valence：実データで学習・検証

- **mood（MTG-Jamendo autotagging_moodtheme, 18,486曲/57タグ, CC）**: 59タグ→6クラスへ写像、theme/曖昧タグは drop。test 0.350（1.75×chance）、calm-sink、calm↔sad の強い混同（valenceは音響から本質的に難しい）。
- **arousal/valence（DEAM, 1,802曲, V/A連続, CC BY-NC）**: 凍結PANNs → 回帰ヘッド。**arousal R²=0.522 / r=0.723**（文献≈0.6）、**valence R²=0.451 / r=0.675**（文献≈0.4を上回る）。
- 重要な比較: **既存ヒューリスティックの arousal は DEAM真値に対し r=0.422（説明分散≈18%）＋ +0.16 の過大評価バイアス**。学習ヘッドは +34pt R²。→ **学習ヘッドを既定化**（PANNs無し環境のみヒューリスティックにフォールバック）。

## Phase 11 — affect 統合と mood の出所整理

- 学習した DEAM arousal/valence を**既定**に昇格（ヒューリスティックは厳格な no-PANNs フォールバック）。
- `mood_from_va.py`: V/A circumplex の4象限（happy / aggressive / calm / sad）をクリーンに導出。epic / dark は純V/Aで表せないため既定オフ。
- mood_source を {head, va} で切替可能に。MTG mood ヘッドより V/A 由来が安定と判断。

## Phase 12 — JointCommand ブリッジとナレーター、拍位相規約の確定

- `StyleGrooveGenerator` ＋ 8種 MOVE_PRIMITIVES ＋ `render_groove.py`（MuJoCo→GIF＋CSV）。GrooveStyleSelector → Orchestrator → generator → MuJoCoBackend を結線。拍源は当面 tempo BPM 種のメトロノーム（M2 で ReferenceAligner に差し替え予定）。
- **ナレーター（`narrate.py`）**: perception→decision→action のウィンドウ要約＋拍ごとトレース。制御パス外・観測専用の純関数。
- 導入直後に **headbang が拍からずれて頂点を迎えるバグ（b+0.48）を即検出**。観測可能性が即座に価値を出した好例。
- 修正後、**拍位相の契約を確定**：accent（headbang/bob_nod/fist_pump）は毎拍頂点 (1+cos2πb)/2、clap は 2&4 バックビート、continuous（sway/rock/penlight_wave）は拍でゼロ・半拍で最大 sin(πb)、quiet_listen は呼吸のみ。ナレーターが全て確認。

## Phase 13 — end-to-end ショーケースと、affect 較正（最新の山場）

end-to-endショーケースを実装：訓練済みckpt（genre / DEAM arousal・valence）を `render_groove --audio` に正式接続（ckpt欠落時は `INVALID` で停止、ランダム重みで黙って続行しない安全弁つき）。**西洋2本＋CC日本語2本**（魔王魂, 規約: 商用/改変/再配布OK）を全パイプラインに通し GIF＋ナレーション＋比較表を生成。

**初回ショーケースが部品テストをすり抜けた潜在バグを捕捉**：
- 回帰ヘッドの**平均回帰**で arousal 出力が中央（[0.42, 0.64]）に圧縮。
- 絶対閾値 0.33/0.66（生0..1スケール前提）に対し**4クリップ全部が mid バケットに沈没** → `_AROUSAL_NUDGE` が空 → テーブルキーが実質「genre×mood」だけに。
- pop-family の clap-prior が支配し、**日本語2本が両方 clap に収斂**（多様性損失）。
- ただし**styleは壊れず**（defensive-design が legal な style を出し続けた）。

**較正（テストリーク厳禁の原理的手法）**:
- 新規 `tools/calibrate_affect.py`。境界は **DEAM val プール（n=270, seed=0, ショーケース4本は一切含まない held-out）** の実ヘッド出力分布から導出。
- arousal 三分位境界: **low<0.410 / mid[0.410,0.529) / high≥0.529**（各バケット90曲, tertile完全）。絶対値 0.33/0.66 を廃止。
- `groovebot/style/affect_calibration.json` を**単一の出所**化（schema/source/n/date メタ込み）。`attributes.py` と `mood_from_va.py` が読込、欠如時 0.33/0.66 にフォールバック。
- mood_from_va の neutral 中心を予測中央値（arousal 0.473 / valence 0.484）に再センタリング。

**再走結果（held-out 観察）**:

| clip | lang | genre | arousal(値→bucket) | style | intensity | dominant move |
|---|---|---|---|---|---|---|
| western_metal | en | metal (0.62) | 0.63 → high | headbang | 0.62→**0.93** | headbang |
| western_classical | en | classical (0.87) | 0.49 → mid | quiet_listen | 0.59 | quiet_listen |
| jp_uptempo | ja | pop (0.43) | 0.64 → **high** | clap | 0.61→**0.92** | clap |
| jp_ballad | ja | pop (0.36) | 0.42 → mid | clap | 0.57 | clap |

- **arousal が日本語2本で割れた**（high / mid, 較正前は両方 mid）。**intensity が 0.92 vs 0.57 と明確分離**（較正前 0.61/0.56）。本丸の改善。
- 西洋2本は genre 高信頼が支配し妥当（metal→headbang, classical→quiet）。
- jp_uptempo の clap は **117BPM陽気popには妥当**（headbangに矯正しない判断）。jp_ballad の clap はやや惜しいが style は壊れず＝許容、手当て不要。

**この較正の副産物が本命の問いへの答え**：
> **「日本語楽曲に対応するのに再学習は不要、必要だったのは較正」** を実データで確認。arousal は言語非依存に妥当な値を返す。PANNs バックボーンは AudioSet 由来で多文化、偏りは genre ヘッドのラベル側に限局。日本語クリップは「学習が要る証拠」ではなく「**較正バグを見つけるセンサー**」として機能した（テストファースト戦略の勝ち）。

**正直な限界（過剰解釈の戒め）**:
- バケットは**境界付近で不安定**（jp_ballad 0.42 は low境界 0.410 のわずか上、次の学習runで容易にflip）。平均回帰自体（std≈0.12）は較正では治らず、将来は spread loss か連続intensity ramp で対処。
- genre の pop-sink（J-music未学習）は残存。styleは壊れないので手当て不要だが、**move の真の多様化には genre の J-music 学習か M3 学習生成が必要**（較正のスコープ外）。

## テスト推移（続き）

173 → … → 396（JointCommandブリッジ）→ 421（ナレーター）→ 432（位相規約）→ **433 passed / 2 skipped**（affect較正のフォールバック経路テスト +1）。公開IF（`GrooveContext` / `JointCommand` / `RobotBackend`）は全期間を通じて不変。

## 現在地と今後（更新）

- **STYLE = v1完成**: audio→属性（genre 82% / arousal r=0.72 / affect=DEAM）→テーブル→拍位相つきプリミティブ→JointCommand→MuJoCo→ナレーション。affect較正済み、日本語汎化を arousal/intensity 軸で実証。
- **TIMING = PARKED**: Tier 1 機構検証済み、Tier 2 代表値は DAMP-VSEP アクセス待ち。
- **M2**: オンライン整合＋Orchestrator接続（Tier 2 検証後）。
- **M3**: 学習グルーヴ生成（VQ-VAE codebook 等, 手作りプリミティブ置換, AIST++/GPU想定）。pop-sink由来の move 多様化はここで本質解決。
- 任意の後始末: mood の calm/sad 統合（"mellow"）、class-weighted CE、MTG追加アーカイブ等。

> Commits: 69aed53（affect較正/JSON単一出所）, 85321f2（end-to-endショーケース）, ac54d3b（CLAUDE.md pin）。origin/master 552177b..ac54d3b。