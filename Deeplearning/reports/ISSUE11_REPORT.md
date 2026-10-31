# Issue #11 実装・再現手順

対象: [Issue #11](https://github.com/tatsukawa0353/Birdsong_tatsukawa/issues/11)。録音に依存しない方法とコードだけをここに記す。

**結果の詳細はローカル限定の `Deeplearning/private_runs/issue11/REPORT.md`。** 冒頭にまず見るべき5ファイルを記載する。原音、評価数値、図、特徴量、生成音、重み、制御、モデル比較の結論はすべてそのフォルダ内に保存し、この公開文書へ転載しない。

## 実装

`phase2.model_configuration()` がJSONの `loss_kind` / `reward_timing` / `gamma` を既存の `ModelConfig` へ渡す。旧設定にこれらのキーがなければ、従来どおりbaseline・context・gamma 0.99を使う。両報酬で新全体損失を監査計算し、評価の `error` は従来の全体誤差、`objective_error` はbalanced全体損失を保持する。波形・旧step報酬を変更する監査ではない。

比較用の [phase2_td3_baseline.json](../configs/phase2_td3_baseline.json) と [phase2_td3_balanced_aligned.json](../configs/phase2_td3_balanced_aligned.json) は、損失と時間集計の選択だけを変える。`objectives.py` の式・係数、物理式・RK4刻み・遅延、描画設定へ変更を加えない。

`issue11.py` は [issue11.json](../configs/issue11.json) の固定手順を実行する。TD3を先に、SACも同じ条件で比較する。主条件はpower8、Case A/B/C、seed 0/1/2、0.35秒、10 ms制御、同一の音声前処理・目標・ゲイン・ネットワーク・評価間隔。Case間は3/4/5自由度で、Case Bの右外力は実適用履歴で検証する。

各手法で1050stepの診断を先に行う。Case内の両報酬・全seedに訓練中発散と評価失敗がなければ、そのCaseの両報酬・全seedを4200stepで新規学習する。スコアの良いseedだけを延長しない。独立したプロセスのRNGを使用し、学習の投入順と集計順を固定する。

評価は全0.35秒の旧誤差・新損失、区間別AC/帯域RMS、初期過渡除外後の持続率、全積分点の左右運動統計、目標活動時の欠落と弱音部の過剰を分ける。目標のみの活動マスクと補助的な低エネルギー診断は用途を明示し、常時発声や白いスペクトログラムだけを成功判定にしない。SAP/SE/ACI等を新しい報酬項目へ追加しない。

## 再実行

リポジトリ直下から実行する。ローカル利用許諾はユーザー確認済みであり、フラグはその確認を記録する。

```bash
export MPLCONFIGDIR=/tmp/birdsong-mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
Deeplearning/.venv/bin/python -m Deeplearning.issue11 all --local-permission-confirmed
```

段階別は `prepare`, `train`, `report`。出力の既定は `private_runs/issue11/`。出力先を変更してもprivate_runs外への保存は拒否し、シンボリックリンク解決後にも確認する。録音の読み込み前に権限確認と出力制約を検査する。

`prepare` は共通ゲインと目標・コード・設定のハッシュをローカルに記録する。`train` はコアコードとキャッシュを照合し、同設定の完了runを再利用する。違う条件は別の `--output Deeplearning/private_runs/...` を指定する。1050と4200は別ディレクトリで、過去Issueの結果を上書きしない。`report` は保存結果の再計算・ペア比較・並列図・ローカルレポートを生成し、学習し直さない。

単一条件を `phase2.py` で実行する場合も出力先を分ける。

```bash
Deeplearning/.venv/bin/python -m Deeplearning.phase2 --target Deeplearning/sample_song/14412_0.35s.wav --config Deeplearning/configs/phase2_td3_baseline.json --output Deeplearning/private_runs/td3_baseline --local-permission-confirmed
Deeplearning/.venv/bin/python -m Deeplearning.phase2 --target Deeplearning/sample_song/14412_0.35s.wav --config Deeplearning/configs/phase2_td3_balanced_aligned.json --output Deeplearning/private_runs/td3_balanced_aligned --local-permission-confirmed
```

## 検証

[issue11_validation.txt](issue11_validation.txt) に合成入力だけを用いた43件のテスト結果を保存する。旧Phase 2の波形・報酬不変、JSON切替、共通ゲイン・スケール、A/B/C行動数、Bの実適用右固定、alignedの割引リターン、目標のみの活動判定、Case単位の追加予算判定、公開出力とシンボリックリンク経由の漏出拒否を含む。既存のC++・RK4・遅延・表示の回帰も実行する。

```bash
Deeplearning/.venv/bin/python -m unittest discover -s Deeplearning/tests -v
```

学習結果の再計算と図の目視による解釈・採用判断はローカルレポートに記す。鳴管は1つの器官で左右に2つの音源を持つ。Case間の自由度や外力制約が異なるため、片方の指標や一つの音節だけから普遍的な優劣を先取りしない。
