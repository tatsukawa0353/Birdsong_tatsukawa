# 研究成果物・フォルダ構成の整理結果

2026-11-26 JST。着手時HEADは `259e7da8923de041041043ca64ac6a50abed2c1d`、Gitはクリーンだった。ホストのPython/鳴管/コンパイラプロセス確認で進行中の計算はなく、出力先を移しても実行中ジョブに影響しない状態で実施した。commit/push、Git履歴の変更、強化学習、物理音生成、oracleの再実行は行っていない。

## 整理前後

| 指標 | 整理前 | 整理後 |
|---|---:|---:|
| results直下のフォルダ数 | 22 | 4 |
| 全階層のフォルダ数（results自身を除く） | 1,528 | 1,515 |
| ファイル参照数（相対リンクを含む） | 8,290 | 8,047 |
| 通常ファイル数 | 8,290 | 6,465 |
| 相対symlink数 | 0 | 1,582 |
| ファイル本体の総容量 [byte] | 1,491,128,021 | 1,265,287,704 |
| 実割当容量・ファイル部分 [byte] | 1,508,630,528 | 1,279,795,200 |
| 画像の参照数（リンクを含む） | 671 | 384 |
| 通常画像ファイル数 | 671 | 344 |
| 画像本体の総容量 [byte] | 161,897,562 | 96,402,870 |

総容量は1491.13 MBから1265.29 MBへ、**225.84 MB（15.15%）減少**した。移行記録・監査・新索引の追加分を含む純減である。参照パス数も243減少した。トップレベルは22から4分類に集約した。目的別の親階層を追加しつつ、空になった旧階層を除去し、全階層のフォルダ数も削減した。

容量は通常ファイルの長さとsymlink自身の長さを合計し、リンク先の重複容量を二重計上しない。実割当はファイルのst_blocks×512で、ディレクトリ/ファイルシステムの管理領域と.gitを含まない。リンクを実体化してコピーすると重複容量が復活する。画像数もリンク込み/通常ファイルを併記し、削減を過大に数えない。

削除した再生成可能画像は **287枚、58.50 MB**。別途、同一SHA256/長さの重複バイナリ **1,582個、173.52 MB** を相対symlinkに置き換えた。削除した本体は合計1,869個・232.01 MBだが、リンク置換分は参照パス数の削減には算入しない。

[機械可読な容量比較](capacity_comparison.json) / [全旧フォルダ別の比較](folder_comparison.csv) / [削除本体のpath・bytes・SHA256・理由・正本](deletions.csv) / [整理前全ファイル](inventory_before.json) / [整理後索引](inventory_after.csv)。単なるarchive移動ではない。

## 移動・統合と主要成果

| 旧results内の保存先 | 新results内の保存先 |
|---|---|
| `phase0_fixed_target_20261108` | `Phase0/synthetic` |
| `phase1_constant_020s` | `Phase1/CaseA/parameter_dimension` |
| `phase1_3d_domain_diagnostics` | `Phase1/CaseA/domain_diagnostics` |
| `phase1_restricted3d_weights` | `Phase1/CaseA/reward_weights/restricted_domain` |
| `phase1_caseA_3d_network_width` | `Phase1/CaseA/network_width` |
| `phase1_caseA_sac128_reward_weights` | `Phase1/CaseA/reward_weights/sac128` |
| `phase1_caseA_sac128_actor_lr` | `Phase1/CaseA/learning_rate/actor` |
| `phase1_caseA_sac128_critic_lr` | `Phase1/CaseA/learning_rate/critic` |
| `phase1_caseA_sac128_stability_diagnostics` | `Phase1/CaseA/stability_diagnostics` |
| `phase1_caseA_wide_local_exploration` | `Phase1/CaseA/exploration_strategy` |
| `phase1_caseB_self_target_020s_sac128` | `Phase1/CaseB/target_diagnostics/initial_stop` |
| `phase1_caseB_self_target_020s_methods` | `Phase1/CaseB/target_diagnostics/high_left_force` |
| `phase1_caseB_self_target_020s_methods_f0lr_scan` | `Phase1/CaseB/target_diagnostics/low_forces` |
| `phase1_caseB_self_target_020s_3targets_3d4d` | `Phase1/CaseB/3targets_3d4d` |
| `issue9` | `reference/phonation_domain` |
| `issue10` | `Phase1/CaseA/reward_diagnostics` |
| `issue12_synthetic` | `reference/activity_mask` |
| `diagnostics` | `reference/implementation_diagnostics` |
| `server_smoke_20261113T190418+0900` | `reference/environment_checks/cpu` |
| `server_smoke_gpu_20261113T194403+0900` | `reference/environment_checks/gpu` |

[全フォルダのIssue/研究目的/条件参照/主要結果/容量/利用先](folder_inventory.csv)、[全旧階層の件数](directory_inventory_before.csv)、[実装・README/REPORT/設定からの参照](references.csv)を保存した。過去の空の作業ディレクトリは除去した。

Case Aは次元数、発声領域、幅、報酬重み、Actor/Critic学習率、安定性、探索戦略ごとに分け、判断根拠となるREPORT・数値・履歴・モデル・Target・主比較図を保持した。Phase0の固定合成Targetと報酬/学習率比較も保持する。発声マップ・活動マスク・実装/環境確認はreferenceへ分類した。

Case Bは[物理診断の統合REPORT](../../Phase1/CaseB/target_diagnostics/REPORT.md)と[最新54runのREPORT](../../Phase1/CaseB/3targets_3d4d/REPORT.md)を入口にした。①f0_l=3.0e5,f0_r=1.0e7が0/9、②4.9e4,1.0e7が9/9、③4.9e4,4.9e4が9/9という判定を27行のCSVへ集約し、27条件共通色軸図、左右ラビア/source/output、減衰、詳細生データを保存した。初回/①の学習未開始の理由は当時のREPORT・未開始一覧とともに維持する。

旧b3_c00の9runは予備結果として区別し、設定、結果数値、履歴、主要考察、原データ/モデルを保持した。追加のTarget/endpoint描画バリエーションを削除し、主学習/Final/Selected/物理図を残した。最新54runの主比較図13枚、3Target、全学習済み推論モデル、学習/評価履歴、各評価役割の生音と主要生成WAVは保持した。T1/T2の候補b2_c10/b3_c10は最新preflightを正本とし、原NPZのhash・全配列・dtypeが一致することを検証した。WAVも整理前hashを保持し、異なる聴取gainの音を無条件に同一視しない。

削除したのはCase Aのseed1/2個別スペクトログラム/包絡、旧報酬診断の個別図、予備b3_c00の不要な描画バリエーション、終了済み環境smokeの画像等である。seed0代表図・集約学習/制御図・全seedの元波形/履歴/描画コードは保持した。重複はNPZ/WAV/model/PNGを完全なbyte一致で判定し、固有の科学データは削除していない。

削除を保留した固有モデル/NPZは **192ファイル、45.23 MB**。[個別一覧と理由](deferred_deletions.csv)。旧b3_c00、smoke、中断履歴のモデル/生データであり、監査再現と再計算コストを優先した。これらは主要結果とは分けて扱い、無条件の永久保存を新たに要求するものではない。

## 参照と管理ルール

Python 94ファイル、文書58ファイル、現行設定1ファイルと.gitignoreの保存先/参照を修正した。[変更一覧](reference_changes.json) / [原コードと現在コードの差分](source_path_style_changes.patch)。実験内のmanifest/hash/実行コードsnapshot/計算時PLANと数値表は原本のままにした。歴史的pathはresult_paths.resolveで変換し、sourceの変更はbefore→after SHA256を明示した移行検証で確認する。未記録のsource変更は許容しない。

新しい[results索引](../../README.md)と[AGENTS](../../../AGENTS.md)に、目的/実験IDによる管理、重複確認、終了時整理、主要結果と再現性の保持、条件変更履歴、共有描画のルールを定めた。共通[plot_style.py](../../../plot_style.py)はSciencePlots、STIXGeneral/数式cm、46/30/29pt、4pt実線、グリッドなし、平滑化/間引きなしを設定し、Issue17/最新Issue27の主描画と新しい保存波形レンダラから利用する。既存の全画像は再描画していない。

GitHubはIssue1〜27の本文を読み、[45種類の旧パスと現在の有無](github_issue_path_details.csv)、[移行対象34組のIssue/パス対応](github_issue_paths.csv)、[取得本文](issues_snapshot.json)を保存した。Issue本文の更新やコメントは実施していない。remote自体も未変更のため、branchパスのリンク切れは将来の移行commit反映後に生じる箇所として報告する。固定commitの歴史的URLは書き換えない。

## 実施した検証と実施しなかった計算

- 元成果8,003参照の保持hash、1,582相対リンク、287削除、原物理/音響コードとlibrary hashを検査し成功。元の数値CSV/履歴/固有モデル/生データも同じ検査に含む。
- 最新54run・221,184学習step・1,782独立評価の保存記録を確認。162モデルを読込み、Target×次元×手法の18 Final方策の推論行動が保存値と完全一致した。検査中のBirdModel.advanceは禁止し、新規物理音は0。
- T1/T2の原NPZと全配列の一致を確認。元のoracle=0の記録は保持し、oracleの生成し直しは行っていない。
- 現行9研究のready確認が成功。初回Case Bの非持続Targetによる停止gateは閉じたまま維持。188 Pythonファイルの構文確認、パス移行の2 unit test、git diff --checkが成功。
- T1保存音から共通書式のスペクトログラム/波形を/tmpへ再描画し、スペクトログラムの表示を確認。全画像の再描画、学習、長い物理診断、過去監査の物理再生部分は未実施。
- ローカルMarkdownの新規リンク切れは0。従来から存在するprivate_runs参照3箇所は維持し、下記へ明記。

[保持監査](retention_audit.json) / [モデル検査](model_loading_audit.json) / [Target移行](target_transfer_audit.json) / [ready結果](ready_after.json) / [リンク検査](link_audit.json) / [再描画検証](redraw_validation.json) / [unit test](path_unit_tests.log) / [総合検証](verification.json)。現在の読み取り専用audit・再描画コマンドは[利用手順](README.md)を参照。

## 残存する問題・利用上の区別

- REPORT.mdとreports/ISSUE9_REPORT.mdのprivate_runsへの3参照は整理前から非公開データが存在しない。今回の移動が原因ではなく、公開の代替データを作成していない。
- Issue20の指定results/phase1_caseAB_constant3d/は整理前から存在しない。実施済みと補完せず、Issue本文の保存先と実在成果を区別した。
- 過去のfull auditやreport/freezeには「当時の画像全保存/元source一致」や物理再生成を要求するものがある。現在の保持確認には移行auditを使う。過去の完全な監査/描画の厳密な再現は旧HEADの別checkoutで行う。現成果上で旧freezeを実行して歴史的hashを再保存しない。
- 過去の補助レンダラには当時の書式も残す。将来の研究図は共有contextで作成する。今回すべての過去図・補助レンダラを最新書式へ改造したわけではない。
- GitHub Issue本文の旧branchパスの修正は未実施。移行内容をcommit/pushしていない。
- 同一データの相対リンクを壊す移送やリンク先の上書きを避ける。学習再開用buffer/optimizerを元々含まない推論モデルは、今回も完全な再開用checkpointには変わらない。

履歴の原本は維持し、現在との対応は[migration manifest](migration_manifest.json)で別に管理する。executed_migration.py/executed_verification.pyは今回実行したコードの記録であり、完成成果上で再実行する手順ではない。
