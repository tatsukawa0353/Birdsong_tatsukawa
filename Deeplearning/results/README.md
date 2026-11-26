# 研究成果の索引

成果物は研究段階・Case・目的で管理する。Issueは経緯を追う識別子とし、改訂ごとに成果フォルダを無制限に増やさない。2026-11-26に整理した。[整理結果・削除/移動/検証記録](maintenance/20261126/REPORT.md)と[旧→新対応表](maintenance/20261126/folder_mapping.csv)を参照。

| 段階・目的 | 保存先 | 内容 |
|---|---|---|
| Phase 0 | [synthetic](Phase0/synthetic/REPORT.md) | 固定合成Target、手法・報酬・学習率比較。旧ランダム目標も識別して保持 |
| Case A・探索次元 | [parameter_dimension](Phase1/CaseA/parameter_dimension/README.md) | 一定1D/2D/3D、直接探索とRL、Issue 13/17等の履歴 |
| Case A・発声領域 | [domain_diagnostics](Phase1/CaseA/domain_diagnostics/README.md) | 全域/制限3D、発声と模倣精度の診断 |
| Case A・報酬診断 | [reward_diagnostics](Phase1/CaseA/reward_diagnostics/) | Issue10の合成0.35秒。結果本文は[旧報告](../reports/ISSUE10_REPORT.md) |
| Case A・ネット幅 | [network_width](Phase1/CaseA/network_width/REPORT.md) | 32/64/128/256、今後の幅選択の根拠 |
| Case A・報酬重み | [reward_weights](Phase1/CaseA/reward_weights/) | restricted_domain: Issue19、sac128: Issue22。報酬のスケールを区別 |
| Case A・学習率 | [learning_rate](Phase1/CaseA/learning_rate/) | actor/criticを分離、固定回帰データも保持 |
| Case A・安定性 | [stability_diagnostics](Phase1/CaseA/stability_diagnostics/REPORT.md) | Selected/Final乖離、Actor/Critic診断 |
| Case A・探索戦略 | [exploration_strategy](Phase1/CaseA/exploration_strategy/REPORT.md) | 広域探索と局所候補生成 |
| Case B・物理診断 | [target_diagnostics](Phase1/CaseB/target_diagnostics/REPORT.md) | 27条件の発声、初回/改訂の停止理由、b3_c00の予備9run |
| Case B・主要学習 | [3targets_3d4d](Phase1/CaseB/3targets_3d4d/REPORT.md) | T1/T2/T3 ×3D/4D ×SAC/TD3/PPO ×3seed、54run |
| 基準・検証 | [reference](reference/) | 基準音、実装診断、発声マップ、活動マスク、環境検証記録 |

新条件は同じ研究目的の下で実験IDと設定ファイルを付ける。学習値・履歴・Target・固有の必要モデル/生データを維持する一方、図は集約比較を中心とし、実験終了時に不要画像・中間/一時生成物を整理する。条件変更と未実施の計算を記録する。新実験前に重複を調べる。実行中の計算に影響する整理は行わない。

描画設定の正本は [plot_style.py](../plot_style.py)。新規図/再描画ではそのcontextを使い、SciencePlots・従来のフォント・46/30/29pt・4pt実線・薄い標本SD帯を共通化する。グリッド、不要タイトル、平滑化、間引きは使用しない。全画像の再描画は今回行っていない。

完了済み実験のmanifest・hash・実行コードsnapshot・計算時PLANは当時の原本である。旧パスを直接開く代わりに [result_paths.resolve](../result_paths.py) を使う。重複ファイルは相対symlinkで正本に集約しており、移行監査で全バイト一致を確認する。リンクを含めて保存/移送する。再描画・追加実験は別の出力先へ書き、正本を上書きしない。削除が不確かなモデルは[保留一覧](maintenance/20261126/deferred_deletions.csv)にある。

現在の保持・パス検証と、物理計算を行わない再描画は[移行後の利用手順](maintenance/20261126/README.md)。旧監査の当時の「削除0」等の主張はその時点の記録であり、今回の削除方針に合わせて書き換えていない。
