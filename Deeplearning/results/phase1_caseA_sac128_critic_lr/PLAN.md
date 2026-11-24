# Stage A — 計算前に固定したIssue25計画

既存Issue23の基準3seedとIssue24が参照した基準manifest/hashを照合した。新規フル学習はCritic LR1e-4/1e-3×3seed×4096候補の6runのみ。Actor/温度LRと共通scheduleは3e-4、他設定・模型・Target・報酬・範囲は不変。全optimizer更新直前/直後と33独立評価で実効LRを検査する。

高LRの初期nativeテストは3seed×32候補、低LRはseed0×32候補を事前確認し、全て成功。通常の有限値検査に加え、全LRで同じ緊急停止基準を固定：勾配L2>1e+06、重み絶対値>1e+06、Adam moment絶対値>1e+12、非有限値または負の二次momentで停止する。クリップ・置換・失敗seedの削除は行わず、使用予算と失敗を残す。

Final=4096最終更新後、Selected=33点の最初のminJ、posthoc=同33点の最初のmin共通E。初回E≤.10かつ持続の方策も1本だけ保持。全33時点で同じ99候補へのQ1/Q2/minQ、Actor行動・∇a minQ・alpha/std/entropy、全Actor更新差分と3LR/数値検査を記録する。独立評価はReplayに入れず、保存方策に戻さない。

固定回帰は既存Issue23学習経験12288件を全てtrain、独立評価99候補をholdoutとして完全隔離する。Issue22は同じデータの再現元であるため重複して加えない。各回帰runは全3source seedの同じtrainを使い、回帰seed0/1/2は初期Criticとbatch列のseedを指す。構造/twin-Q/Adam/batch32/損失を固定し、Critic LR3e-4/1e-4/1e-3×3seedの9本を各4080更新する。オンライン完了後に開始し、Actor・温度・target Criticを学習せず、オンラインActorへ接続しない。

初期Criticは同seedの標準SACから取り、全LRで同じ重み。mini-batch index列はlocal NumPy seed250025+seedで先に生成し、全LRで同じ列を使う。更新目標はfloat32(r)（SB3 Replayと同じ丸め）、損失0.5*(MSE(Q1,r)+MSE(Q2,r))。報告は元float64 rも用いる。更新数は評価値によらず固定、早期停止・holdout選抜・サンプル追加/重み付けは0。診断時刻は[0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 4080]、Final回帰だけを主比較する。

点ごとのhashと重複・近傍重複を[dataset_audit](dataset_audit.json)と[近傍CSV](holdout_nearest_neighbors.csv)へ保存した。raw完全重複0、Critic入力完全重複0。同じTarget・相関した方策軌道・近傍点は共有し、新Target/一様領域の汎化性能とは呼ばない。holdoutの結果で今回のLRや更新数を変更しない。

追加の局所鳴管計算上限は**0音**。Issue23の既存36音と3固定anchorの真の中心差分だけを参照し、現在のCriticの勾配を同じ既存点で照会できる。新しい訪問点の真の勾配を測ったとは言わない。全99候補・lowJ10・実際の自身の学習経験を別集計する。

研究用のStage B/C実行前にこの計画・データ・実装hashをmanifestへ凍結する。数値異常は失敗として保持し、欠けたFinalを途中方策へ置き換えない。
