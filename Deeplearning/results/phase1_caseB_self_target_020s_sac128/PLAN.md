# Issue27 — 計算前の計画

Case B自身のTarget [4.1e7,3.5e7,3.4e6,3e5,1e7]を新規生成する。左右α=1.2e6/β1500を保持し、右f0=1e7を強制する。実ロードlibraryのメモリをread-only probeで読み、元bird_advanceの関数ポインタのみで積分する。元native/model.cppとlibraryを変更しない。x0=.02cm/D=.01、0.20s/50kHz/10000sample、dt1e-7、初期4状態/遅延/filter=0を保持。

先にCase B Target、同B oracle、同じ5制御でCase A対照を各1音生成する。ソース再ビルドが実libraryとbinary一致しなければ、同じTarget制御でreference libraryを1音追加し波形/状態の一致を検証する。Targetの末尾1000sample AC std/Target peak>.05、有限値、Bの左右係数非ゼロ、右寄与非ゼロ、oracle誤差≤1e-12を物理ゲートとする。不合格では学習せず診断報告に進む。変位だけから開閉を断定せず、右の微小運動も保存する。

物理ゲートと短いReplay/実効LR/seed/正規化/再生試験を通過後、SAC128・3seed(0,1,2)を各4096候補、独立Actor評価0,128,...,4096の33点で実行する。ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、f0_l[49000,800000]、右ε_r3.5e7/f0_r1e7固定。通常SACのみ。Actor/Critic[128,128]、3LR3e-4、auto_0.01/target_entropy−3、batch32/buffer50000/warmup16/gamma1/train_freq1/gradient_steps1、その他Issue23設定と同一。

r=−J、J=.75E_spec+.25E_env（係数和1、倍補正なし）、共通E=E_spec+E_env。Finalは4096終了後、Selectedは33点の最初minJ、posthocは最初minE。training_bestは実学習4096候補の最初minJで、Actor保存方策と区別。独立評価はReplay・学習RNG・最良学習候補に入れない。全更新のLR・有限性・Adam状態、実行行動/物理値/Replayを検査。数値停止基準は既存Issue25と同じで、失敗runを保存し再始動・救済・条件置換しない。

獲得は初期Actor Eから半減する最初の独立評価点と発声proxyを記述し、Selected→FinalのJ/Eと持続状態で保持を別記する。E≤.10は合否・延長・追加探索のゲートにしない。3seed平均±標本SD、全seed/非持続/失敗を表示する。

Final/Selected/posthoc/training_bestをCase Bで独立再生し、左右末尾状態・native全刻みの振動統計、π̃の左右source項と出力項/遅延feedbackを計測する。分解は現在状態の代数的寄与で、片側除去時の因果寄与ではない。WAVはB Target peak由来の共通gain、PSD色基準/表示範囲も全音で共通。Case A Issue23は自身の別Targetに対する参考値のみで、誤差だけによる分岐優劣/学習難易度の結論を出さない。

既存Issue17〜26を上書き/再生成/削除せず専用出力に保存する。Case A旧Targetの流用、旧Issue20の実験、局所探索、左右切替、5D、時変、実録音、別reward、Critic高LRは実施しない。commit/push/Issue更新・外部送信は行わない。
