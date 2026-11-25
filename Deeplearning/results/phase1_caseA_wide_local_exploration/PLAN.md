# Issue26 — 計算前の凍結計画

AはIssue23の制限域[49000,800000]・SAC128・λ=.75・3LR=3e-4の3seedを読み取り専用参照。B/C/Dは全てf0_l=[49000,10000000]。Stage BのB3seed×4096候補を先に実行・記録し、結果が良くても悪くてもStage CのC/D×3seed×4096候補を省略しない。新規合計9run、warmup16・評価0,128,...,4096の33点・4080更新/optimizer/run。

最初256試行は全広域条件で標準SACの同じ生成方法。256の後はC/Dのみ、choice RNGでlocal80%/SAC20%。choice seed260026+seed、candidate seed260126+seedの独立PCG64を使い、学習のPython/NumPy/Torch RNGは使わない。条件C/Dでchoice系列を共通にする。毎試行、標準SAC提案を先に生成し、local試行だけ実行行動とReplay行動の双方を置き換える。local試行の未実行SAC提案は別欄に記録する。

中心は過去の実学習候補の最初のminJ（同点は先着）。独立評価・Target oracleは使わず、非持続も除外しない。Cの物理半径[625000,37500,50000]、Dは[625000,37500,250000]。各次元の[center−radius,center+radius]とglobal boundsの共通区間で連続一様サンプリング。格子の刻みではない。float32写像の端点超過だけ最大16回まで再抽選し、クリップせず、attempt数と丸め誤差を記録する。実行はfloat32正規化の丸めを伴い、無限精度の連続値そのものではない。

中心は実行された制御値で更新し、実行行動・環境の物理値・Replay stored action/rewardを全試行で一致検査する。隣接試行差が半径を超えることを許す。通常SAC20%は全域の可能性を残すが、一様な全域被覆を保証しない。初めの256はlocal0%なので、全4096での期待local率は75%、257以降の期待率80%。

数値停止は既存と同じ：非有限、負のAdam二次moment、勾配L2>1e+06、重み絶対値>1e+06、moment絶対値>1e+12で停止して失敗保存。条件削除・再始動・幅調整・oracleによる中心初期化はしない。

事前テストは3seed×標準SAC/B/C/Dで256試行の行動・報酬・ネットワーク/optimizer/RNG/Replayが一致。正式B/C/Dの各seedでも最初256件と128/256時点の全学習器署名を照合する。独立評価・ログ・保存では学習RNG、local RNG、中心が不変であることを検査。

発見=学習候補のminJとその試行・共通E/発声。獲得=決定論的Actor33点。保持=Selected→Final。Finalは4096終了後、Selectedは33点の最初minJ、posthocは同点の最初min共通E。学習bestをActorの出力と同一視しない。Bのprefix256モデル/波形だけ追加保持する。

相対的な到達基準を事前固定：(1)学習試行257以降でJ≤0.5×そのseedの最初256学習候補minJ、(2)独立ActorのE≤0.5×同条件/seedの初期Actor E、(3)256以降のActor J≤1.25×その条件/seedのprefix256 minJ。Aは自身の既存prefixで、B/C/Dは同一prefix。これらは記述指標で合否・追加計算のゲートではない。E_total≤.10の達成を今回の目標として最適化しない。

f0分布の固定区間は[49000.0, 800000.0, 2500000.0, 5000000.0, 10000000.0]、高域割合は>800000、>2500000、>5000000。各区間の発声・ε_l/p_sの組合せも示し、高f0を一律に無音・不要と断定しない。local hit=直前incumbentのJを厳密に改善した件数。local/SACを分け、非持続/失敗を除外せず、実偏差・境界共通区間・中心履歴・実割合を残す。

Case A、0.20s/50kHz/10000sample、dt1e-7、x0=.02cm/D=.01、初期状態/遅延0、同一Target/hash、ε_l[2.5e7,5e7]・p_s[2.5e6,4e6]・右ε_r3.5e7/f0_r49000を固定。r=−(.75E_spec+.25E_env)、共通E=E_spec+E_env、2倍補正なし。f0_lはHzでなく開閉外力係数。Actor/Critic[128,128]、全LR3e-4、auto_0.01/target_entropy−3、batch32/buffer50000/gamma1/train_freq1/gradient_steps1/CPU1threadを保持する。

A/Bは物理範囲、B対C/Dはoff-policy行動データ生成アルゴリズム、C/Dはf0局所半径の差と解釈する。正規化行動やstdはA/Bで物理尺度が違う。元音・共通gain/色基準の表示を保存。追加の探索グリッド・Case B・時変・実録音・学習率/entropy調整・最良方策復帰は行わない。Case A全体の総括と教員共有、Case B自身のTarget生成・Issue20旧仕様の改訂は次の作業で、今回外部へ送信しない。
