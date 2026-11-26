# Issue27再改訂 — 左右外力18点と3手法の事前計画

以前のIssue17〜27の全成果を保持し、新しい独立フォルダへ保存する。native/model.cpp/元model.py/実libraryの式・係数は不変。Case Bのnative condition1と左右α1.2e6/β1500を維持、ε_r3.5e7固定、x0=.02cm/D=.01/dt1e-7、0.20s/50kHz/10000sample、初期4状態/遅延/16次15kHz lowpass状態0を保持。

元PythonのBirdModel(B)は右f0を1e7へ固定するため、別のConfiguredCaseB subclassの_controlsだけで今回の固定右f0を設定する。native resetはB/1を呼び、右係数を0にするAや別条件へ置換しない。実メモリの右f0と左右αβを全経路で検査する。

Stage Aは①左3e5/右1e7の旧9音を読み取り専用参照。②左49000/右1e7、③左49000/右49000をε_l[3e7,3.5e7,4.1e7]×p_s[3e6,3.4e6,3.8e6]で各9点、新規18点のみ。各2,000,000 native step、計36,000,000。全点の波形/state/source/output/全刻み統計を保存。末尾1000sample AC std/自身peak>.05、finite、generated_silent偽を候補とし、直線除去後ratio>.05もドリフト除外に確認する。左右の周期的変動proxyは末尾traceのde-trended stdとdrift割合で別に記述し、生理的発声とは呼ばない。

選定はp_s3.0e6を優先し、持続性・波形・周波数構造・spectral entropy/peak数を物理診断で確認。左右の持続変動と相互作用が確認された③も積極的に検討。RL誤差を一切見ず候補1つと理由を先に凍結する。18点不適格なら追加探索/時間延長せず停止。選定音は別Bインスタンスでoracleを1音再生し、波形・状態・native運動統計の一致とE_spec/E_env/E/J=0、右source/output非ゼロを検査。右f0はTargetに対応する値を9run全てで固定し、行動へ含めない。

Stage BはSAC/TD3/PPO×seed0/1/2の9run、各厳密4096環境step＝音、合計36864学習音。独立評価0,128,...,4096の33点/runは別会計。一定3D、観測8D reset[-1,0,...]、1episode1行動。ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、f0_l[49000,800000]を探索し、Target生成の左f0固定を学習へ引き継がない。r=−(.75E_spec+.25E_env)、共通E=E_spec+E_env。

CPU1thread、全隠れ層128×128/LR3e-4/gamma1/batch32。SAC:buffer50000,warmup16,train_freq1,gradient_steps1,auto_0.01,target_entropy−3。TD3:同buffer/warmup等、NormalActionNoise(0,.1)、既定policy_delay2等。PPO:pi/vf別128×128,n_steps128,n_epochs10,clip.2、他既定（max_grad_norm.5等）。全層/実パラメータ数をassert。SAC optimizer各4080、TD3 Critic4080/Actor2040、PPO32rollout×40minibatch=1280 optimizer step（内部epoch counter320）を予定し、更新数一致とは呼ばない。

各手法128step・seed0のsmokeを別計上、評価0/128と保存3方策+training_bestの4再生/本を検査。標準と計測付き実装を3手法×128step×2のsynthetic環境で照合し、9初期seedと構造も確認（native0）。SAC/TD3の実行行動/Replay/報酬/終端を検査。PPOは元Gaussian raw行動をrolloutへ保存し、環境実行clip行動と分ける標準仕様を保持する。全optimizerの実効LR・有限性・Adam状態を観測、既存Issue25の数値停止基準を適用。PPO既定gradient clipは変更しない。

評価は128step後の全更新完了後（PPOの次rollout開始時、最終はlearn終了後）とし、学習器/optimizer/Buffer/RNG/学習bestを変えず評価を学習へ混入しない。Final=4096終了後、Selected=33点の最初minJ、posthoc=最初minE、training_best=実学習の最初minJを別記する。最良方策へ復帰しない。E≤.10は合否/追加計算/やり直しのゲートにしない。failed runは保持し置換/追加学習しない。

完了後は3役割Actorと学習bestの計36音をBで詳細独立再生し、波形/制御/Jと右sourceを確認。全seedの生音と共通gain WAV、共通Target PSD色軸のTarget最上段→SAC/TD3/PPO比較図、包絡/左右状態を保存する。代表seed1を事前固定。①②③の27音は旧①PSD基準・色軸を維持した図も保存する。異なる人工TargetのA/B誤差から構造優位を主張せず、将来の共通実録音比較と分ける。3seedで一般的手法優位・制御の一意復元を主張しない。TQC、直接最適化、時変、5D、実録音、追加grid、固定物理係数変更、commit/push/Issue更新/外部送信はしない。
