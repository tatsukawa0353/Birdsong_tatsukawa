# Issue27改訂 — 計算前の凍結計画

旧Issue27停止記録、旧Case A17〜26、native/model.cppと実libraryを保持する。新規成果はこの独立フォルダへ保存。Case Bの左右α=1.2e6/β1500、右ε_r3.5e7/f0_r1e7、x0=.02cm/D=.01、dt1e-7を維持する。全0.20s一定、4状態・遅延・16次15kHz filter=0から50kHz/10000sampleを生成。

Stage Aはε_l[3e7,3.5e7,4.1e7]×p_s[3e6,3.4e6,3.8e6]の9条件のみ。左f0_l=3e5、右固定。各点の末尾20ms AC std/自身peak>.05、finite、generated_silent偽を基本候補とし、直線除去後std/自身peak>.05も要求してドリフトだけの候補を除く。波形・左右動態を確認し、p_sは3.0→3.4→3.8e6の順を優先、同じp_sでは直線除去後ratioが高い順（同点はε_l昇順）にTargetを選ぶ。これは物理指標であり学習誤差を見ない。候補0なら本学習へ進まず追加探索しない。

選定後、同じBの別インスタンスでoracleを1音、同じ5制御でA対照を1音生成する。B oracleの波形/状態/運動統計とE_spec/E_env/E/Jが0誤差、B右source/output非ゼロ、A右source0を確認。Target byte列/hash・選定根拠を9run共通で固定する。

Stage BはSAC/TD3/PPO×seed0/1/2の9run、各厳密4096環境step/音。独立評価は0,128,...,4096の33点で予算と別会計。f0_l範囲[49000,800000]、ε_l[2.5e7,5e7]、p_s[2.5e6,4e6]、右固定。観測8D reset[-1,0,...]、1episode1行動。r=−J、J=.75E_spec+.25E_env、共通E=E_spec+E_env。

全ネットの隠れ層128×128、LR3e-4、gamma1、batch32、CPU1thread。SACはbuffer50000/warmup16/train_freq1/gradient_steps1/auto_.01/target_entropy−3。TD3は同buffer/warmup/train_freq/gradient_steps、NormalActionNoise(0,.1)、その他標準policy_delay2等。PPOはpi/vfを別の128×128、n_steps128/n_epochs10/clip_range.2、その他既定。PPOは128rollout×32回で4096、40minibatch/rollout×32=1280 optimizer step、内部epoch counter320。SACはActor/Critic/温度各4080、TD3はCritic4080/Actor2040で、更新数は同一とは呼ばない。

正式run前に各手法128stepのsmokeをseed0で1本ずつ、2独立評価/本と保存モデル/学習best4音再生/本を別会計。標準実装と計測付き実装の128step synthetic試験を3手法で照合（native0）。各seedの初期化と実ネット層/総パラメータ数もassert。smokeは本番に混ぜない。

SAC/TD3は実行正規化行動とReplay action/reward/終端を照合。PPOの標準Gaussianは環境へclipして実行するが、rolloutには元の非clip行動を保存してlog_probと一致させる。この標準PPOの仕様を変えず、raw/実行行動を別欄に記録する。全optimizerのLR・有限性・Adam状態を観測し既存Issue25の数値停止基準を適用、失敗は置換しない。PPOの既定max_grad_norm=.5は保持する。

評価は各128環境step後の全更新を終えてから実行し、PPOのrollout終了/更新前との混同を避ける。Finalは4096終了後、Selectedは33点の最初minJ、posthocは最初minE、training_bestは実行学習候補の最初minJ。評価・保存では学習器/optimizer/Buffer/RNG/最良学習候補を変えず、最良方策へ復帰しない。E≤.10は目標/合否/延長/追加計算のゲートにしない。

Final/Selected/posthoc/training_bestの36音を独立に詳細再生し、波形・J・制御一致、左右末尾運動とsource/output寄与を監査する。全音のWAVは選定Target peak由来の共通gain、表示PSDは同Target基準で共通色軸。代表音はseed1、全seedの数値を保存。3seed平均±標本SD、未完走/非持続も表示し、制御値の一意復元や一般的手法優位を主張しない。

今回の比較はCase B内部の3手法。別人工TargetのCase A誤差から構造優位を結論しない。共通実録音Targetでの2音源構造比較は将来。TQC、直接最適化、時変、5D、左右切替、実録音、局所探索、物理式/固定係数変更、許可外の追加探索は行わない。commit/push/Issue更新/外部送信なし。
