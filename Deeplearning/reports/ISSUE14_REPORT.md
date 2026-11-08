# Issue #14：固定目標音によるPhase 0の検証

学習・評価の目標を同じ0.35秒の正弦波グライドに固定し、PPO・TD3・SACを各4096環境step、seed 0/1/2の計9 runで比較した。目標周波数・振幅を観測に与える追従課題であり、鳴管モデル・実録音の実験ではない。

数値・図・計算時間・旧ランダム目標条件との比較・検証記録は、専用フォルダの [結果レポート](../results/phase0_fixed_target_20261108/REPORT.md) にまとめた。

- 8/9 runで学習前より最終音響誤差が小さくなった。最終平均はPPO 0.115400、TD3 0.0443437、SAC 0.0824136（小さいほどよい）。
- 最終平均は旧ランダム目標学習より3手法とも大きかった。固定目標にしたことによる一律の改善は確認していない。周波数・振幅の追従誤差も残る。
- [独立監査](../results/phase0_fixed_target_20261108/audit.json) は10,731確認、[既存成果の保護](../results/phase0_fixed_target_20261108/preservation_check.json) は2,097ファイルでPASS。原因の断定や追加学習は行っていない。

まず見る資料：

- [seed 0の3手法の学習曲線](../results/phase0_fixed_target_20261108/figures/learning_curves_seed0.png)
- [PPOの最終スペクトログラム](../results/phase0_fixed_target_20261108/figures/PPO_seed0_final_spectrogram.png)
- [TD3の最終スペクトログラム](../results/phase0_fixed_target_20261108/figures/TD3_seed0_final_spectrogram.png)
- [SACの最終スペクトログラム](../results/phase0_fixed_target_20261108/figures/SAC_seed0_final_spectrogram.png)
- [TD3の目標／生成f・A制御](../results/phase0_fixed_target_20261108/figures/TD3_seed0_controls.png)

学習曲線の終点・summary・生成音・制御図は最終更新後の方策へ揃える。各10 ms区間の報酬と全0.35秒評価を区別し、新条件の指標をheld-outや未知音への一般化性能とは呼ばない。旧結果は保存済みsummaryから参照し、再生成・上書きしない。

報告・描画は [issue14_report.py](../issue14_report.py) がJSON/NPZから独立に生成する。再生成コマンド：

```bash
MPLCONFIGDIR=/tmp/birdsong-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  Deeplearning/.venv/bin/python -m Deeplearning.issue14_report \
  --output Deeplearning/results/phase0_fixed_target_20261108
```

発表では「鳴管モデルへの適用前に、固定目標の正弦波モデルでPPO・TD3・SACの学習挙動を確認した」と説明する。固定目標での性能、学習の安定性、未知目標への一般化、左右音源の効果を混同しない。
