# Issue #14：固定目標音の長期学習

2026-11-14改訂のIssueに従い、PPO・TD3・SAC × seed 0/1/2、計9 runを**各102,400環境ステップ**、ゼロから学習した。評価は2,048ステップ間隔。報酬・音生成・観測・行動・既存ハイパーパラメータは維持し、行動空間の不要な再seedだけを今回の設定で修正した。

[詳細結果レポート](../results/phase0_fixed_target_20261108/REPORT.md) に学習曲線、収束傾向、全seedのMAE、途中最良、更新回数、計算費用を保存した。

| 手法 | 最終全区間音響誤差（3 seed平均±標本SD） |
| --- | ---: |
| PPO | 0.030880 ± 0.006337 |
| TD3 | 0.020582 ± 0.001583 |
| SAC | 0.029000 ± 0.004149 |

主結果は最終更新後の方策で、途中最良への置き換えは行わない。今回の条件ではTD3の最終平均誤差とseed間SDが最小だった。全手法で誤差の上下があり、途中最良より最終値が悪いrunもあるため、単調な改善や数学的収束を確定しない。

- [seed 0の学習曲線・51点](../results/phase0_fixed_target_20261108/figures/learning_curves_seed0.png)
- [TD3の最終スペクトログラム](../results/phase0_fixed_target_20261108/figures/TD3_seed0_final_spectrogram.png)
- [TD3の周波数・振幅追従](../results/phase0_fixed_target_20261108/figures/TD3_seed0_controls.png)
- [全3 seedの補足図](../results/phase0_fixed_target_20261108/figures/learning_curves_all_seeds.png)

9 runの学習は合計921,600ステップ、run内評価は16,380ステップ、全run実行wall時間555.59秒。CPU各1スレッド・独立6プロセスで実行した。19テストPASS、独立監査PASS、既存成果2,156ファイルの保護照合PASS。監査・描画・回帰テストは本比較の学習予算とは別計上。

一時領域で全run・監査を完了後、既存の `results/phase0_fixed_target_20261108/` を同じ相対パスで更新した。旧4,096ステップ結果のcommit・hash・復元手順は [legacy_recovery.json](../results/phase0_fixed_target_20261108/legacy_recovery.json)。途中予算別の恒久フォルダや全評価時点のcheckpointは作らず、各runの最終/最良agentだけを保持する。

旧4,096ステップ値との比較には乱数管理・ライブラリ版の差もあるため参考比較とし、予算の推移は今回の同一run内の4,096→102,400ステップで確認する。目標f/Aが観測に含まれる固定正弦波追従のin-target評価であり、未知音への汎化・鳴管制御・実録音模倣・左右音源の効果を証明した結果ではない。Phase 1/2とC++モデルには変更を加えていない。

リポジトリ直下から再実行（監査成功後に指定の既存成果を更新）:

```bash
PYTHONDONTWRITEBYTECODE=1 Deeplearning/.venv-rl/bin/python -m Deeplearning.issue14 --replace --workers 6
```

学習せず図・レポートだけ再生成する場合:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/birdsong-mpl Deeplearning/.venv-rl/bin/python -m Deeplearning.issue14_report --output Deeplearning/results/phase0_fixed_target_20261108
```
