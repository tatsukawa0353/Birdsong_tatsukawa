# 2026-11-26 成果物整理の記録と利用手順

[整理結果](REPORT.md) / [全フォルダ調査](folder_inventory.csv) / [全階層の整理前件数](directory_inventory_before.csv) / [移行対応](folder_mapping.csv) / [削除・重複一本化](deletions.csv) / [削除保留](deferred_deletions.csv) / [GitHub旧パス](github_issue_paths.csv)。整理前HEADと全ファイルのhashは[inventory_before.json](inventory_before.json)。計算/評価は再実行していない。

## 移行後の監査

リポジトリ親から実行する。新しい音、学習、元成果の上書きは発生しない。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-mpl \
  Deeplearning/.venv-rl/bin/python -m Deeplearning.results_maintenance audit
```

この監査は移動後の全保持ファイルのhash、相対リンク、承認された文書/パス修正、元物理コードを照合する。旧監査が要求する「当時の全画像が存在し削除0」「実行時sourceと現sourceが同一」は今回の整理後には成立しないため、旧記録をこの監査で置き換えて解釈する。旧監査を再実行したことにはしない。元の計算の完全な検査を再現する場合は整理前commit `259e7da` の別checkoutを使う。凍結sourceの比較は [source_changes.json](source_changes.json) に列挙した実際の前後hashだけを受理し、digest自体を書き換えない。

## 再描画

削除した個別図の元データは全て保持している。NPZの波形から現在の共通書式でスペクトログラム/波形を描くには、別出力先を指定する。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/birdsong-mpl \
  Deeplearning/.venv-rl/bin/python -m Deeplearning.results_maintenance redraw \
  --input Deeplearning/results/Phase1/CaseB/3targets_3d4d/preflight/T1/target.npz \
  --output /tmp/birdsong-redraw-T1
```

Case Aのrollout.npzも同じコマンドで読み込める。`--reference` にTarget NPZを指定するとそのPSD基準を使う。科学的な比較では同じTarget/referenceを全音に指定する。表示STFTは従来の `acoustic_views.display_psd` を維持する。このコマンドは新書式の保存データ表示であり、削除PNGをバイト単位で復元するものではない。

元の学習曲線/集約図の描画コードも保持した。Issue17/27の主描画は共通 `plot_style.py` を使用する。その他の過去レンダラは当時の小さいフォント等を含む再現用コードでもあるため、将来の研究図は共通contextと十分な図サイズで描く。旧report/freezeコマンドは歴史的manifest/hashを再保存するので、完成成果の上で実行せず別checkout/別出力先を使う。

履歴CSV/JSON内の旧パスは改変せず、`Deeplearning.result_paths.resolve(path)` で現配置へ変換する。1,582個の重複バイナリは相対symlinkで旧の各参照位置も保持する。固有モデルの削除や学習のやり直しは行っていない。
