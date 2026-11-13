ここでは論文発表の再現プログラムと実行結果


10*10パラメータの100通りシミュレーションはファイル名に10*10とかいてあるもの
カラーマップ作製は基本的にver2.0を使用
詳細なカラーマップはver1.1,1.2

追記：カラーマップ作製は手動の場合，BirdsongModel_colormap_Manual.py
「_1」はfoを大きくしたver,「_2」はα＝β＝0　ver.

g++ main.cpp BirdsongModel_rk_test1.cpp -o birdsong_sim -std=c++11
./birdsong_sim

## ファイル整理（2026-11-13）

上記は従来の記録をそのまま保存したものです。上記の `main.cpp` は
`legacy/src/main.cpp` に移動しました。ただし旧ビルド例は、既存の
`legacy/src/BirdsongModel_rk_test1.cpp` とヘッダーの `saveData` の不一致により失敗します。
今回、アルゴリズム・パラメータ・ソースコード本文の変更は行っていません。

```text
Birdsong_tatsukawa/
├── Deeplearning/          # 強化学習・設定・テスト・既存結果（構造も内容も維持）
├── BirdsongModel*.py      # 既存カラーマップ5件（今回はC++のみ移動）
├── legacy/
│   ├── src/              # 鳴管モデル.cpp/.h・main*.cpp・パラメータ探索
│   ├── analysis/         # 音響指標・比較・可視化・音声化のPython
│   │   ├── Color map/    # 手動カラーマップ（既存のまとまりを維持）
│   │   ├── Complex_2/    # 複雑度差の可視化
│   │   └── __pycache__/  # 既存キャッシュを削除せず保管
│   └── bin/              # 既存の実行バイナリ
├── results/legacy/       # 既存の集計CSV・図
├── docs/                 # 整理報告・全ファイル台帳・移動一覧・検証結果
│   └── archive/          # Gitログと思われる既存のテキスト資料
├── .venv/                # 変更なし
├── .vscode/              # 既存のエディター設定
├── ss/                   # 既存の個別.gitignoreを維持
├── README.md
└── .gitignore            # 作業開始前の未コミット変更を保持
```

### パスと実行方法

従来コードの相対入出力は**リポジトリ直下を作業ディレクトリ**として解釈します。
`legacy/analysis/` や `legacy/bin/` に `cd` せず、直下からパスを指定してください。
同じディレクトリにあるPython同士のimportは維持しています。
`data/` は、今回移す独立した入力データが確認できなかったため新設していません。
入力CSV・録音の多くはこのチェックアウトに存在しません。必要な入力名は
[参照調査](docs/legacy_dependencies.md)と各スクリプトの設定を確認してください。

以下のC++コマンドはコンパイル・リンクを確認済みです。出力先は `/tmp` とし、
既存バイナリを上書きしません。シミュレーション本体は今回実行していません。

```bash
# リポジトリ直下から（C++17、g++）
g++ -std=c++17 legacy/src/main_distance1.cpp legacy/src/BirdsongModel_distance1.cpp -o /tmp/birdsong_distance1_sim
# 実行する場合：/tmp/birdsong_distance1_sim
# 生成される simulation_distance1.csv を解析する場合（入力未生成のため今回は未実行）
# python3 legacy/analysis/generate_distance1_audio.py
# python3 legacy/analysis/calculate_distance1_aci_se.py
# python3 legacy/analysis/kashika_distance1.py
```

Pythonの実データを用いた実行は未確認です。上記3スクリプトの入力・出力と
構文、およびローカルimportを静的に確認しています。
既存バイナリは `./legacy/bin/birdsong_sim` のように直下から指定できますが、
再ビルド元との一致や別環境での動作は保証していません。
強化学習の実行手順は既存の [Deeplearning/README.md](Deeplearning/README.md) を参照してください。
今回、学習・大規模計算・GPU処理は実行していません。

### 整理の記録

- [整理報告](docs/FILE_ORGANIZATION.md)：変更前後・保留理由・確認範囲・管理上の注意
- [全ファイル台帳](docs/file_inventory.tsv)：既存2340ファイルの分類・新旧パス・移動前SHA256
- [移動一覧](docs/file_moves.tsv)：前回91件＋今回22件の移動元と移動先
- [検証ログ](docs/organization_checks.txt)：構文・リンク・import・内容保持の確認


### C++モデルの追加整理（2026-11-13）

直下のC++モデル14件とヘッダー8件を `legacy/src/` に集約しました。
C++本文は変更せず、Deeplearningのビルドパスと保護ハッシュのパス解決のみ調整しました。
`Deeplearning/legacy_paths.py` が新配置と過去記録の旧パスを対応づけます。
過去の結果・ハッシュ記録・ソーススナップショットは変更していません。
過去の実験はPythonソースのパス修正によるハッシュ差分を検出するため、
同じ結果ディレクトリを再利用せず、新しい実験ディレクトリを使用してください。
詳細は [追加整理報告](docs/MODEL_RELOCATION.md) と [検証ログ](docs/model_move_checks.txt) を参照してください。
