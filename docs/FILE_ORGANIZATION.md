# ファイル整理報告（2026-11-13）

この文書は初回整理時点の記録です。後続のC++モデル22件の移動で、直下モデルの保留理由とビルドパスは更新されています。現状は [追加整理報告](MODEL_RELOCATION.md) を参照してください。

作業ブランチは開始時・終了時とも `main_deeplearning`。開始時の変更は `.gitignore` のみ。
`main` の参照は不変。コミット・push・マージ・削除・履歴操作は行っていません。

## 整理前

```text
./
├── BirdsongModel*.cpp / *.h   鳴管モデル実装・複数の研究条件
├── main*.cpp                 実行・パラメータ探索
├── *.py                      音響解析・比較・可視化・音声化
├── birdsong_*sim             既存実行バイナリ
├── *.csv / *.png             集計結果・図
├── Color map/                手動カラーマップPython
├── Complex_2/                複雑度差Python
├── __pycache__/              追跡済みキャッシュ
├── et c722261                ANSI色付きGitログと思われるテキスト
├── Deeplearning/             強化学習コード・設定・資料・計算結果
├── ss/                       個別.gitignoreのみ
├── .venv/ / .vscode/
└── README.md / .gitignore
```

整理後のツリーは [README](../README.md) に記載。
移動は [file_moves.tsv](file_moves.tsv) に全件記載しています。

## 採用した構成と調査

解析と描画が混在するスクリプトが多いため `legacy/analysis/` を共通の置き場としました。
音響指標はACI・SCI・HNR・スペクトルエントロピー・複雑度・PSDなど、比較コードは
Aci_similar/complexity_similar/kyori、描画はkashika/colormapなどです。
音声化はaudio/generate_distance1_audio、generate_distance1_single_source_outputsは
解析・音声化・描画を束ねます。複数バージョンは統合していません。
C++はモデル実装とmainの実行・探索処理を分けて扱いました。
CSVの列はepsilon, ps, HNR, SCI, ACI, SEなどの集計結果であり、入力録音ではありません。
PNGはs.pyの出力名と一致する既存図です。これらは `results/legacy/` に保管しました。
コードの参照名と一致する既存入力データは今回の移動対象にありません。

全追跡ファイル2340件を台帳化し内容ハッシュを保存しました。
研究コードのimport/include・入出力設定を読み、Deeplearningの参照・保護用ハッシュ処理を調査しました。
大量の既存結果は拡張子・ディレクトリ・メタデータの役割で分類しています。
音声・配列・学習重みの全データを再解析する調査ではありません。
個別の静的依存情報は [legacy_dependencies.md](legacy_dependencies.md) を参照してください。

## 移動せず残したもの

- 直下の `BirdsongModel*` 全件。`Deeplearning/reproduce.py` と `tests/test_core.py` は
  `BirdsongModel_10*10.cpp` と `BirdsongModel_2.h` を直下からビルドします。
  phase1_validation.py / issue13.py / issue14.py は直下の `BirdsongModel*` を列挙して
  パス付きで内容を保護します。既存結果のprotected_before.json等にも旧パスが残るため、
  Pythonのcolormapも含め一括して維持しました。これらのモデルを移すsrc/includeは新設していません。
- `Deeplearning/` 全体。コード・結果・source_snapshot・設定・資料をバイト単位で保持。
- `.venv/` 全体。走査・移動・変更を行っていません。
- `.vscode/` と `ss/.gitignore`、直下 `.gitignore`。既存設定を維持しました。

## 変更内容と参照パス

コード本文の変更は0件。アルゴリズム・物理定数・初期条件・計算条件も変更していません。
新しい呼び出しパスと `g++ -I.` をREADMEに追記しました。
解析は `python3 legacy/analysis/<ファイル>.py` で直下から呼び出す方式です。
サブディレクトリ内の手動描画・複雑度差も従来と同じ作業ディレクトリを指定してください。
既存の結果保存先文字列は変更せず、新規実行時の出力先も従来どおりです。

`git mv` は `.git/index.lock` を作れない読み取り専用エラーとなったため、
通常のファイル移動を使用しました。インデックスには手を加えていません。
現在のgit statusは旧パスの削除と新ディレクトリの未追跡表示となります。
ファイル本体を削除したわけではなく、全移動ファイルのハッシュ一致を確認済みです。

## 検証結果

- Python 127ファイル（既存スナップショット含む）：ASTによる構文確認成功。
  import時にI/Oを実行するコードがあるため一括importは行わず、ローカルimport 6参照を静的確認。
  外部依存パッケージのインストール・全機能の実行確認は行っていません。
- C++ 30翻訳単位：C++17の構文確認は28件成功、2件は既存不整合。
  `BirdsongModel_rk_test1.cpp` と `BirdsongModel_10*10_s(t).cpp` の
  saveData(double,double) と対応ヘッダーの saveData() が不一致です。
  両ソースとヘッダーは開始時と同じハッシュであり、移動起因ではありません。
- 実行ドライバ13組：12組コンパイル・リンク成功。
  main.cpp + BirdsongModel_rk_test1.cpp のみ上記不整合で失敗。
- Deeplearning/native/reference.cpp + 直下モデル：C++11でリンク成功。
  native/model.cppと保存スナップショットも構文確認成功。
- 生成した検証バイナリは `/tmp/birdsong-organize/` のみ。
  シミュレーション・学習・GPU処理は未実行。
- 既存2340ファイルの移動先を含むSHA256はすべて一致（README追記前に確認）。
  READMEは元の全文を接頭部分として保持し追記のみ。
  `.gitignore` は開始時のバイト列と一致。mainの参照も一致。

詳細は [organization_checks.txt](organization_checks.txt) を参照してください。

## 今後の管理上の注意

- 移動済みコードは直下から実行してください。データ入出力を今後一括変更する場合は、
  CSV内に記録される出力パスや解析側の入力パスも合わせて検討する必要があります。
- バイナリのソース対応・環境互換性は未検証です。今後の再ビルド時は元バイナリを保管してください。
- 直下に残したモデルの移動にはDeeplearning側と過去の保護ハッシュ記録の取り扱い検討が必要です。
- 新たな研究結果は用途別のresultsを推奨しますが、今回コード内の出力先は変更していません。
- `.gitignore` の既存変更で `*.pyc` と `__pycache__/` は無視されます。
  移動した追跡済みキャッシュ12件をGitにも保持する場合は、その新パスを明示的に
  強制追加する必要があります。台帳を確認し、意図しない削除としてコミットしないでください。
- 最終ステージ時に移動一覧と `git diff --cached --summary` を照合してください。
  ステージ・コミット・pushは今回行っていません。

移動件数（配置先別）：

- `docs/archive/`: 1件
- `legacy/analysis/`: 66件
- `legacy/bin/`: 9件
- `legacy/src/`: 13件
- `results/legacy/`: 2件
