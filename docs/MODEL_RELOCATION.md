# C++モデル追加整理（2026-11-13）

作業開始・終了とも `main_deeplearning`。前回の未コミット整理と `.gitignore` を保持しました。
今回はC++ソース14件・ヘッダー8件の22件のみを移動しました。前回の移動一覧は91件です。
既に整理済みのファイルの再移動・統合・削除は行っていません。

## 移動と用途

すべて直下から `legacy/src/` に移動。コード本文とSHA256は全件一致しています。
`git mv` は前回同様 `.git` が読み取り専用のため利用できず、通常移動を使用しました。
インデックスは変更していません。旧パスの削除・新パスの未追跡表示になります。

| 移動元 | 移動先 | 用途 |
|---|---|---|
| `BirdsongModel.cpp` | `legacy/src/BirdsongModel.cpp` | 論文再現用の初期モデル実装 |
| `BirdsongModel.h` | `legacy/src/BirdsongModel.h` | 初期モデル・rk系のクラス宣言 |
| `BirdsongModel_10*10.cpp` | `legacy/src/BirdsongModel_10*10.cpp` | epsilon・psの10×10探索用モデル。RL参照実装に使用 |
| `BirdsongModel_10*10_2.cpp` | `legacy/src/BirdsongModel_10*10_2.cpp` | 左右のepsilonを独立指定する10×10探索用モデル |
| `BirdsongModel_10*10_s(t).cpp` | `legacy/src/BirdsongModel_10*10_s(t).cpp` | 探索モデルの左右音源出力保存を含む変種（既存コンパイル不整合） |
| `BirdsongModel_2.h` | `legacy/src/BirdsongModel_2.h` | 単一epsilon指定の探索モデル宣言 |
| `BirdsongModel_3.h` | `legacy/src/BirdsongModel_3.h` | 左右独立epsilon指定の探索モデル宣言 |
| `BirdsongModel_distance1.cpp` | `legacy/src/BirdsongModel_distance1.cpp` | epsilonとpsの時間変化・単一音源切替を持つ距離評価用モデル |
| `BirdsongModel_distance1.h` | `legacy/src/BirdsongModel_distance1.h` | 対応する BirdsongModel_distance1.cpp のクラス・計算条件指定の宣言 |
| `BirdsongModel_epsilon_change.cpp` | `legacy/src/BirdsongModel_epsilon_change.cpp` | 線形・二次・段階・正弦波などのepsilon変化モデル |
| `BirdsongModel_epsilon_change.h` | `legacy/src/BirdsongModel_epsilon_change.h` | 対応する BirdsongModel_epsilon_change.cpp のクラス・計算条件指定の宣言 |
| `BirdsongModel_linear_epsilon.cpp` | `legacy/src/BirdsongModel_linear_epsilon.cpp` | epsilon線形変化、右音源切替を持つモデル |
| `BirdsongModel_linear_epsilon.h` | `legacy/src/BirdsongModel_linear_epsilon.h` | 対応する BirdsongModel_linear_epsilon.cpp のクラス・計算条件指定の宣言 |
| `BirdsongModel_linear_epsilon_ps.cpp` | `legacy/src/BirdsongModel_linear_epsilon_ps.cpp` | epsilonとpsの線形変化モデル |
| `BirdsongModel_linear_epsilon_ps.h` | `legacy/src/BirdsongModel_linear_epsilon_ps.h` | 対応する BirdsongModel_linear_epsilon_ps.cpp のクラス・計算条件指定の宣言 |
| `BirdsongModel_linear_ps.cpp` | `legacy/src/BirdsongModel_linear_ps.cpp` | ps線形変化モデル |
| `BirdsongModel_linear_ps.h` | `legacy/src/BirdsongModel_linear_ps.h` | 対応する BirdsongModel_linear_ps.cpp のクラス・計算条件指定の宣言 |
| `BirdsongModel_rk_fig5a.cpp` | `legacy/src/BirdsongModel_rk_fig5a.cpp` | RK法による図5(a)条件のモデル |
| `BirdsongModel_rk_fig5b.cpp` | `legacy/src/BirdsongModel_rk_fig5b.cpp` | RK法による図5(b)条件のモデル |
| `BirdsongModel_rk_fig6.cpp` | `legacy/src/BirdsongModel_rk_fig6.cpp` | RK法による図6条件のモデル |
| `BirdsongModel_rk_normal.cpp` | `legacy/src/BirdsongModel_rk_normal.cpp` | RK法による通常条件のモデル |
| `BirdsongModel_rk_test1.cpp` | `legacy/src/BirdsongModel_rk_test1.cpp` | RK試験モデル（既存コンパイル不整合） |

## includeと実行時の関係

すべてのモデルとヘッダーを同じディレクトリに置くことで、C++のinclude本文を変更していません。
main系も同じ場所にあり、ローカルincludeがそのまま解決します。複数のモデル変種が同じ
クラス名を定義するため、全.cppをまとめてリンクせず、適切なmainと実装1組でビルドします。

| 実装 | 直接includeするモデルヘッダー |
|---|---|
| `BirdsongModel.cpp` | `BirdsongModel.h` |
| `BirdsongModel_10*10.cpp` | `BirdsongModel_2.h` |
| `BirdsongModel_10*10_2.cpp` | `BirdsongModel_3.h` |
| `BirdsongModel_10*10_s(t).cpp` | `BirdsongModel_2.h` |
| `BirdsongModel_distance1.cpp` | `BirdsongModel_distance1.h` |
| `BirdsongModel_epsilon_change.cpp` | `BirdsongModel_epsilon_change.h` |
| `BirdsongModel_linear_epsilon.cpp` | `BirdsongModel_linear_epsilon.h` |
| `BirdsongModel_linear_epsilon_ps.cpp` | `BirdsongModel_linear_epsilon_ps.h` |
| `BirdsongModel_linear_ps.cpp` | `BirdsongModel_linear_ps.h` |
| `BirdsongModel_rk_fig5a.cpp` | `BirdsongModel.h` |
| `BirdsongModel_rk_fig5b.cpp` | `BirdsongModel.h` |
| `BirdsongModel_rk_fig6.cpp` | `BirdsongModel.h` |
| `BirdsongModel_rk_normal.cpp` | `BirdsongModel.h` |
| `BirdsongModel_rk_test1.cpp` | `BirdsongModel.h` |

リポジトリ直下からの確認済みビルド例：

```bash
g++ -std=c++17 legacy/src/main_distance1.cpp legacy/src/BirdsongModel_distance1.cpp -o /tmp/birdsong_distance1_sim
g++ -std=c++11 -Ilegacy/src Deeplearning/native/reference.cpp 'legacy/src/BirdsongModel_10*10.cpp' -o /tmp/birdsong_reference
```

シミュレーションの入力・CSV・結果保存先は変更していません。実行する場合は従来どおり直下を作業ディレクトリとしてください。

## 修正した参照パス

- `Deeplearning/reproduce.py`：参照モデルのソースパスと `-I` を `legacy/src/` に変更。
- `Deeplearning/tests/test_core.py`：同じ参照モデルのコンパイル先を変更。
- `Deeplearning/phase1_validation.py`、`issue13.py`、`issue14.py`：保護対象の列挙先を新配置に対応。
- `Deeplearning/issue14.py`、`issue13_audit.py`、`issue14_audit.py`：旧モデルパスを現在の配置へ解決。
- `Deeplearning/legacy_paths.py`：上記の共通パス定義・旧パス解決。学習や数値計算の処理は持ちません。
- README・台帳・移動一覧：現在の配置を反映。初回報告は当時の記録として保持し、追加報告へのリンクを追記。

`Deeplearning/native/model.cpp` は従来どおり自己完結したRL用数値実装です。内容・配置は変更していません。
`native/reference.cpp` のincludeはビルド時の `-Ilegacy/src` で解決します。

## 過去のハッシュ記録

過去の `protected_before.json`・`source_sha256.json`・manifest・結果内のソーススナップショットは変更していません。
モデル保護のキーは従来のbasename（例 `BirdsongModel_2.h`）を論理的な識別子として維持し、
新規保護記録でも同じキーを使用します。実体の読み取り先だけを `legacy/src/` に対応づけます。
旧キーの変換対象は直下の `BirdsongModel*.cpp` / `.h` のみに限定し、ハッシュ値の変更や検証のスキップはありません。
他のパスはそのまま扱い、監査のリポジトリ外パス拒否も維持しました。

過去3実験のモデル27件（C++/ヘッダー22件＋残存Python5件）は、各記録のSHA256と全件一致。
監査の実際のhashesメソッドを依存ライブラリから独立させて実行し、正常・偽ハッシュ・範囲外パスを確認しました。

一方、今回パス修正したPythonソースは過去のソースハッシュと異なります。
既存実験全体の監査や再開でsource_changedが出る可能性を、モデル移動の検証とは区別してください。
この検査を無効化せず、過去の結果は保存し、新しい実験には新しい出力ディレクトリを使ってください。
当時のスナップショットをそのまま実行する場合は、当時の配置の別チェックアウトが必要です。

## 動作確認

- 移動前後でC++14翻訳単位の構文検査結果が一致：12件成功、2件は既存エラー。
- `BirdsongModel_rk_test1.cpp` と `BirdsongModel_10*10_s(t).cpp` は、移動前から
  `saveData(double,double)` とヘッダーの `saveData()` が不一致。修正はしていません。
- mainとの13組は12組リンク成功。main.cpp＋rk_test1のみ上記エラーで失敗。
- RL用native/model.cppの共有ライブラリとnative/reference.cpp＋移動モデルのリンク成功。
- Python128ファイルのAST構文確認成功。修正箇所のパスと旧記録のハッシュ解決を独立検証。
- `.venv/` にstable_baselines3がなく、学習モジュール全体のimport・既存回帰テストは未実行。
  インストール・環境変更は行っていません。
- 学習・GPU・大規模計算・シミュレーション本体は実行していません。検証バイナリは/tmp配下のみ。
- ログは [model_move_checks.txt](model_move_checks.txt)。

## 移動せず残したファイル

直下の5つの `BirdsongModel_colormap*.py` は今回はC++集約の対象外として保持しました。
既存保護記録との照合を維持しており、Python解析としての用途は変わりません。
今後 `legacy/analysis/` に移す場合はPython旧パスにも今回と同じ対応づけを拡張できます。
C++モデル・ヘッダーの直下残存は0件です。空の旧ディレクトリも削除していません。

## 直下に残る通常ファイル一覧

- `.gitignore`
- `BirdsongModel_colormap_1.py`
- `BirdsongModel_colormap_2.py`
- `BirdsongModel_colormap_ver1.1.py`
- `BirdsongModel_colormap_ver1.2.py`
- `BirdsongModel_colormap_ver2.0.py`
- `README.md`

直下のディレクトリ（隠し項目を含む）：

- `.agents/`
- `.aws/`
- `.codex/`
- `.git/`
- `.venv/`
- `.vscode/`
- `Color map/`
- `Complex_2/`
- `Deeplearning/`
- `__pycache__/`
- `docs/`
- `legacy/`
- `results/`
- `ss/`
