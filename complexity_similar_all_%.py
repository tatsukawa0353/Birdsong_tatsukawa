import pandas as pd
import numpy as np
import os

# ==========================================
# 設定: ここを変更してください
# ==========================================
wav_file_path = 'wav_complexity_results_12k_limited.csv' 

sim_file_paths = [
    'complexity_data_1.csv',
     'complexity_data_2.csv',  # 追加したいファイルがあればコメントを外して記述
     'complexity_data_1_f0=0.4e7.csv',
     'complexity_data_1_f0=0.7e7.csv',
     'complexity_data_1_f0=0.1e7.csv',
     'complexity_data_1_f0=0.05e7.csv',
     'complexity_data_1_f0=4.9e4.csv',
     'complexity_data_linked_eps_0.6.csv',

]

output_file_path = 'wav_sim_matches_95%.csv'

# 類似度が何パーセント以上のものを抽出するか
SIMILARITY_THRESHOLD = 95.0 
# ==========================================

def main():
    # 1. さえずりデータの読み込み
    try:
        df_wav = pd.read_csv(wav_file_path)
        print(f"さえずりデータを読み込みました: {wav_file_path}")
    except FileNotFoundError:
        print(f"エラー: さえずりファイルが見つかりません: {wav_file_path}")
        return

    # 2. 複数のシミュレーションデータを読み込んで結合
    df_sim_list = []
    sim_col_name = 'raw_complexity'

    print("シミュレーションデータを読み込んでいます...")
    for sim_path in sim_file_paths:
        if not os.path.exists(sim_path):
            print(f"  [スキップ] ファイルが見つかりません: {sim_path}")
            continue
            
        try:
            df_temp = pd.read_csv(sim_path)
            if sim_col_name not in df_temp.columns:
                print(f"  [スキップ] {sim_path} にカラムがありません。")
                continue
            
            df_temp['Source_File'] = sim_path
            df_sim_list.append(df_temp)
            print(f"  [OK] 読み込み完了: {sim_path} ({len(df_temp)}件)")
            
        except Exception as e:
            print(f"  [エラー] {sim_path} の読み込み失敗: {e}")

    if not df_sim_list:
        print("有効なデータがありません。終了します。")
        return

    df_sim_all = pd.concat(df_sim_list, ignore_index=True)
    df_sim_clean = df_sim_all.dropna(subset=[sim_col_name]).copy()
    print(f"全シミュレーションデータ結合完了: 計 {len(df_sim_clean)} 件")

    # 3. マッチング処理
    wav_col_name = 'Complexity'
    if wav_col_name not in df_wav.columns:
        print(f"エラー: さえずりデータに '{wav_col_name}' カラムが見つかりません。")
        return

    print(f"マッチング計算を開始します... (閾値: {SIMILARITY_THRESHOLD}%)")
    results = []

    for idx, row in df_wav.iterrows():
        target_val = row[wav_col_name]
        filename = row.get('Filename', f'File_{idx}')
        
        # 差分と類似度を一括計算
        diffs = (df_sim_clean[sim_col_name] - target_val).abs()
        similarities = (1 - diffs) * 100
        
        # 閾値以上のインデックスを取得
        mask = similarities >= SIMILARITY_THRESHOLD
        
        # 該当する行だけ抽出
        matches = df_sim_clean[mask].copy()
        
        if matches.empty:
            continue
            
        # 必要な情報を付与
        matches['Wav_Filename'] = filename
        matches['Wav_Complexity'] = target_val
        matches['Diff'] = diffs[mask]
        matches['Similarity_Percent'] = similarities[mask]
        
        results.append(matches)

    # 4. CSVに保存
    if results:
        df_result = pd.concat(results, ignore_index=True)

        # 【修正】ここで列名を統一します
        # raw_complexity -> Sim_Complexity
        # epsilon -> Sim_epsilon
        # ps -> Sim_ps
        df_result = df_result.rename(columns={
            sim_col_name: 'Sim_Complexity',
            'epsilon': 'Sim_epsilon',
            'ps': 'Sim_ps'
        })
        
        # カラム整理（指定した順序で並べる）
        cols = ['Wav_Filename', 'Similarity_Percent', 'Diff', 'Source_File', 
                'Wav_Complexity', 'Sim_Complexity', 'Sim_epsilon', 'Sim_ps']
        
        # 存在しないカラムは無視してフィルタリング
        cols = [c for c in cols if c in df_result.columns]
        df_result = df_result[cols]
        
        # 並び替え: 類似度が高い順 (Descending)
        df_result = df_result.sort_values(by=['Wav_Filename', 'Similarity_Percent'], ascending=[True, False])
        
        df_result.to_csv(output_file_path, index=False)
        print(f"完了しました。結果は '{output_file_path}' に保存されました。")
        print(f"抽出されたデータ数: {len(df_result)} 行")
        print(df_result.head()) # 確認用に出力
    else:
        print(f"閾値 {SIMILARITY_THRESHOLD}% を超えるマッチングは見つかりませんでした。")

if __name__ == "__main__":
    main()