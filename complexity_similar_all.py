import pandas as pd
import numpy as np
import os

# ==========================================
# 設定: ここを変更してください
# ==========================================
# さえずりの複雑度データのファイル名
wav_file_path = 'wav_complexity_results_22k_limited.csv' 

# シミュレーションの複雑度データのファイルリスト
# 比較したいファイルを [] の中にカンマ区切りで列挙してください
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

# 出力するファイル名
output_file_path = 'wav_sim_matches_multi_top5.csv'

# 何位まで抽出するか
TOP_K = 5
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
    sim_col_name = 'raw_complexity' # カラム名は統一されている前提

    print("シミュレーションデータを読み込んでいます...")
    for sim_path in sim_file_paths:
        if not os.path.exists(sim_path):
            print(f"  [スキップ] ファイルが見つかりません: {sim_path}")
            continue
            
        try:
            df_temp = pd.read_csv(sim_path)
            
            # カラム確認
            if sim_col_name not in df_temp.columns:
                print(f"  [スキップ] {sim_path} に '{sim_col_name}' カラムがありません。")
                continue
                
            # どのファイル由来かわかるようにカラムを追加
            df_temp['Source_File'] = sim_path
            
            df_sim_list.append(df_temp)
            print(f"  [OK] 読み込み完了: {sim_path} ({len(df_temp)}件)")
            
        except Exception as e:
            print(f"  [エラー] {sim_path} の読み込み失敗: {e}")

    if not df_sim_list:
        print("有効なシミュレーションデータが1つもありませんでした。終了します。")
        return

    # 全データを1つに結合 (縦に積む)
    df_sim_all = pd.concat(df_sim_list, ignore_index=True)
    
    # NaN除去
    df_sim_clean = df_sim_all.dropna(subset=[sim_col_name]).copy()
    print(f"全シミュレーションデータ結合完了: 計 {len(df_sim_clean)} 件の有効データ")

    
    # 3. マッチング処理
    # さえずり側のカラム名確認
    wav_col_name = 'Complexity'
    if wav_col_name not in df_wav.columns:
        print(f"エラー: さえずりデータに '{wav_col_name}' カラムが見つかりません。")
        return

    print(f"マッチング計算を開始します... (Top {TOP_K})")
    results = []

    for idx, row in df_wav.iterrows():
        target_val = row[wav_col_name]
        filename = row.get('Filename', f'File_{idx}')
        
        # 差分の絶対値を計算
        df_sim_clean['diff'] = (df_sim_clean[sim_col_name] - target_val).abs()
        
        # 全ファイル合わせた中から、差が小さい順にトップK個を取得
        top_matches = df_sim_clean.nsmallest(TOP_K, 'diff')
        
        rank = 1
        for i, match_row in top_matches.iterrows():
            # 類似度計算
            similarity = (1 - match_row['diff']) * 100
            
            results.append({
                'Wav_Filename': filename,
                'Rank': rank,
                'Wav_Complexity': target_val,
                'Source_File': match_row['Source_File'], # どのファイルか
                'Sim_epsilon': match_row['epsilon'],
                'Sim_ps': match_row['ps'],
                'Sim_Complexity': match_row[sim_col_name],
                'Diff': match_row['diff'],
                'Similarity_Percent': similarity
            })
            rank += 1

    # 4. CSVに保存
    df_result = pd.DataFrame(results)
    # カラムの並び順を見やすく整理
    cols = ['Wav_Filename', 'Rank', 'Similarity_Percent', 'Diff', 'Source_File', 
            'Wav_Complexity', 'Sim_Complexity', 'Sim_epsilon', 'Sim_ps']
    df_result = df_result[cols]
    
    df_result.to_csv(output_file_path, index=False)
    
    print(f"完了しました。結果は '{output_file_path}' に保存されました。")

if __name__ == "__main__":
    main()