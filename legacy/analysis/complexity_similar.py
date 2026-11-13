import pandas as pd
import numpy as np

# ==========================================
# 設定: ファイル名をここで指定してください
# ==========================================
wav_file_path = 'wav_complexity_results_12k_limited.csv' 
sim_file_path = 'complexity_data_2.csv'
output_file_path = 'wav_sim_matches_top5_2.csv'
TOP_K = 5
# ==========================================

def main():
    # 1. データの読み込み
    try:
        df_wav = pd.read_csv(wav_file_path)
        df_sim = pd.read_csv(sim_file_path)
        print("ファイルを読み込みました。")
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください。\n{e}")
        return

    # 2. データの前処理
    sim_col_name = 'raw_complexity'
    if sim_col_name not in df_sim.columns:
        print(f"警告: シミュレーションデータに '{sim_col_name}' カラムが見つかりません。")
        return
    
    df_sim_clean = df_sim.dropna(subset=[sim_col_name]).copy()
    wav_col_name = 'Complexity'
    if wav_col_name not in df_wav.columns:
        print(f"警告: さえずりデータに '{wav_col_name}' カラムが見つかりません。")
        return

    # 3. マッチング処理
    print(f"マッチング計算を開始します... (Top {TOP_K})")
    results = []

    for idx, row in df_wav.iterrows():
        target_val = row[wav_col_name]
        filename = row.get('Filename', f'File_{idx}')
        
        # 差分の絶対値を計算
        df_sim_clean['diff'] = (df_sim_clean[sim_col_name] - target_val).abs()
        
        # 差が小さい順にトップK個を取得
        top_matches = df_sim_clean.nsmallest(TOP_K, 'diff')
        
        rank = 1
        for i, match_row in top_matches.iterrows():
            # 類似度の計算 (1 - 差分) * 100
            # 複雑度が0~1の範囲である前提
            similarity = (1 - match_row['diff']) * 100
            
            results.append({
                'Wav_Filename': filename,
                'Rank': rank,
                'Wav_Complexity': target_val,
                'Sim_epsilon': match_row['epsilon'],
                'Sim_ps': match_row['ps'],
                'Sim_Complexity': match_row[sim_col_name],
                'Diff': match_row['diff'],
                'Similarity_Percent': similarity # 追加した項目
            })
            rank += 1

    # 4. CSVに保存
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file_path, index=False)
    
    print(f"完了しました。結果は '{output_file_path}' に保存されました。")
    print(df_result.head()) # 確認用表示

if __name__ == "__main__":
    main()