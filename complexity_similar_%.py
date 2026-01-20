import pandas as pd
import numpy as np

# ==========================================
# 設定: ファイル名をここで指定してください
# ==========================================
# さえずりの複雑度データのファイル名
wav_file_path = 'wav_complexity_results_12k_limited.csv' 
# シミュレーションの複雑度データのファイル名
sim_file_path = 'complexity_data_linked_eps_0.6.csv'
# 出力するファイル名
output_file_path = 'wav_sim_matches_linked_eps_0.6_95%.csv'

# 類似度が何パーセント以上のものを抽出するか
SIMILARITY_THRESHOLD = 95.0 
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

    # 3. マッチング処理 (閾値フィルタリング)
    print(f"マッチング計算を開始します... (閾値: {SIMILARITY_THRESHOLD}%)")
    results = []

    for idx, row in df_wav.iterrows():
        target_val = row[wav_col_name]
        filename = row.get('Filename', f'File_{idx}')
        
        # 差分と類似度を一括計算
        # 高速化のため、Seriesとして計算
        diffs = (df_sim_clean[sim_col_name] - target_val).abs()
        similarities = (1 - diffs) * 100
        
        # 閾値以上のものだけ抽出 (Boolean Indexing)
        mask = similarities >= SIMILARITY_THRESHOLD
        
        # 該当する行だけ取り出す
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

        # 列名を統一 (Sim_epsilon, Sim_ps など)
        df_result = df_result.rename(columns={
            sim_col_name: 'Sim_Complexity',
            'epsilon': 'Sim_epsilon',
            'ps': 'Sim_ps'
        })
        
        # カラムの整理
        cols = ['Wav_Filename', 'Similarity_Percent', 'Diff', 
                'Wav_Complexity', 'Sim_Complexity', 'Sim_epsilon', 'Sim_ps']
        # 存在しないカラムは除外
        cols = [c for c in cols if c in df_result.columns]
        df_result = df_result[cols]
        
        # 並び替え設定
        # 1. Wav_Filename (A->Z順)
        # 2. Similarity_Percent (高い順)
        df_result = df_result.sort_values(by=['Wav_Filename', 'Similarity_Percent'], ascending=[True, False])
        
        df_result.to_csv(output_file_path, index=False)
        
        print(f"完了しました。結果は '{output_file_path}' に保存されました。")
        print(f"抽出されたデータ数: {len(df_result)} 行")
        print(df_result.head()) # 確認用表示
    else:
        print(f"閾値 {SIMILARITY_THRESHOLD}% を超えるマッチングは見つかりませんでした。")

if __name__ == "__main__":
    main()