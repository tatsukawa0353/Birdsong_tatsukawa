import pandas as pd
import numpy as np
import os

# ==========================================
# 設定: ここを変更してください
# ==========================================
# さえずり(WAV)のACI解析結果ファイル
wav_file_path = 'wav_aci_results_12k_limited.csv' 

# 比較対象のシミュレーションデータ(ACI)のリスト
sim_file_paths = [
    'aci_data_1.csv',
    'aci_data_2.csv',
    'aci_data_1_f0=0.4e7.csv',
    'aci_data_1_f0=0.7e7.csv',
    'aci_data_1_f0=0.1e7.csv',
    'aci_data_1_f0=0.05e7.csv',
    'aci_data_1_f0=4.9e4.csv',
    'aci_data_linked_eps_0.6.csv',
]

# 出力するファイル名
output_file_path = 'wav_sim_matches_aci_multi_top5.csv'

# 何位まで抽出するか
TOP_K = 5
# ==========================================

def main():
    # 1. さえずりデータの読み込み
    try:
        df_wav = pd.read_csv(wav_file_path)
        
        # カラム名の対応 (ACI or aci)
        if 'ACI' in df_wav.columns:
            wav_col_name = 'ACI'
        elif 'aci' in df_wav.columns:
            wav_col_name = 'aci'
        else:
            print(f"エラー: さえずりデータに 'ACI' または 'aci' カラムが見つかりません。")
            return
            
        print(f"さえずりデータを読み込みました: {wav_file_path}")
    except FileNotFoundError:
        print(f"エラー: さえずりファイルが見つかりません: {wav_file_path}")
        return

    # 2. 複数のシミュレーションデータを読み込んで結合
    df_sim_list = []
    sim_col_name = 'aci' # シミュレーション側のカラム名

    print("シミュレーションデータを読み込んでいます...")
    for sim_path in sim_file_paths:
        if not os.path.exists(sim_path):
            print(f"  [スキップ] ファイルが見つかりません: {sim_path}")
            continue
            
        try:
            df_temp = pd.read_csv(sim_path)
            
            # カラム名の確認
            if sim_col_name not in df_temp.columns:
                if 'ACI' in df_temp.columns:
                    df_temp = df_temp.rename(columns={'ACI': 'aci'})
                else:
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
    print(f"マッチング計算を開始します... (Top {TOP_K})")
    results = []

    for idx, row in df_wav.iterrows():
        target_val = row[wav_col_name]
        filename = row.get('Filename', f'File_{idx}')
        
        # --- ACI用の類似度計算 (ターゲット基準) ---
        # 差分の絶対値
        sim_vals = df_sim_clean[sim_col_name]
        diffs = (sim_vals - target_val).abs()
        
        # 類似度(%) = 100 * (1 - 誤差率)
        # 分母は実データの値 (ゼロ除算防止)
        denom = max(target_val, 1e-9)
        similarities = (1.0 - (diffs / denom)) * 100.0
        
        # 計算結果を一時的にデータフレームに追加
        # (コピーを作って警告回避)
        df_calc = df_sim_clean.copy()
        df_calc['diff'] = diffs
        df_calc['similarity'] = similarities
        
        # 差が小さい順（または類似度が高い順）にトップK個を取得
        top_matches = df_calc.nsmallest(TOP_K, 'diff')
        
        rank = 1
        for i, match_row in top_matches.iterrows():
            results.append({
                'Wav_Filename': filename,
                'Rank': rank,
                'Similarity_Percent': match_row['similarity'], # 計算済みの類似度
                'Diff': match_row['diff'],
                'Source_File': match_row['Source_File'],
                'Wav_ACI': target_val,
                'Sim_ACI': match_row[sim_col_name],
                'Sim_epsilon': match_row['epsilon'],
                'Sim_ps': match_row['ps']
            })
            rank += 1

    # 4. CSVに保存
    if results:
        df_result = pd.DataFrame(results)
        
        # カラムの並び順を見やすく整理
        cols = ['Wav_Filename', 'Rank', 'Similarity_Percent', 'Diff', 'Source_File', 
                'Wav_ACI', 'Sim_ACI', 'Sim_epsilon', 'Sim_ps']
        
        # 存在しないカラムは除外
        cols = [c for c in cols if c in df_result.columns]
        df_result = df_result[cols]
        
        df_result.to_csv(output_file_path, index=False)
        print(f"完了しました。結果は '{output_file_path}' に保存されました。")
        print(df_result.head())
    else:
        print("マッチング結果がありませんでした。")

if __name__ == "__main__":
    main()