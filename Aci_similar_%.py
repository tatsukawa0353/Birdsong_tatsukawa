import pandas as pd
import numpy as np
import os

# ==========================================
# 設定: ここを変更してください
# ==========================================
# さえずり(WAV)のACI解析結果ファイル
wav_file_path = 'wav_aci_results_12k_limited.csv' 

# 比較対象のシミュレーションデータ(ACI)
sim_file_paths = [
    #'aci_data_1.csv',
    #'aci_data_2.csv',
    #'aci_data_1_f0=0.4e7.csv',
    #'aci_data_1_f0=0.7e7.csv',
    #'aci_data_1_f0=0.1e7.csv',
    #'aci_data_1_f0=0.05e7.csv',
    #'aci_data_1_f0=4.9e4.csv',
    'aci_data_linked_eps_0.6.csv',
]

output_file_path = 'wav_sim_matches_aci_linked_eps_0.6.csv'

# 類似度が何パーセント以上のものを抽出するか
SIMILARITY_THRESHOLD = 95.0 
# ==========================================

def main():
    # 1. さえずりデータの読み込み
    try:
        df_wav = pd.read_csv(wav_file_path)
        # カラム名の揺らぎ吸収 (ACI or aci)
        if 'ACI' in df_wav.columns:
            wav_col_name = 'ACI'
        elif 'aci' in df_wav.columns:
            wav_col_name = 'aci'
        else:
            print(f"エラー: さえずりデータに 'ACI' または 'aci' カラムが見つかりません。")
            return
            
        print(f"さえずりデータを読み込みました: {wav_file_path} (Column: {wav_col_name})")
    except FileNotFoundError:
        print(f"エラー: さえずりファイルが見つかりません: {wav_file_path}")
        return

    # 2. 複数のシミュレーションデータを読み込んで結合
    df_sim_list = []
    sim_col_name = 'aci' 

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
    print(f"マッチング計算を開始します... (閾値: {SIMILARITY_THRESHOLD}%)")
    results = []

    for idx, row in df_wav.iterrows():
        target_val = row[wav_col_name]
        filename = row.get('Filename', f'File_{idx}')
        
        # --- ACI用の類似度計算 (ターゲット基準) ---
        # 変更点: 分母を「WAVのACI値」に固定しました。
        # 類似度(%) = 100 * (1 - |diff| / wav_val)
        
        sim_vals = df_sim_clean[sim_col_name]
        
        # 差分の絶対値
        diffs = (sim_vals - target_val).abs()
        
        # 分母: WAVの値を使用 (ゼロ除算防止)
        denom = max(target_val, 1e-9)
        
        # 類似度計算
        # ※ずれが大きい場合、マイナスになることがありますが、閾値フィルタで弾かれます
        similarities = (1.0 - (diffs / denom)) * 100.0
        
        # 閾値以上のインデックスを取得
        mask = similarities >= SIMILARITY_THRESHOLD
        
        # 該当する行だけ抽出
        matches = df_sim_clean[mask].copy()
        
        if matches.empty:
            continue
            
        # 必要な情報を付与
        matches['Wav_Filename'] = filename
        matches['Wav_ACI'] = target_val
        matches['Diff'] = diffs[mask]
        matches['Similarity_Percent'] = similarities[mask]
        
        results.append(matches)

    # 4. CSVに保存
    if results:
        df_result = pd.concat(results, ignore_index=True)

        # カラム名変更
        df_result = df_result.rename(columns={
            sim_col_name: 'Sim_ACI',
            'epsilon': 'Sim_epsilon',
            'ps': 'Sim_ps'
        })
        
        # カラム整理
        cols = ['Wav_Filename', 'Similarity_Percent', 'Diff', 'Source_File', 
                'Wav_ACI', 'Sim_ACI', 'Sim_epsilon', 'Sim_ps']
        
        # 存在しないカラムは無視
        cols = [c for c in cols if c in df_result.columns]
        df_result = df_result[cols]
        
        # 並び替え: ファイル名順 -> 類似度順(降順)
        df_result = df_result.sort_values(by=['Wav_Filename', 'Similarity_Percent'], ascending=[True, False])
        
        df_result.to_csv(output_file_path, index=False)
        print(f"完了しました。結果は '{output_file_path}' に保存されました。")
        print(f"抽出されたデータ数: {len(df_result)} 行")
        print(df_result.head()) 
    else:
        print(f"閾値 {SIMILARITY_THRESHOLD}% を超えるマッチングは見つかりませんでした。")

if __name__ == "__main__":
    main()