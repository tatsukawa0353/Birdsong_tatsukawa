import pandas as pd
import numpy as np
import os

# ==========================================
# ★設定エリア
# ==========================================

# 1. さえずり解析結果（WAV）
wav_aci_csv = 'wav_aci_results_12k_limited.csv'
wav_ent_csv = 'wav_complexity_results_12k_limited.csv'

# 2. シミュレーション結果のファイルリスト
# ※リストの並び順が対応している必要があります（1番目同士、2番目同士がペアになります）
sim_aci_files = [
    'aci_data_1.csv',
    'aci_data_2.csv',
    'aci_data_1_f0=0.4e7.csv',
    'aci_data_1_f0=0.7e7.csv',
    'aci_data_1_f0=0.1e7.csv',
    'aci_data_1_f0=0.05e7.csv',
    'aci_data_1_f0=4.9e4.csv',
    'aci_data_linked_eps_0.6.csv',
]

sim_ent_files = [
    'complexity_data_1.csv',
    'complexity_data_2.csv',
    'complexity_data_1_f0=0.4e7.csv',
    'complexity_data_1_f0=0.7e7.csv',
    'complexity_data_1_f0=0.1e7.csv',
    'complexity_data_1_f0=0.05e7.csv',
    'complexity_data_1_f0=4.9e4.csv',
    'complexity_data_linked_eps_0.6.csv',
]

# 出力ファイル名
output_file = 'wav_sim_matches_2D_best_all_v2.csv'
TOP_K = 5  # 上位何個を残すか
# ==========================================

def load_and_merge_pair(aci_path, ent_path):
    """ ACIとSEのペアファイルを読み込んで結合する """
    try:
        if not os.path.exists(aci_path):
            print(f"  [Skip] ACI file not found: {aci_path}")
            return None
        if not os.path.exists(ent_path):
            print(f"  [Skip] SE file not found: {ent_path}")
            return None

        # 読み込み
        df_aci = pd.read_csv(aci_path)
        df_ent = pd.read_csv(ent_path)

        # カラム名の統一
        for c in df_aci.columns:
            if c in ['ACI', 'aci']: df_aci = df_aci.rename(columns={c: 'aci'})
        for c in df_ent.columns:
            if c in ['raw_complexity', 'Complexity']: df_ent = df_ent.rename(columns={c: 'entropy'})

        # 必要なカラムがあるか確認
        if 'aci' not in df_aci.columns or 'entropy' not in df_ent.columns:
            return None

        # 結合 (epsilon, ps がキー)
        # suffixesをつけることで、万が一カラム名が重複してもエラーを防ぐ
        merged = pd.merge(df_aci, df_ent, on=['epsilon', 'ps'], how='inner', suffixes=('', '_ent'))

        # ソースファイル名を記録 (ACI側のファイル名を代表とする)
        merged['Source_File'] = os.path.basename(aci_path)
        
        # 必要なデータだけ返す
        return merged[['epsilon', 'ps', 'aci', 'entropy', 'Source_File']]

    except Exception as e:
        print(f"  [Error] Failed to process pair ({aci_path}, {ent_path}): {e}")
        return None

def main():
    print("=== 解析開始 (ペアリング厳密化 & sklearn不使用版) ===")

    # ---------------------------------------------------------
    # 1. WAVデータの読み込み
    # ---------------------------------------------------------
    print("\nStep 1: さえずりデータを準備中...")
    try:
        df_w_aci = pd.read_csv(wav_aci_csv)
        df_w_ent = pd.read_csv(wav_ent_csv)

        if 'ACI' in df_w_aci.columns: df_w_aci = df_w_aci.rename(columns={'ACI': 'aci'})
        # エントロピーのカラム名対応
        col_map_ent = {c: 'entropy' for c in df_w_ent.columns if c in ['raw_complexity', 'Complexity']}
        df_w_ent = df_w_ent.rename(columns=col_map_ent)

        df_wav = pd.merge(df_w_aci, df_w_ent, on='Filename', how='inner')
        print(f"  対象さえずり数: {len(df_wav)} 件")
    except Exception as e:
        print(f"  [エラー] WAVデータの読み込み失敗: {e}")
        return

    # ---------------------------------------------------------
    # 2. シミュレーションデータの読み込み (ペアごとに処理)
    # ---------------------------------------------------------
    print("\nStep 2: シミュレーションデータをペアごとに読み込み中...")
    
    df_sim_list = []
    
    # zipを使って ACIファイル と SEファイル を1つずつ取り出して処理
    for aci_f, ent_f in zip(sim_aci_files, sim_ent_files):
        print(f"  Processing pair: {os.path.basename(aci_f)} + {os.path.basename(ent_f)}")
        merged_df = load_and_merge_pair(aci_f, ent_f)
        
        if merged_df is not None and not merged_df.empty:
            df_sim_list.append(merged_df)
            print(f"    -> Combined: {len(merged_df)} rows")
        else:
            print("    -> No match or empty.")

    if not df_sim_list:
        print("\n[エラー] 有効なシミュレーションデータが1つもありませんでした。")
        return

    # 全ペアの結果を1つのDataFrameにまとめる
    df_sim = pd.concat(df_sim_list, ignore_index=True)
    
    # 欠損値除去
    df_sim = df_sim.dropna(subset=['aci', 'entropy'])
    print(f"\n  有効な全シミュレーション解: {len(df_sim)} 件")
    print("  データの内訳:")
    print(df_sim['Source_File'].value_counts())

    # ---------------------------------------------------------
    # 3. マッチング計算 (手動標準化)
    # ---------------------------------------------------------
    print("\nStep 3: マッチング計算開始...")

    # --- 標準化 (Z-score normalization) ---
    # 平均と標準偏差を計算 (全シミュレーションデータを母集団とする)
    sim_aci_mean = df_sim['aci'].mean()
    sim_aci_std = df_sim['aci'].std()
    sim_ent_mean = df_sim['entropy'].mean()
    sim_ent_std = df_sim['entropy'].std()

    # ゼロ除算防止
    if sim_aci_std == 0: sim_aci_std = 1.0
    if sim_ent_std == 0: sim_ent_std = 1.0

    # シミュレーションデータを標準化
    # (値 - 平均) / 標準偏差
    # PandasのSeries同士の演算で一括処理
    sim_aci_norm = (df_sim['aci'] - sim_aci_mean) / sim_aci_std
    sim_ent_norm = (df_sim['entropy'] - sim_ent_mean) / sim_ent_std

    # WAVデータも「同じ基準」で標準化
    wav_aci_norm = (df_wav['aci'] - sim_aci_mean) / sim_aci_std
    wav_ent_norm = (df_wav['entropy'] - sim_ent_mean) / sim_ent_std

    # 高速化のためNumPy配列に変換
    # X_sim_scaled: [ [aci_norm, ent_norm], ... ]
    X_sim_scaled = np.column_stack((sim_aci_norm.values, sim_ent_norm.values))
    X_wav_scaled = np.column_stack((wav_aci_norm.values, wav_ent_norm.values))

    results = []

    for i, row_wav in df_wav.iterrows():
        target_vec = X_wav_scaled[i]  # ターゲットの座標
        filename = row_wav['Filename']

        # 全シミュレーションとのユークリッド距離を一括計算
        # axis=1 は行方向（(x1-x2)^2 + (y1-y2)^2）の和
        diff_sq = (X_sim_scaled - target_vec) ** 2
        distances = np.sqrt(np.sum(diff_sq, axis=1))

        # 計算結果を一時的にDataFrameに入れてソート
        df_calc = df_sim.copy()
        df_calc['distance'] = distances

        # 距離が小さい順にTop Kを取得
        top_matches = df_calc.nsmallest(TOP_K, 'distance')

        rank = 1
        for _, match in top_matches.iterrows():
            results.append({
                'Wav_Filename': filename,
                'Rank': rank,
                'Distance': match['distance'],
                'Wav_ACI': row_wav['aci'],
                'Wav_Ent': row_wav['entropy'],
                'Sim_ACI': match['aci'],
                'Sim_Ent': match['entropy'],
                'Sim_eps': match['epsilon'],
                'Sim_ps': match['ps'],
                'Source_File': match['Source_File']
            })
            rank += 1

    # ---------------------------------------------------------
    # 4. 保存
    # ---------------------------------------------------------
    df_result = pd.DataFrame(results)

    # カラム並べ替え
    cols = ['Wav_Filename', 'Rank', 'Distance', 'Sim_eps', 'Sim_ps',
            'Wav_ACI', 'Sim_ACI', 'Wav_Ent', 'Sim_Ent', 'Source_File']
    # 存在確認してからフィルタ
    cols = [c for c in cols if c in df_result.columns]
    df_result = df_result[cols]

    df_result.to_csv(output_file, index=False)
    print(f"\n完了しました！ 結果ファイル: {output_file}")
    print("Distanceの値が小さいほど、ACIとSEの両方が似ていることを示します。")

if __name__ == "__main__":
    main()