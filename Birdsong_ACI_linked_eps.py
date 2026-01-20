# 仮想環境に入るため，実行するときにまず source venv/bin/activate をターミナルで実行する．

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.signal import spectrogram
import glob
import os
import re

# ==========================================
# ★★★ 設定エリア ★★★
# ==========================================

# グラフのZ軸（高さ）の最大値を固定
Z_AXIS_LIMIT = 250.0

ANALYSIS_SETS = [
    # --- セット1 ---
    {
        "input_folders": [
            "simulation_results_x0=0.02_linked_low eps*0.6/",
            "simulation_results_x0=0.02_linked_eps*0.6/"
        ],
        # ファイル名をACI用に変更
        "output_csv": "aci_data_linked_eps_0.6.csv",
        "output_img": "aci_3d_linked_eps_0.6.png"
    },
    # 必要に応じてセットを追加してください
]

# 分析パラメータ
nperseg_local = 245760 
noverlap_local = 184320
window_type = 'blackmanharris'

# ★解析開始時刻（秒）
START_TIME = 0.0

# ==========================================
# 関数定義（ACI計算: ノイズ除去版）
# ==========================================

def calculate_aci_limited(csv_filepath):
    """ 
    ACI (Acoustic Complexity Index) 計算関数
    Linkedモデル用
    - 絶対閾値(2e5)で無音をカット
    - 相対閾値(1%)で背景ノイズをカット
    """
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty or 'pi' not in df or 'time' not in df:
            return 0.0
        
        # 時間フィルタリング
        df_segment = df[df['time'] >= START_TIME]
        if len(df_segment) < nperseg_local:
            return 0.0
            
        sampling_rate = 1.0 / (df['time'].values[1] - df['time'].values[0])
        pi = df_segment['pi'].values 

        # 1. スペクトログラム計算
        f, t, Sxx = spectrogram(pi, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
        
        # ---------------------------------------------------------
        # ★ステップ1: 計算対象の周波数帯を抽出
        # ---------------------------------------------------------
        CALC_MIN_FREQ = 250.0
        CALC_MAX_FREQ = 12000.0
        
        valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
        Sxx_filtered = Sxx[valid_indices, :]
        
        if Sxx_filtered.size == 0:
            return 0.0

        # ---------------------------------------------------------
        # ★ステップ2: ノイズフロアの除去（重要）
        # ---------------------------------------------------------
        # 周波数ビンごとのパワー総和
        sum_power = np.sum(Sxx_filtered, axis=1)
        
        # このスペクトル内で「最も音が大きい周波数」のパワーを取得
        max_power_in_spectrum = np.max(sum_power)
        
        # --- 絶対閾値による無音判定 ---
        # 低ps領域の計算ノイズを弾くため、高めの値を設定 (2e5)
        SILENCE_ABSOLUTE_THRESHOLD = 1e5 
        
        if max_power_in_spectrum < SILENCE_ABSOLUTE_THRESHOLD:
            return 0.0

        # --- 相対閾値による背景ノイズ除去 ---
        # 「最大パワーの 1% (0.01) 未満」の周波数帯は背景ノイズとみなす
        NOISE_THRESHOLD_RATIO = 0.01 
        power_threshold = max_power_in_spectrum * NOISE_THRESHOLD_RATIO

        # ---------------------------------------------------------
        # ★ステップ3: ACI計算
        # ---------------------------------------------------------
        # 時間方向の差分の絶対値 |I_t - I_{t-1}|
        diff_abs = np.abs(np.diff(Sxx_filtered, axis=1))
        
        # 差分の総和
        sum_diff = np.sum(diff_abs, axis=1)
        
        # ACI = (差分の総和) / (パワーの総和)
        with np.errstate(divide='ignore', invalid='ignore'):
            aci_per_bin = sum_diff / sum_power
            
            # ★閾値処理: パワーが閾値以下のビン（背景）はACIを0にする
            aci_per_bin[sum_power < power_threshold] = 0.0
            
            aci_per_bin = np.nan_to_num(aci_per_bin)

        # 合計ACI値を返す
        total_aci = np.sum(aci_per_bin)
        
        return total_aci

    except Exception as e:
        print(f"Error processing {csv_filepath}: {e}")
        return 0.0

# ==========================================
# メイン処理
# ==========================================

print(f"解析モード: Linkedモデル ACI解析 (閾値処理済み)")
# 正規表現：epsL と epsR に対応
pattern = re.compile(r"sim_output_epsL_([0-9\.e\+\-]+)_epsR_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

for set_idx, config in enumerate(ANALYSIS_SETS):
    target_folders = config["input_folders"]
    output_csv = config["output_csv"]
    output_img = config["output_img"]

    print("\n" + "="*60)
    print(f"解析セット {set_idx + 1} を処理中...")
    print(f"  入力フォルダ: {target_folders}")
    print("="*60)

    results = [] 

    # --- データ収集 ---
    for folder in target_folders:
        if not os.path.exists(folder):
            print(f"  [警告] フォルダが見つかりません: {folder}")
            continue
        
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"  フォルダ読み込み: {folder} ({len(csv_files)} files)")
        
        for i, csv_filepath in enumerate(csv_files):
            filename = os.path.basename(csv_filepath)
            match = pattern.match(filename)
            if not match: continue

            # epsL, epsR, ps を取得
            raw_epsL = float(match.group(1))
            # raw_epsR = float(match.group(2)) 
            raw_ps = float(match.group(3))

            epsL = float(f"{raw_epsL:.1e}")
            ps = float(f"{raw_ps:.1e}")
            
            # ACI計算
            aci_val = calculate_aci_limited(csv_filepath)
            
            results.append({
                'epsilon': epsL, # X軸には epsL を使う
                'ps': ps, 
                'aci': aci_val
            })
            
            if (i+1) % 100 == 0: print(f"    Processing... {i+1} files done")

    # --- データ保存 ---
    if not results:
        print("  [エラー] 有効なデータが見つかりませんでした。スキップします。")
        continue

    df_results = pd.DataFrame(results)
    df_results = df_results.drop_duplicates(subset=['epsilon', 'ps'], keep='last')
    df_results = df_results.sort_values(by=['epsilon', 'ps'])

    print(f"  データを {output_csv} に保存中...")
    df_results.to_csv(output_csv, index=False)

    # --- グラフ描画 ---
    print("  3Dグラフを作成中...")
    
    epsilon_axis = sorted(df_results['epsilon'].unique())
    ps_axis = sorted(df_results['ps'].unique())

    Z_matrix = np.zeros((len(ps_axis), len(epsilon_axis)))
    res_map = {}
    for _, row in df_results.iterrows():
        res_map[(row['epsilon'], row['ps'])] = row['aci']

    for i, ps in enumerate(ps_axis):
        for j, eps in enumerate(epsilon_axis):
            Z_matrix[i, j] = res_map.get((eps, ps), 0.0)

    x_pos, y_pos, z_pos = [], [], []
    dx, dy, dz = [], [], []

    # バーの太さ・間隔計算
    ps_diffs = np.diff(ps_axis) if len(ps_axis) > 1 else np.array([1.0])
    ps_diffs = np.append(ps_diffs, ps_diffs[-1]) if len(ps_diffs) > 0 else np.array([1.0])
    
    eps_diffs = np.diff(epsilon_axis) if len(epsilon_axis) > 1 else np.array([1.0])
    eps_diffs = np.append(eps_diffs, eps_diffs[-1]) if len(eps_diffs) > 0 else np.array([1.0])

    for i, ps in enumerate(ps_axis):
        current_dy = ps_diffs[i] * 0.8 
        for j, eps in enumerate(epsilon_axis):
            val = Z_matrix[i, j]
            current_dx = eps_diffs[j] * 0.8 
            
            x_pos.append(eps)
            y_pos.append(ps)
            z_pos.append(0)
            dx.append(current_dx)
            dy.append(current_dy)
            dz.append(val)

    # カラーマップ
    colors_list = ['#ffffff', '#87CEFA', '#DC143C'] 
    cmap_custom = mcolors.LinearSegmentedColormap.from_list('white_to_red', colors_list)
    
    # 正規化も上限400に合わせる
    norm = mcolors.Normalize(vmin=0, vmax=Z_AXIS_LIMIT)
    colors = np.array([cmap_custom(norm(h)) for h in dz])

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)
    z_pos = np.array(z_pos)
    dx = np.array(dx)
    dy = np.array(dy)
    dz = np.array(dz)
    colors = np.array(colors)

    mask = dz > 0.001
    x_pos = x_pos[mask]
    y_pos = y_pos[mask]
    z_pos = z_pos[mask]
    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]
    colors = colors[mask]

    score = x_pos + y_pos 
    sort_order = np.argsort(score)[::-1] 
    x_pos = x_pos[sort_order]
    y_pos = y_pos[sort_order]
    z_pos = z_pos[sort_order]
    dx = dx[sort_order]
    dy = dy[sort_order]
    dz = dz[sort_order]
    colors = colors[sort_order]

    # プロット作成
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 10
    ax.set_xlabel('Left Epsilon (ε)', fontsize=LABEL_FONTSIZE, labelpad=20)
    ax.set_ylabel('Pressure (ps)', fontsize=LABEL_FONTSIZE, labelpad=20)
    ax.set_zlabel('ACI Value', fontsize=LABEL_FONTSIZE, labelpad=10)

    x_ticks = epsilon_axis[::3]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{val:.1e}" for val in x_ticks], rotation=45, fontsize=TICK_FONTSIZE)

    y_ticks = ps_axis[::3]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{val:.1e}" for val in y_ticks], fontsize=TICK_FONTSIZE, rotation=-15)

    # Z軸固定
    ax.set_zlim(0, Z_AXIS_LIMIT)
    
    ax.view_init(elev=30, azim=-60)
    plt.title(f'3D ACI Map (Linked)\n{output_img}', fontsize=16)
    plt.tight_layout()

    plt.savefig(output_img, dpi=150)
    plt.close() 
    print(f"  完了: 画像を {output_img} に保存しました。")

print("\n全セットの処理が完了しました。")