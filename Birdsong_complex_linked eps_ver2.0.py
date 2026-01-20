# 仮想環境に入るため，実行するときにまず source venv/bin/activate をターミナルで実行する．

import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.signal import spectrogram
from scipy.stats import entropy
import glob
import os
import re

# ==========================================
# ★★★ 設定エリア ★★★
# ==========================================

ANALYSIS_SETS = [
    # --- セット1 ---
    {
        "input_folders": [
            "simulation_results_x0=0.02_linked_low eps*0.6/",
            "simulation_results_x0=0.02_linked_eps*0.6/"
        ],
        "output_csv": "complexity_data_linked_eps_0.6.csv",
        "output_img": "complexity_3d_linked_eps_0.6.png"
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
# 関数定義（検問400Hz -> 計算100Hz）
# ==========================================

def calculate_spectral_entropy_limited(csv_filepath):
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

        # スペクトログラム計算
        f, t, Sxx = spectrogram(pi, fs=sampling_rate, window=window_type, nperseg=nperseg_local, noverlap=noverlap_local)
        
        # 平均スペクトル
        mean_spectrum_full = np.mean(Sxx, axis=1)

        # ---------------------------------------------------------
        # ★ステップ1: 「検問」 (ドリフト対策)
        # ---------------------------------------------------------
        # 0Hz付近の巨大な波を無視するため、判定だけは「100Hz以上」で行う
        CHECK_MIN_FREQ = 250.0
        CHECK_MAX_FREQ = 12000.0

        check_indices = (f >= CHECK_MIN_FREQ) & (f <= CHECK_MAX_FREQ)
        spectrum_for_check = mean_spectrum_full[check_indices]

        if spectrum_for_check.size == 0:
            return 0.0

        # ピークパワーで判定（閾値20万）
        peak_power_in_band = np.max(spectrum_for_check)
        SILENCE_THRESHOLD = 1e5 

        if peak_power_in_band < SILENCE_THRESHOLD:
            return 0.0  # 音が無いとみなす

        # ---------------------------------------------------------
        # ★ステップ2: 「計算」 (本番)
        # ---------------------------------------------------------
        # ★ここを「100Hz」に設定（低音成分もしっかり拾う！）
        CALC_MIN_FREQ = 250.0
        CALC_MAX_FREQ = 12000.0
        
        valid_indices = (f >= CALC_MIN_FREQ) & (f <= CALC_MAX_FREQ)
        Sxx_filtered = Sxx[valid_indices, :]
        
        if Sxx_filtered.size == 0:
            return 0.0

        mean_spectrum = np.mean(Sxx_filtered, axis=1)
        
        # 正規化してエントロピー計算
        total_power = np.sum(mean_spectrum)
        if total_power == 0:
            return 0.0
            
        psd_norm = mean_spectrum / total_power
        ent = entropy(psd_norm, base=2)
        max_entropy = np.log2(len(psd_norm))
        
        if max_entropy == 0:
            return 0.0

        return ent / max_entropy

    except Exception as e:
        print(f"Error processing {csv_filepath}: {e}")
        return 0.0

# ==========================================
# メイン処理
# ==========================================

print(f"解析モード: Linkedモデル (epsL使用, 計算250Hz～)")
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
            # raw_epsR = float(match.group(2)) # 今回は使わない
            raw_ps = float(match.group(3))

            epsL = float(f"{raw_epsL:.1e}")
            ps = float(f"{raw_ps:.1e}")
            
            # 複雑度計算
            complexity = calculate_spectral_entropy_limited(csv_filepath)
            
            results.append({
                'epsilon': epsL, # X軸には epsL を使う
                'ps': ps, 
                'raw_complexity': complexity
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
        res_map[(row['epsilon'], row['ps'])] = row['raw_complexity']

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
        current_dy = ps_diffs[i] * 0.8 # 細くする
        for j, eps in enumerate(epsilon_axis):
            val = Z_matrix[i, j]
            current_dx = eps_diffs[j] * 0.8 # 細くする
            
            x_pos.append(eps)
            y_pos.append(ps)
            z_pos.append(0)
            dx.append(current_dx)
            dy.append(current_dy)
            dz.append(val)

    colors_list = ['#ffffff', '#87CEFA', '#DC143C'] 
    cmap_custom = mcolors.LinearSegmentedColormap.from_list('white_to_red', colors_list)
    colors = np.array([cmap_custom(h) for h in dz])

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
    ax.set_zlabel('Complexity (0-22kHz)', fontsize=LABEL_FONTSIZE, labelpad=10)

    # 軸目盛：3つおき
    x_ticks = epsilon_axis[::3]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{val:.1e}" for val in x_ticks], rotation=45, fontsize=TICK_FONTSIZE)

    y_ticks = ps_axis[::3]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{val:.1e}" for val in y_ticks], fontsize=TICK_FONTSIZE, rotation=-15)

    ax.set_zlim(0, 1.0)
    ax.view_init(elev=30, azim=-60)
    plt.title(f'3D Complexity Map (Linked)\n{output_img}', fontsize=16)
    plt.tight_layout()

    plt.savefig(output_img, dpi=150)
    plt.close() 
    print(f"  完了: 画像を {output_img} に保存しました。")

print("\n全セットの処理が完了しました。")