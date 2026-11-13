# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
from scipy.io.wavfile import write

# --- ここで設定を変更できます ---
# 読み込むCSVファイル名を指定
# 例: 'simulation_output_timed_chaos.csv'
csv_filename = 'sim_output_eps_2.0e+07_ps_6.0e+06.csv'

# 出力するWAVファイル名
output_wav_filename = 'sim_output_eps_2.0e+07_ps_6.0e+06.wav'
# --------------------------------

print(f"'{csv_filename}' から音声データを読み込んでいます...")

# CSVファイルを読み込む
df = pd.read_csv(csv_filename)

pi = df['pi'].values
time = df['time'].values

# サンプリング周波数を計算
sampling_rate = int(1.0 / (time[1] - time[0])) if len(time) > 1 else 44100
print(f"サンプリング周波数: {sampling_rate} Hz")

# --- 音声データの前処理 ---

# 1. 正規化: piの値を -1.0 から 1.0 の範囲に収める
#    まず、絶対値の最大値で全体を割ります。
pi_normalized = pi / np.max(np.abs(pi))

# 2. 16ビット整数へスケーリング
#    WAVファイルで一般的な16ビット整数の範囲 (-32768 から 32767) に変換します。
audio_data = (pi_normalized * 32767).astype(np.int16)

# -------------------------

# WAVファイルとして書き出す
print(f"'{output_wav_filename}' に音声データを書き出しています...")
write(output_wav_filename, sampling_rate, audio_data)

print(f"音声ファイルの生成が完了しました！")

### 実行と再生の方法

#1.  **仮想環境を有効にする:**
 #   ```bash
  #  source venv/bin/activate
  #  ```
#2.  **スクリプトを実行する:**
#    ```bash
 #   python3 generate_audio.py
  #  ```
#3.  **音声ファイルを再生する:**
 #   スクリプトが完了すると`birdsong_output.wav`というファイルが作成されます。WSL環境からこのファイルを再生するには、以下のコマンドを実行するのが一番簡単です。

  #  ```bash
   # explorer.exe .