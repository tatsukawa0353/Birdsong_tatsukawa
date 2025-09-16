# 仮想環境に入るため，実行するときにまずsource venv/bin/activateをターミナルで実行する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# CSVファイルを読み込む
df = pd.read_csv('simulation_output_2(a).csv')

time = df['time'].values
x_left = df['x_left'].values
x_right = df['x_right'].values



#