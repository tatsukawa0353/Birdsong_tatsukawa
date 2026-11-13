# 従来コードの参照調査

移動前のファイル名と行番号。import/include、入出力パスの静的抽出。コメント・未使用の設定も含むため、実行可能性を保証する一覧ではありません。

### Aci_similar_%.py
1: import pandas as pd
2: import numpy as np
3: import os
9: wav_file_path = 'wav_aci_results_12k_limited.csv' 
13:     #'aci_data_1.csv',
14:     #'aci_data_2.csv',
15:     #'aci_data_1_f0=0.4e7.csv',
16:     #'aci_data_1_f0=0.7e7.csv',
17:     #'aci_data_1_f0=0.1e7.csv',
18:     #'aci_data_1_f0=0.05e7.csv',
19:     #'aci_data_1_f0=4.9e4.csv',
20:     'aci_data_linked_eps_0.6.csv',
23: output_file_path = 'wav_sim_matches_aci_linked_eps_0.6.csv'

### Aci_similar_all.py
1: import pandas as pd
2: import numpy as np
3: import os
9: wav_file_path = 'wav_aci_results_12k_limited.csv' 
13:     'aci_data_1.csv',
14:     'aci_data_2.csv',
15:     'aci_data_1_f0=0.4e7.csv',
16:     'aci_data_1_f0=0.7e7.csv',
17:     'aci_data_1_f0=0.1e7.csv',
18:     'aci_data_1_f0=0.05e7.csv',
19:     'aci_data_1_f0=4.9e4.csv',
20:     'aci_data_linked_eps_0.6.csv',
24: output_file_path = 'wav_sim_matches_aci_multi_top5.csv'

### Aci_similar_all_%.py
1: import pandas as pd
2: import numpy as np
3: import os
9: wav_file_path = 'wav_aci_results_12k_limited.csv' 
13:     'aci_data_1.csv',
14:     'aci_data_2.csv',
15:     'aci_data_1_f0=0.4e7.csv',
16:     'aci_data_1_f0=0.7e7.csv',
17:     'aci_data_1_f0=0.1e7.csv',
18:     'aci_data_1_f0=0.05e7.csv',
19:     'aci_data_1_f0=4.9e4.csv',
20:     'aci_data_linked_eps_0.6.csv',
23: output_file_path = 'wav_sim_matches_aci_target_base.csv'

### BirdsongModel.cpp
3: #include "BirdsongModel.h"
4: #include <iostream>
34:     outfile.open("simulation_output_2(a).csv");

### BirdsongModel.h
4: #include <vector>
5: #include <fstream>
6: #include <string>

### BirdsongModel_10*10.cpp
3: #include "BirdsongModel_2.h"
4: #include <iostream>
5: #include <cmath>
6: #include <iomanip>

### BirdsongModel_10*10_2.cpp
3: #include "BirdsongModel_3.h"
4: #include <iostream>
5: #include <cmath>
6: #include <iomanip>

### BirdsongModel_10*10_s(t).cpp
3: #include "BirdsongModel_2.h"
4: #include <iostream>
5: #include <cmath>
6: #include <iomanip>

### BirdsongModel_2.h
4: #include <vector>
5: #include <fstream>
6: #include <string> 

### BirdsongModel_3.h
4: #include <vector>
5: #include <fstream>
6: #include <string> 

### BirdsongModel_colormap_1.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from scipy.signal import spectrogram, find_peaks
7: import glob
8: import os
9: import re # ファイル名から数値を抽出するため
10: import matplotlib.colors as mcolors
130: csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
143: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

### BirdsongModel_colormap_2.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from scipy.signal import spectrogram, find_peaks
7: import glob
8: import os
9: import re # ファイル名から数値を抽出するため
10: import matplotlib.colors as mcolors
130: csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
143: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

### BirdsongModel_colormap_ver1.1.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from scipy.signal import spectrogram, find_peaks
7: import glob
8: import os
9: import re # ファイル名から数値を抽出するため
10: import matplotlib.colors as mcolors
196: csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
209: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

### BirdsongModel_colormap_ver1.2.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from scipy.signal import spectrogram, find_peaks
7: import glob
8: import os
9: import re # ファイル名から数値を抽出するため
10: import matplotlib.colors as mcolors
202: csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
215: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

### BirdsongModel_colormap_ver2.0.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from scipy.signal import spectrogram, find_peaks
7: import glob
8: import os
9: import re # ファイル名から数値を抽出するため
10: import matplotlib.colors as mcolors
181: csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
194: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")

### BirdsongModel_distance1.cpp
1: #include "BirdsongModel_distance1.h"
3: #include <algorithm>
4: #include <cmath>
5: #include <iomanip>
6: #include <stdexcept>

### BirdsongModel_distance1.h
4: #include <fstream>
5: #include <string>
6: #include <vector>

### BirdsongModel_epsilon_change.cpp
1: #include "BirdsongModel_epsilon_change.h"
3: #include <algorithm>
4: #include <cmath>
5: #include <iomanip>
6: #include <stdexcept>

### BirdsongModel_epsilon_change.h
4: #include <fstream>
5: #include <string>
6: #include <vector>

### BirdsongModel_linear_epsilon.cpp
1: #include "BirdsongModel_linear_epsilon.h"
3: #include <cmath>
4: #include <iomanip>
5: #include <stdexcept>

### BirdsongModel_linear_epsilon.h
4: #include <fstream>
5: #include <string>
6: #include <vector>

### BirdsongModel_linear_epsilon_ps.cpp
1: #include "BirdsongModel_linear_epsilon_ps.h"
3: #include <cmath>
4: #include <iomanip>
5: #include <stdexcept>

### BirdsongModel_linear_epsilon_ps.h
4: #include <fstream>
5: #include <string>
6: #include <vector>

### BirdsongModel_linear_ps.cpp
1: #include "BirdsongModel_linear_ps.h"
3: #include <cmath>
4: #include <iomanip>
5: #include <stdexcept>

### BirdsongModel_linear_ps.h
4: #include <fstream>
5: #include <string>
6: #include <vector>

### BirdsongModel_rk_fig5a.cpp
3: #include "BirdsongModel.h"
4: #include <iostream>
40:     outfile.open("simulation-rk4_output_1(a).csv");

### BirdsongModel_rk_fig5b.cpp
3: #include "BirdsongModel.h"
4: #include <iostream>
40:     outfile.open("simulation-rk4_output_1(b).csv");

### BirdsongModel_rk_fig6.cpp
3: #include "BirdsongModel.h"
4: #include <iostream>
40:     outfile.open("simulation-rk4_output_2(c).csv");

### BirdsongModel_rk_normal.cpp
3: #include "BirdsongModel.h"
4: #include <iostream>
40:     outfile.open("simulation-rk4_output_normal.csv");

### BirdsongModel_rk_test1.cpp
3: #include "BirdsongModel.h"
4: #include <iostream>
31:     outfile.open("simulation_output_test1.csv");

### Birdsong_ACI.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from mpl_toolkits.mplot3d import Axes3D
10: import matplotlib.colors as mcolors
11: from scipy.signal import spectrogram
12: import glob
13: import os
14: import re
31:         "output_csv": "aci_data_1_f0=0.1e7.csv",
40:         "output_csv": "aci_data_1_f0=0.4e7.csv",
49:         "output_csv": "aci_data_1_f0=0.7e7.csv",
58:         "output_csv": "aci_data_1_f0=0.05e7.csv",
67:         "output_csv": "aci_data_1_f0=4.9e4.csv",
76:         "output_csv": "aci_data_1.csv",
85:         "output_csv": "aci_data_2.csv",
185: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")
207:         csv_files = glob.glob(os.path.join(folder, "*.csv"))

### Birdsong_ACI_linked_eps.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from mpl_toolkits.mplot3d import Axes3D
10: import matplotlib.colors as mcolors
11: from scipy.signal import spectrogram
12: import glob
13: import os
14: import re
31:         "output_csv": "aci_data_linked_eps_0.6.csv",
138: pattern = re.compile(r"sim_output_epsL_([0-9\.e\+\-]+)_epsR_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")
158:         csv_files = glob.glob(os.path.join(folder, "*.csv"))

### Birdsong_HNR_index.py
1: import glob
2: import os
3: import re
5: import numpy as np
6: import pandas as pd
7: from scipy.signal import correlate, spectrogram, welch
8: from scipy.stats import entropy
21:      #   "output_csv": "hnr_sci_aci_se_1_f0=1.0e7_3.4.csv",
27:         "output_csv": "hnr_sci_aci_se_2_update.csv",
68:     r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv"
463:             csv_files = glob.glob(os.path.join(folder, "*.csv"))

### Birdsong_SCI_Jitter_ver2.py
1: import pandas as pd
2: import numpy as np
3: from scipy.signal import find_peaks, welch, butter, filtfilt, correlate
4: import glob
5: import os
6: import re
20:         "output_csv": "sci_data_1_f0=0.1e7.csv",
27:         "output_csv": "sci_data_1_f0=1.0e7.csv",
34:         "output_csv": "sci_data_2.csv",
41:         "output_csv": "sci_data_1_f0=0.4e7.csv",
48:         "output_csv": "sci_data_1_f0=0.7e7.csv",
55:         "output_csv": "sci_data_1_f0=0.05e7.csv",
62:         "output_csv": "sci_data_1_f0=4.9e4.csv",
191: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")
210:         csv_files = glob.glob(os.path.join(folder, "*.csv"))

### Birdsong_Time-Averaged Power Spectrum.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from scipy.signal import spectrogram
10: import glob
11: import os
93: csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
101:         file_name = os.path.basename(csv_file).replace('.csv', '.png')

### Birdsong_complex.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from mpl_toolkits.mplot3d import Axes3D
10: import matplotlib.colors as mcolors
11: from scipy.signal import spectrogram
12: from scipy.stats import entropy
13: import glob
14: import os
15: import re
22: OUTPUT_CSV_FILE = "complexity_data_linked_eps*0.5.csv"       # 数値データの保存名
96: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")
100:     csv_files = glob.glob(os.path.join(folder, "*.csv"))

### Birdsong_complex_linked eps.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from mpl_toolkits.mplot3d import Axes3D
10: import matplotlib.colors as mcolors
11: from scipy.signal import spectrogram
12: from scipy.stats import entropy
13: import glob
14: import os
15: import re
22: OUTPUT_CSV_FILE = "complexity_data_linked_eps*0.6.csv"       # 数値データの保存名
96: pattern = re.compile(r"sim_output_epsL_([0-9\.e\+\-]+)_epsR_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")
100:     csv_files = glob.glob(os.path.join(folder, "*.csv"))

### Birdsong_complex_linked eps_ver2.0.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from mpl_toolkits.mplot3d import Axes3D
10: import matplotlib.colors as mcolors
11: from scipy.signal import spectrogram
12: from scipy.stats import entropy
13: import glob
14: import os
15: import re
28:         "output_csv": "complexity_data_linked_eps_0.6.csv",
125: pattern = re.compile(r"sim_output_epsL_([0-9\.e\+\-]+)_epsR_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")
145:         csv_files = glob.glob(os.path.join(folder, "*.csv"))

### Birdsong_complex_ver2.0.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from mpl_toolkits.mplot3d import Axes3D
10: import matplotlib.colors as mcolors
11: from scipy.signal import spectrogram
12: from scipy.stats import entropy
13: import glob
14: import os
15: import re
28:         "output_csv": "complexity_data_1_f0=0.1e7.csv",
37:         "output_csv": "complexity_data_1_f0=0.4e7.csv",
46:         "output_csv": "complexity_data_1_f0=0.7e7.csv",
55:         "output_csv": "complexity_data_1_f0=0.05e7.csv",
64:         "output_csv": "complexity_data_1_f0=4.9e4.csv",
73:         "output_csv": "complexity_data_1.csv",
82:         "output_csv": "complexity_data_2.csv",
186: pattern = re.compile(r"sim_output_eps_([0-9\.e\+\-]+)_ps_([0-9\.e\+\-]+)\.csv")
208:         csv_files = glob.glob(os.path.join(folder, "*.csv"))

### Birdsong_psd.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from scipy.signal import spectrogram
10: import glob
11: import os
95: csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
102:         file_name = os.path.basename(csv_file).replace('.csv', '.png')

### Birdsong_psd_normalized.py
3: import matplotlib
6: import pandas as pd
7: import numpy as np
8: import matplotlib.pyplot as plt
9: from scipy.signal import spectrogram
10: import glob
11: import os
99: csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
106:         # 出力ファイル名を作成 (.csv -> .png)
107:         file_name = os.path.basename(csv_file).replace('.csv', '.png')

### Color map/BirdsongModel_colormap_Manual.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import matplotlib.colors as mcolors
7: import os
10: MANUAL_DATA_FILE = "parameter_map_1_x0=0.02 add.csv"

### Color map/BirdsongModel_colormap_Manual_ver2.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import matplotlib.colors as mcolors
7: import os
10: MANUAL_DATA_FILE = "parameter_map_2_x0=0.02 add.csv"

### Color map/BirdsongModel_colormap_Manual_ver3.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import matplotlib.colors as mcolors
7: import os
10: MANUAL_DATA_FILE = "parameter_map_1_x0=0.02 add.csv"

### Complex_2/complexity_difference.py
1: import pandas as pd
2: import matplotlib.pyplot as plt
3: import numpy as np
7: df1 = pd.read_csv('complexity_data_linked_eps_0.6.csv')
8: df2 = pd.read_csv('complexity_data_1_f0=4.9e4.csv')

### audio.py
3: import pandas as pd
4: import numpy as np
5: from scipy.io.wavfile import write
9: # 例: 'simulation_output_timed_chaos.csv'
10: csv_filename = 'sim_output_eps_2.0e+07_ps_6.0e+06.csv'
13: output_wav_filename = 'sim_output_eps_2.0e+07_ps_6.0e+06.wav'
57:  #   スクリプトが完了すると`birdsong_output.wav`というファイルが作成されます。WSL環境からこのファイルを再生するには、以下のコマンドを実行するのが一番簡単です。

### calculate_distance1_aci_se.py
1: from pathlib import Path
3: import numpy as np
4: import pandas as pd
5: from scipy.signal import spectrogram
6: from scipy.stats import entropy
9: CSV_FILE = Path("simulation_distance1.csv")
10: OUTPUT_FILE = Path("simulation_distance1_aci_se.csv")

### complexity_similar.py
1: import pandas as pd
2: import numpy as np
7: wav_file_path = 'wav_complexity_results_12k_limited.csv' 
8: sim_file_path = 'complexity_data_2.csv'
9: output_file_path = 'wav_sim_matches_top5_2.csv'

### complexity_similar_%.py
1: import pandas as pd
2: import numpy as np
8: wav_file_path = 'wav_complexity_results_12k_limited.csv' 
10: sim_file_path = 'complexity_data_1_f0=0.7e7.csv'
12: output_file_path = 'wav_sim_matches_1_fo=0.7e7_95%.csv'

### complexity_similar_all.py
1: import pandas as pd
2: import numpy as np
3: import os
9: wav_file_path = 'wav_complexity_results_12k_limited.csv' 
14:     'complexity_data_1.csv',
15:      'complexity_data_2.csv',  # 追加したいファイルがあればコメントを外して記述
16:      'complexity_data_1_f0=0.4e7.csv',
17:      'complexity_data_1_f0=0.7e7.csv',
18:      'complexity_data_1_f0=0.1e7.csv',
19:      'complexity_data_1_f0=0.05e7.csv',
20:      'complexity_data_1_f0=4.9e4.csv',
21:      'complexity_data_linked_eps_0.6.csv',
25: output_file_path = 'wav_sim_matches_multi_top5.csv'

### complexity_similar_all_%.py
1: import pandas as pd
2: import numpy as np
3: import os
8: wav_file_path = 'wav_complexity_results_12k_limited.csv' 
11:     'complexity_data_1.csv',
12:      'complexity_data_2.csv',  # 追加したいファイルがあればコメントを外して記述
13:      'complexity_data_1_f0=0.4e7.csv',
14:      'complexity_data_1_f0=0.7e7.csv',
15:      'complexity_data_1_f0=0.1e7.csv',
16:      'complexity_data_1_f0=0.05e7.csv',
17:      'complexity_data_1_f0=4.9e4.csv',
18:      'complexity_data_linked_eps_0.6.csv',
22: output_file_path = 'wav_sim_matches_95%.csv'

### generate_distance1_audio.py
1: from math import gcd
2: from pathlib import Path
4: import numpy as np
5: import pandas as pd
6: from scipy.io import wavfile
7: from scipy.signal import resample_poly
10: CSV_FILE = Path("simulation_distance1.csv")
11: OUTPUT_WAV = Path("simulation_distance1.wav")

### generate_distance1_single_source_outputs.py
1: from pathlib import Path
3: import calculate_distance1_aci_se as indices
4: import generate_distance1_audio as audio
5: import kashika_distance1 as visualization
6: import plot_distance1_psd as psd
9: CSV_FILE = Path("simulation_distance1_single_source.csv")
10: OUTPUT_DIRECTORY = Path("distance1_figures")
20:     audio.OUTPUT_WAV = Path(f"{STEM}.wav")
24:     indices.OUTPUT_FILE = Path(f"{STEM}_aci_se.csv")

### hnr_sci_2d.py
1: import os
3: import matplotlib.colors as colors
4: import matplotlib.pyplot as plt
5: import numpy as np
6: import pandas as pd
9:     import scienceplots  # noqa: F401
24:         "hnr_sci_aci_se_1_f0=0.05e7.csv",
29:         "hnr_sci_aci_se_1_f0=0.1e7.csv",
34:         "hnr_sci_aci_se_1_f0=0.4e7.csv",
39:         "hnr_sci_aci_se_1_f0=0.7e7.csv",
44:         "hnr_sci_aci_se_1_f0=1.0e7.csv",
49:         "hnr_sci_aci_se_1_f0=4.9e4.csv",
54:         "hnr_sci_aci_se_2.csv",

### kashika_10*10.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import scienceplots
7: from scipy.signal import spectrogram
8: import glob
9: import os
118:         #title_filename = os.path.basename(csv_filepath).replace('.csv', '')
151:     csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
162:         output_filename = os.path.basename(csv_filepath).replace('.csv', '.png')

### kashika_10*10_2.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import scienceplots  # 1. 追加
7: from scipy.signal import spectrogram
8: import glob
9: import os
109:         base_name = os.path.basename(csv_filepath).replace('.csv', '')
160:     csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
171:         output_filename = os.path.basename(csv_filepath).replace('.csv', '.png')

### kashika_distance1.py
5: from pathlib import Path
7: import matplotlib
11: import matplotlib.pyplot as plt
12: import numpy as np
13: import pandas as pd
14: import scienceplots  # noqa: F401  (science スタイルの登録に必要)
15: from scipy.signal import spectrogram
35: CSV_FILE = Path("simulation_distance1.csv")
36: OUTPUT_DIRECTORY = Path("distance1_figures")

### kashika_epsilon_change.py
4: from pathlib import Path
6: import matplotlib
10: import matplotlib.pyplot as plt
11: import numpy as np
12: import pandas as pd
13: import scienceplots
14: from scipy.signal import spectrogram
39: OUTPUT_DIRECTORY = Path("epsilon_change_figures")
42:     Path("simulation_epsilon_linear_decrease.csv"),
43:     Path("simulation_epsilon_linear_increase.csv"),
44:     Path("simulation_epsilon_quadratic_decrease.csv"),
45:     Path("simulation_epsilon_quadratic_increase.csv"),
46:     Path("simulation_epsilon_step_decrease.csv"),
47:     Path("simulation_epsilon_step_increase.csv"),
48:     Path("simulation_epsilon_sine_f3.333333Hz.csv"),
49:     Path("simulation_epsilon_sine_f6.666667Hz.csv"),
50:     Path("simulation_epsilon_sine_f10Hz.csv"),

### kashika_fig5.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from matplotlib.colors import ListedColormap
7: from scipy.signal import spectrogram
15: df = pd.read_csv('simulation-rk4_output_1(a).csv')

### kashika_fig5_2a.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import scienceplots  # 1. 追加
7: from scipy.signal import spectrogram
14: # 例: 'simulation_output_fig5a_final.csv'
15: csv_filename = 'simulation-rk4_output_1(a).csv'

### kashika_fig5_2b.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import scienceplots  # 1. 追加
7: from scipy.signal import spectrogram
14: # 例: 'simulation_output_fig5a_final.csv'
15: csv_filename = 'simulation-rk4_output_1(b).csv'

### kashika_fig6a.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import scienceplots  # 1. 追加
7: from scipy.signal import spectrogram
13: df = pd.read_csv('simulation-rk4_output_2(a).csv')

### kashika_fig6b.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import scienceplots  # 1. 追加
7: from scipy.signal import spectrogram
13: df = pd.read_csv('simulation-rk4_output_2(b).csv')

### kashika_fig6c.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import scienceplots  # 1. 追加
7: from scipy.signal import spectrogram
13: df = pd.read_csv('simulation-rk4_output_2(c).csv')

### kashika_linear_epsilon.py
4: from pathlib import Path
6: import matplotlib
10: import matplotlib.pyplot as plt
11: import numpy as np
12: import pandas as pd
13: import scienceplots
14: from scipy.signal import spectrogram
39: OUTPUT_DIRECTORY = Path("linear_epsilon_figures")
42:     Path("simulation_linear_epsilon_start2.7e7_delta0.0e7.csv"),
43:     Path("simulation_linear_epsilon_start3.2e7_delta0.0e7.csv"),
44:     Path("simulation_linear_epsilon_start3.7e7_delta0.0e7.csv"),
45:     Path("simulation_linear_epsilon_start3.2e7_delta-0.5e7.csv"),
46:     Path("simulation_linear_epsilon_start3.2e7_delta-0.25e7.csv"),
47:     Path("simulation_linear_epsilon_start3.2e7_delta+0.25e7.csv"),
48:     Path("simulation_linear_epsilon_start3.2e7_delta+0.5e7.csv"),
49:     Path("simulation_linear_epsilon_start2.7e7_delta+0.5e7.csv"),
50:     Path("simulation_linear_epsilon_start3.7e7_delta-0.5e7.csv"),

### kashika_linear_epsilon_ps.py
4: from pathlib import Path
6: import matplotlib
10: import matplotlib.pyplot as plt
11: import numpy as np
12: import pandas as pd
13: import scienceplots
14: from scipy.signal import spectrogram
41: RESULT_DIRECTORY = Path("linear_epsilon_ps_results")
43: CONDITION_FILE = RESULT_DIRECTORY / "simulation_linear_epsilon_ps_conditions.csv"
187:         ylim_by_filename[Path(row["output_filename"])] = group_limits[key]
224:         ylim_by_filename[Path(row["output_filename"])] = group_limits[key]
338:     csv_files = [Path(filename) for filename in condition_data["output_filename"]]

### kashika_linear_ps.py
4: from pathlib import Path
6: import matplotlib
10: import matplotlib.pyplot as plt
11: import numpy as np
12: import pandas as pd
13: import scienceplots
14: from scipy.signal import spectrogram
39: RESULT_DIRECTORY = Path("linear_ps_results")
41: CONDITION_FILE = RESULT_DIRECTORY / "simulation_linear_ps_conditions.csv"
171:         ylim_by_filename[Path(row["output_filename"])] = group_limits[key]
246:     csv_files = [Path(filename) for filename in condition_data["output_filename"]]

### kashika_opposite_linear_epsilon_ps.py
4: from pathlib import Path
6: import matplotlib
10: import matplotlib.pyplot as plt
11: import numpy as np
12: import pandas as pd
13: import scienceplots
14: from scipy.signal import spectrogram
41: RESULT_DIRECTORY = Path("opposite_linear_epsilon_ps_results")
43: CONDITION_FILE = RESULT_DIRECTORY / "simulation_opposite_linear_epsilon_ps_conditions.csv"
176:         ylim_by_filename[Path(row["output_filename"])] = group_limits[key]
299:     csv_files = [Path(filename) for filename in condition_data["output_filename"]]

### kashika_s(t).py
1: import pandas as pd
2: import numpy as np
3: import matplotlib.pyplot as plt
4: import scienceplots
5: import glob
6: import os
119:     csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
125:         output_filename = os.path.basename(csv_filepath).replace('.csv', '.png')

### kashika_single_source_linear_epsilon.py
1: from pathlib import Path
3: import kashika_subharmonic_linear_epsilon as base
6: base.RESULT_DIRECTORY = Path("single_source_linear_epsilon_results")
9:     base.RESULT_DIRECTORY / "simulation_single_source_linear_epsilon_conditions.csv"

### kashika_single_source_linear_ps.py
1: from pathlib import Path
3: import kashika_linear_ps as base
6: base.RESULT_DIRECTORY = Path("single_source_linear_ps_results")
9:     base.RESULT_DIRECTORY / "simulation_single_source_linear_ps_conditions.csv"

### kashika_subharmonic_linear_epsilon.py
4: from pathlib import Path
6: import matplotlib
10: import matplotlib.pyplot as plt
11: import numpy as np
12: import pandas as pd
13: import scienceplots
14: from scipy.signal import spectrogram
39: RESULT_DIRECTORY = Path("subharmonic_linear_epsilon_results")
41: CONDITION_FILE = RESULT_DIRECTORY / "simulation_subharmonic_linear_epsilon_conditions.csv"
175:         ylim_by_filename[Path(row["output_filename"])] = group_limits[key]
250:     csv_files = [Path(filename) for filename in condition_data["output_filename"]]

### kashika_test1.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from scipy.signal import spectrogram
10: # 例: 'simulation_output_fig5a_final.csv'
11: csv_filename = 'simulation_output_test1.csv'

### kashika_wav.py
3: import numpy as np
4: import matplotlib.pyplot as plt
5: from scipy.signal import spectrogram, resample
6: from scipy.io import wavfile
13: wav_filename = 'Wsst_1.wav' # 拡張子は実際のファイルに合わせてください(.wav等)

### kashika_waveform.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from scipy.signal import spectrogram
13: df = pd.read_csv('simulation_output_test1.csv')

### kashika_waveform10*10.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: import scienceplots
7: import glob
8: import os
133:     csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
144:         output_filename = os.path.basename(csv_filepath).replace('.csv', '.png')

### kashika_waveform_kasane.py
3: import pandas as pd
4: import numpy as np
5: import matplotlib.pyplot as plt
6: from scipy.signal import spectrogram
13: df = pd.read_csv('simulation_output_2(b).csv')

### kyori.py
1: import pandas as pd
2: import numpy as np
3: import os
10: wav_aci_csv = 'wav_aci_results_12k_limited.csv'
11: wav_ent_csv = 'wav_complexity_results_12k_limited.csv'
16:     'aci_data_1.csv',
17:     'aci_data_2.csv',
18:     'aci_data_1_f0=0.4e7.csv',
19:     'aci_data_1_f0=0.7e7.csv',
20:     'aci_data_1_f0=0.1e7.csv',
21:     'aci_data_1_f0=0.05e7.csv',
22:     'aci_data_1_f0=4.9e4.csv',
23:     'aci_data_linked_eps_0.6.csv',
27:     'complexity_data_1.csv',
28:     'complexity_data_2.csv',
29:     'complexity_data_1_f0=0.4e7.csv',
30:     'complexity_data_1_f0=0.7e7.csv',
31:     'complexity_data_1_f0=0.1e7.csv',
32:     'complexity_data_1_f0=0.05e7.csv',
33:     'complexity_data_1_f0=4.9e4.csv',
34:     'complexity_data_linked_eps_0.6.csv',
38: output_file = 'wav_sim_matches_2D_best_all_v2.csv'

### main.cpp
1: #include "BirdsongModel.h"
2: #include <iostream>
31:     cout << "\nSimulation finished. Data saved to simulation_output_test1.csv" << endl;

### main_10*10.cpp
1: #include "BirdsongModel_2.h"
2: #include <iostream>
3: #include <vector>
4: #include <string>
5: #include <sstream>
6: #include <iomanip>
7: #include <sys/stat.h>
8: #include <sys/types.h> 
9: #include <unistd.h>
21:     const string output_folder = "simulation_results_2_x0=0.02_update/";
59:                << "_ps_" << std::scientific << std::setprecision(2) << current_ps << ".csv";

### main_10*10_2.cpp
1: #include "BirdsongModel_3.h"
2: #include <iostream>
3: #include <vector>
4: #include <string>
5: #include <sstream>
6: #include <iomanip>
7: #include <sys/stat.h>
8: #include <sys/types.h> 
9: #include <unistd.h>
21:     const string output_folder = "simulation_results_x0=0.02_linked_eps*0.6/";
66:                << "_ps_" << std::scientific << std::setprecision(2) << current_ps << ".csv";

### main_distance1.cpp
1: #include "BirdsongModel_distance1.h"
3: #include <cmath>
4: #include <iostream>
21:     constexpr const char* output_filename = "simulation_distance1.csv";

### main_distance1_single_source.cpp
1: #include "BirdsongModel_distance1.h"
3: #include <cmath>
4: #include <iostream>
23:         "simulation_distance1_single_source.csv";

### main_epsilon_change.cpp
1: #include "BirdsongModel_epsilon_change.h"
3: #include <cmath>
4: #include <iostream>
5: #include <string>
6: #include <vector>
34:             "simulation_epsilon_linear_decrease.csv"
42:             "simulation_epsilon_linear_increase.csv"
50:             "simulation_epsilon_quadratic_decrease.csv"
58:             "simulation_epsilon_quadratic_increase.csv"
66:             "simulation_epsilon_step_decrease.csv"
74:             "simulation_epsilon_step_increase.csv"
82:             "simulation_epsilon_sine_f3.333333Hz.csv"
90:             "simulation_epsilon_sine_f6.666667Hz.csv"
98:             "simulation_epsilon_sine_f10Hz.csv"

### main_linear_epsilon.cpp
1: #include "BirdsongModel_linear_epsilon.h"
3: #include <cmath>
4: #include <fstream>
5: #include <iomanip>
6: #include <iostream>
7: #include <string>
8: #include <vector>
36:             "simulation_linear_epsilon_start2.7e7_delta0.0e7.csv"
42:             "simulation_linear_epsilon_start3.2e7_delta0.0e7.csv"
48:             "simulation_linear_epsilon_start3.7e7_delta0.0e7.csv"
57:             "simulation_linear_epsilon_start3.2e7_delta-0.5e7.csv"
63:             "simulation_linear_epsilon_start3.2e7_delta-0.25e7.csv"
69:             "simulation_linear_epsilon_start3.2e7_delta+0.25e7.csv"
75:             "simulation_linear_epsilon_start3.2e7_delta+0.5e7.csv"
83:             "simulation_linear_epsilon_start2.7e7_delta+0.5e7.csv"
89:             "simulation_linear_epsilon_start3.7e7_delta-0.5e7.csv"
93:     std::ofstream condition_file("simulation_linear_epsilon_conditions.csv");
95:         std::cerr << "Could not open simulation_linear_epsilon_conditions.csv\n";

### main_linear_epsilon_ps.cpp
1: #include "BirdsongModel_linear_epsilon_ps.h"
3: #include <cmath>
4: #include <filesystem>
5: #include <fstream>
6: #include <iomanip>
7: #include <iostream>
8: #include <sstream>
9: #include <string>
10: #include <vector>
64:     const std::filesystem::path result_directory = "linear_epsilon_ps_results";
65:     const std::filesystem::path csv_directory = result_directory / "csv";
136:                         + ".csv"
144:         result_directory / "simulation_linear_epsilon_ps_conditions.csv";

### main_linear_ps.cpp
1: #include "BirdsongModel_linear_ps.h"
3: #include <cmath>
4: #include <filesystem>
5: #include <fstream>
6: #include <iomanip>
7: #include <iostream>
8: #include <sstream>
9: #include <string>
10: #include <vector>
54:     const std::filesystem::path result_directory = "linear_ps_results";
55:     const std::filesystem::path csv_directory = result_directory / "csv";
104:                     + "_delta" + delta_label + ".csv"
111:         result_directory / "simulation_linear_ps_conditions.csv";

### main_opposite_linear_epsilon_ps.cpp
1: #include "BirdsongModel_linear_epsilon_ps.h"
3: #include <cmath>
4: #include <filesystem>
5: #include <fstream>
6: #include <iomanip>
7: #include <iostream>
8: #include <sstream>
9: #include <string>
10: #include <vector>
64:     const std::filesystem::path result_directory = "opposite_linear_epsilon_ps_results";
65:     const std::filesystem::path csv_directory = result_directory / "csv";
133:                         + ".csv"
141:         result_directory / "simulation_opposite_linear_epsilon_ps_conditions.csv";

### main_single_source_linear_epsilon.cpp
1: #include "BirdsongModel_linear_epsilon.h"
3: #include <cmath>
4: #include <filesystem>
5: #include <fstream>
6: #include <iomanip>
7: #include <iostream>
8: #include <sstream>
9: #include <string>
10: #include <vector>
54:     const std::filesystem::path result_directory =
56:     const std::filesystem::path csv_directory = result_directory / "csv";
103:                     + "_delta" + delta_label + ".csv"
110:         result_directory / "simulation_single_source_linear_epsilon_conditions.csv";

### main_single_source_linear_ps.cpp
1: #include "BirdsongModel_linear_ps.h"
3: #include <cmath>
4: #include <filesystem>
5: #include <fstream>
6: #include <iomanip>
7: #include <iostream>
8: #include <sstream>
9: #include <string>
10: #include <vector>
54:     const std::filesystem::path result_directory =
56:     const std::filesystem::path csv_directory = result_directory / "csv";
99:                     + "_delta" + delta_label + ".csv"
106:         result_directory / "simulation_single_source_linear_ps_conditions.csv";

### main_subharmonic_linear_epsilon.cpp
1: #include "BirdsongModel_linear_epsilon.h"
3: #include <cmath>
4: #include <filesystem>
5: #include <fstream>
6: #include <iomanip>
7: #include <iostream>
8: #include <sstream>
9: #include <string>
10: #include <vector>
54:     const std::filesystem::path result_directory =
56:     const std::filesystem::path csv_directory = result_directory / "csv";
109:                     + "_delta" + delta_label + ".csv"
116:         result_directory / "simulation_subharmonic_linear_epsilon_conditions.csv";

### plot_distance1_psd.py
1: from pathlib import Path
3: import matplotlib
7: import matplotlib.pyplot as plt
8: import numpy as np
9: import pandas as pd
10: import scienceplots  # noqa: F401
11: from scipy.signal import spectrogram
14: CSV_FILE = Path("simulation_distance1.csv")
15: OUTPUT_FILE = Path("distance1_figures/simulation_distance1_psd.png")
16: NORMALIZED_OUTPUT_FILE = Path(

### s.py
1: import pandas as pd
2: import numpy as np
3: import matplotlib.pyplot as plt
4: import os
10: INPUT_CSV = "simulation_output_test1.csv"

### sci_2d.py
1: import pandas as pd
2: import matplotlib.pyplot as plt
3: import scienceplots
4: import numpy as np
5: import matplotlib.colors as colors
15:     #('sci_data_1_f0=1.0e7.csv', 'sci_2d_1.png', 'SCI Map (f0=1.0e7)'),
17:     #('sci_data_2.csv', 'sci_2d_2.png', 'SCI Map 2'),
18:     ('sci_data_1_f0=0.4e7.csv', 'sci_2d_1_f0=0.4e7.png', 'SCI MAP(f0=0.4e7)'),
19:     ('sci_data_1_f0=0.7e7.csv', 'sci_2d_1_f0=0.7e7.png', 'SCI MAP(f0=0.7e7)'),
20:     ('sci_data_1_f0=0.05e7.csv', 'sci_2d_1_f0=0.05e7.png', 'SCI MAP(f0=0.05e7)'),
21:     ('sci_data_1_f0=4.9e4.csv', 'sci_2d_1_f0=4.9e4.png', 'SCI MAP(f0=4.9e4)'),