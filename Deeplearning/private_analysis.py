"""Assess saved real-target runs locally; no original recording is read again."""
import json
import os

import numpy as np
from scipy.io import wavfile

from .evaluation import descriptors
from .experiments import write_json
from .model import ROOT, BirdModel, effective_controls
from .plotting import learning_curves, control_trajectories


def main():
    output = ROOT/"private_runs/phase2"
    summary = json.loads((output/"summary.json").read_text())
    config = json.loads((output/"config.json").read_text())
    supplementary, curves = [], {key: [] for key in "ABC"}
    for item in summary:
        condition, seed = item["condition"], item["seed"]
        directory = output/f"{condition}_{seed}"
        entries = json.loads((directory/"curve.json").read_text())
        curves[condition].append(([e["steps"] for e in entries], [e["error"] for e in entries]))
        with np.load(directory/"rollout.npz") as data:
            rate = int(data["sample_rate"])
            audio, target = data["audio"], data["target"]
            controls = np.array([effective_controls(c, condition) for c in data["physical_controls"]])
            moments = []
            with BirdModel(condition) as model:
                for c in controls:
                    model.set_controls(c)
                    _, rms = model.advance(.01, return_rms=True)
                    moments.append(rms)
            moments = np.array(moments)
            # Final 0.25 s excludes the initial transient. Velocity RMS, not offset x.
            rms = np.sqrt(np.mean(moments[-25:]**2, axis=0))
            info = dict(condition=condition, seed=seed,
                        target_descriptors=descriptors(target, rate),
                        generated_descriptors=descriptors(audio, rate),
                        steady_state_rms=dict(zip(["x_l", "y_l", "x_r", "y_r"], rms.tolist())),
                        right_left_velocity_ratio=float(rms[3]/max(rms[1], 1e-12)))
            supplementary.append(info)
            wavfile.write(directory/"generated.wav", rate,
                          (audio/max(1, float(np.max(abs(audio))))).astype(np.float32))
            write_json(directory/"supplementary.json", info)
            control_trajectories(controls, directory/"controls.png")
    write_json(output/"supplementary.json", supplementary)
    learning_curves(curves, output/"learning_curves.png")
    lines = ["# 実録音0.35秒のローカル実験レポート", "", "2026-10-30 JST。研究グループ外に共有しない成果物。", "",
             "目標はユーザー指定の0.35秒版のみ。1秒版は実験に使用していない。原音の変更・アップロードはしていない。", "",
             "## 条件", "", "SAC、seed 0/1/2、各1050訓練ステップ、評価は350ステップごと。全条件で同じ目標・報酬・予算。",
             "A: 右alpha=beta=0（左epsilon、ps、左f0の3自由度）。B: 右f0=1e7を固定（左右epsilon、ps、左f0の4自由度）。C: 左右epsilon、ps、左右f0の5自由度。", "",
             "外力f0の範囲は[4.9e4,1e7]。psは[2.5e6,4e6]、epsilonは[2.5e7,5e7]。区間内一定・左右同時更新。初期状態と遅延履歴はゼロ。左右差は許すが交互発声を強制しない。", "",
             "モデルA基準のRMSを使って、全A/B/Cに共通のゲインを一度だけ校正した。WAV振幅を物理圧力へ校正したものではない。", "",
             "## 前処理", "", "```json", json.dumps(config["preprocessing"], ensure_ascii=False, indent=2), "```", "",
             "先頭から正確に0.35秒を選択。余分な末尾のみを除外し、時間伸縮はしていない。50 kHzへアンチエイリアス付きでリサンプリングした。", "",
             "## 評価結果", "", "誤差は小さいほどよい。初期値は未学習方策、最終値は1050ステップ後の探索なし評価。", "",
             "| 条件 | seed | 初期誤差 | 最終誤差 | 閾値0.08到達step | 訓練＋途中評価秒 |", "|---|---:|---:|---:|---:|---:|"]
    for x in summary:
        lines.append(f"| {x['condition']} | {x['seed']} | {x['initial_error']:.5f} | {x['error']:.5f} | {x['first_threshold_step'] if x['first_threshold_step'] is not None else '未到達'} | {x['training_and_eval_seconds']:.2f} |")
    lines += ["", "| 条件 | 平均±seed標準偏差 | 最小–最大 |", "|---|---:|---:|"]
    for condition in "ABC":
        values = np.array([x["error"] for x in summary if x["condition"] == condition])
        lines.append(f"| {condition} | {values.mean():.5f} ± {values.std(ddof=1):.5f} | {values.min():.5f}–{values.max():.5f} |")
    lines += ["", "![学習曲線](learning_curves.png)", "", "## 振動と特徴量", "",
              "SCIは既存コードのf_MSF/f_AFFに対応する参考実装でSAPの認証値ではない。SE・ACI・自己相関HNRは記述量であり、複雑さの単一尺度にしない。低い周期性の音も損失計算から除外していない。詳細はsupplementary.json。", "",
              "右の抑制は制御周期の端点だけでは判定できないため、保存済みの制御を再生し、全積分ステップの速度二乗平均を測定した。初期0.1秒を除外した最後0.25秒の値。Aでは右が動いても寄与が除外されている。", "",
              "| 条件 | seed | 左速度RMS | 右速度RMS | 右/左 |", "|---|---:|---:|---:|---:|"]
    for x in supplementary:
        r = x["steady_state_rms"]
        lines.append(f"| {x['condition']} | {x['seed']} | {r['y_l']:.5g} | {r['y_r']:.5g} | {x['right_left_velocity_ratio']:.5g} |")
    lines += ["", "## 解釈と残作業", "",
              "3 seedの小規模実験であり、誤差の大小だけから音源条件の普遍的優位性を結論しない。自由度、声道フィルタ、位相、振幅校正、初期過渡、部分観測の影響が残る。", "",
              "Phase 1でもSACの安定した模倣は確立していないため、今回の実録音の成績からモデル表現力の限界は判定できない。実録音に対する直接最適化、制御範囲の調査、より長い学習、別評価区間や指標の感度確認が必要。鳥の神経学的な学習機構を説明する結果ではない。", "",
              "各A_0等のフォルダにスペクトログラム、生成音WAV、制御軌道、目標と生成音のNPZ、モデル、学習曲線JSONを保存した。generated.wavの聴取用ゲインは評価値に使用していない。", "",
              "数値発散の最終評価失敗はsummary.jsonのfailed欄で確認できる。訓練途中の発散回数を集計するログはこの実験では未実装のため、最終評価失敗率と同一視しない。", ""]
    (output/"REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    previous = os.umask(0o077)
    try:
        main()
    finally:
        os.umask(previous)
