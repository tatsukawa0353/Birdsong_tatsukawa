# Stage B — 広域通常SACを先行

[
  {
    "condition": "A",
    "seed": 0,
    "status": "complete",
    "final_E": 0.2635241261689831,
    "final_J": 0.1853700342086613,
    "training": {
      "episodes": 4096,
      "decision_steps": 4096,
      "native_blocks_attempted": 81920,
      "native_blocks_completed": 81920,
      "native_steps_attempted": 8192000000,
      "native_steps_completed": 8192000000,
      "integrated_s_attempted": 819.2,
      "integrated_s_completed": 819.2,
      "generated_audio_s": 819.2,
      "failures": 0,
      "silent_episodes": 0,
      "nonvoiced_episodes": 169,
      "wall_s": 893.5951123544946
    },
    "evaluations": 33
  },
  {
    "condition": "A",
    "seed": 1,
    "status": "complete",
    "final_E": 0.0959899773280352,
    "final_J": 0.04452709578495854,
    "training": {
      "episodes": 4096,
      "decision_steps": 4096,
      "native_blocks_attempted": 81920,
      "native_blocks_completed": 81920,
      "native_steps_attempted": 8192000000,
      "native_steps_completed": 8192000000,
      "integrated_s_attempted": 819.2,
      "integrated_s_completed": 819.2,
      "generated_audio_s": 819.2,
      "failures": 0,
      "silent_episodes": 0,
      "nonvoiced_episodes": 159,
      "wall_s": 799.8635551724583
    },
    "evaluations": 33
  },
  {
    "condition": "A",
    "seed": 2,
    "status": "complete",
    "final_E": 0.08283012680566956,
    "final_J": 0.03972429856739772,
    "training": {
      "episodes": 4096,
      "decision_steps": 4096,
      "native_blocks_attempted": 81920,
      "native_blocks_completed": 81920,
      "native_steps_attempted": 8192000000,
      "native_steps_completed": 8192000000,
      "integrated_s_attempted": 819.2,
      "integrated_s_completed": 819.2,
      "generated_audio_s": 819.2,
      "failures": 0,
      "silent_episodes": 0,
      "nonvoiced_episodes": 202,
      "wall_s": 803.0251225568354
    },
    "evaluations": 33
  },
  {
    "condition": "B",
    "seed": 0,
    "status": "complete",
    "final_E": 0.4489481934584324,
    "final_J": 0.21644967846959018,
    "training": {
      "episodes": 4096,
      "decision_steps": 4096,
      "native_blocks_attempted": 81920,
      "native_blocks_completed": 81920,
      "native_steps_attempted": 8192000000,
      "native_steps_completed": 8192000000,
      "integrated_s_attempted": 819.2,
      "integrated_s_completed": 819.2,
      "generated_audio_s": 819.2,
      "failures": 0,
      "silent_episodes": 0,
      "nonvoiced_episodes": 4095,
      "wall_s": 818.9697605771944
    },
    "evaluations": 33
  },
  {
    "condition": "B",
    "seed": 1,
    "status": "complete",
    "final_E": 0.4484543657776755,
    "final_J": 0.21631138714409043,
    "training": {
      "episodes": 4096,
      "decision_steps": 4096,
      "native_blocks_attempted": 81920,
      "native_blocks_completed": 81920,
      "native_steps_attempted": 8192000000,
      "native_steps_completed": 8192000000,
      "integrated_s_attempted": 819.2,
      "integrated_s_completed": 819.2,
      "generated_audio_s": 819.2,
      "failures": 0,
      "silent_episodes": 0,
      "nonvoiced_episodes": 4095,
      "wall_s": 818.850256184116
    },
    "evaluations": 33
  },
  {
    "condition": "B",
    "seed": 2,
    "status": "complete",
    "final_E": 0.44913489943161755,
    "final_J": 0.21649695666084373,
    "training": {
      "episodes": 4096,
      "decision_steps": 4096,
      "native_blocks_attempted": 81920,
      "native_blocks_completed": 81920,
      "native_steps_attempted": 8192000000,
      "native_steps_completed": 8192000000,
      "integrated_s_attempted": 819.2,
      "integrated_s_completed": 819.2,
      "generated_audio_s": 819.2,
      "failures": 0,
      "silent_episodes": 0,
      "nonvoiced_episodes": 4093,
      "wall_s": 819.852775067091
    },
    "evaluations": 33
  }
]

Bの結果によらずC/Dを同じ凍結計画で全て実施する。0.10の閾値による打切りや追加調整はしない。独立再生監査はStage Dでまとめる。
