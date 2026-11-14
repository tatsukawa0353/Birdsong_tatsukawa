# Issue #15: lambda=0.5 weighted-reward follow-up

Training reward per 10 ms step: r = -(E_spec + 0.5 E_env).
All four conditions share the same final 0.35 s combined evaluation E_spec+E_env.
The independent baseline A/B/C models are reused, not retrained or modified.
This evaluates in-target tracking, not bird-song generalization.

## Final policy, 3 seeds (mean +/- sample SD)

| Condition | Algorithm | Combined acoustic error | Frequency MAE [Hz] | Amplitude MAE |
| --- | --- | ---: | ---: | ---: |
| combined | PPO | 0.030880 +/- 0.006337 | 95.655 +/- 45.024 | 0.007045 +/- 0.003484 |
| combined | TD3 | 0.020582 +/- 0.001583 | 30.511 +/- 3.710 | 0.001885 +/- 0.000500 |
| combined | SAC | 0.029000 +/- 0.004149 | 68.668 +/- 25.825 | 0.001744 +/- 0.001159 |
| spectral_only | PPO | 0.061926 +/- 0.006444 | 83.118 +/- 10.174 | 0.045653 +/- 0.006534 |
| spectral_only | TD3 | 0.056940 +/- 0.005956 | 22.030 +/- 18.069 | 0.033296 +/- 0.004778 |
| spectral_only | SAC | 0.070406 +/- 0.010367 | 16.956 +/- 7.346 | 0.046826 +/- 0.007473 |
| envelope_only | PPO | 0.064683 +/- 0.019779 | 481.630 +/- 318.414 | 0.005762 +/- 0.000794 |
| envelope_only | TD3 | 0.084909 +/- 0.011404 | 703.458 +/- 147.545 | 0.001849 +/- 0.000816 |
| envelope_only | SAC | 0.058296 +/- 0.000578 | 367.755 +/- 1.455 | 0.000710 +/- 0.000246 |
| weighted_lambda_0p5 | PPO | 0.028461 +/- 0.010261 | 65.092 +/- 65.988 | 0.010906 +/- 0.004964 |
| weighted_lambda_0p5 | TD3 | 0.017682 +/- 0.002967 | 12.067 +/- 7.131 | 0.002000 +/- 0.000429 |
| weighted_lambda_0p5 | SAC | 0.025942 +/- 0.004365 | 46.243 +/- 10.477 | 0.003195 +/- 0.001053 |

## Saved figures

- [Seed-0 learning curves](figures/learning_curves_seed0.png)
- [3-seed means and SD](figures/learning_curves_mean_sd.png)
- PPO: [spectrogram](figures/PPO_seed0_spectrogram.png), [frequency and amplitude tracking](figures/PPO_seed0_controls.png)
- TD3: [spectrogram](figures/TD3_seed0_spectrogram.png), [frequency and amplitude tracking](figures/TD3_seed0_controls.png)
- SAC: [spectrogram](figures/SAC_seed0_spectrogram.png), [frequency and amplitude tracking](figures/SAC_seed0_controls.png)

## Cost and interpretation

Nine new weighted runs: 533.465 s elapsed wall time with concurrent workers.
Report/plot creation: 6.828 s.
Check E_spec and E_env separately, along with f/A MAE. The weighted coefficient is not the same as a direct frequency penalty.
Do not interpret a single lambda as the optimum; all runs share the same fixed target.
The original Issue-15 report and #14 results remain unchanged.
