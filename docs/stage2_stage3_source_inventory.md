# Stage-2/Stage-3 Source Inventory

Generated: 2026-07-12T12:26:59.429352+00:00

Source root: `/mnt/infini-data/test/BeyondMimic`

Release root: `/mnt/infini-data/test/BeyondMimic-Reproduction`

Full `find` source file count: **93072**

Keyword/priority candidate count: **3726**

Semantically migrated/copied count: **2975**

Legacy or superseded count: **81**

Stage counts: `{'Adapter/Controller': 2904, 'Stage-2': 88, 'Stage-3': 504, 'Related': 130, 'Stage-1/Shared source': 100}`

No checkpoint, ONNX, NPZ, video, wandb, cache, or credential files are copied into this release repo. H20 experiment scripts with local absolute paths are SHA-indexed and re-expressed through package interfaces.

## Candidate Preview

| Source | Stage | Purpose | Destination | Migrated | Runnable | Superseded | SHA256 |
|---|---|---|---|---:|---:|---:|---|
| `README.md` | Adapter/Controller | teacher asset/rollout, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `175d19724db6` |
| `goal.md` | Adapter/Controller | teacher asset/rollout, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `e477ba315c22` |
| `isaac_mp4_need/README_ISAAC_MP4_RTX.md` | Adapter/Controller | rollout, Isaac adapter, ONNX/controller export, motion/data | src/beyondmimic_repro/adapters/ | True | False | False | `e0b02e4f0e32` |
| `isaac_mp4_need/isaac_mp4_need_manifest.json` | Adapter/Controller | Isaac adapter | src/beyondmimic_repro/adapters/ | True | False | False | `5c9184325a12` |
| `isaac_mp4_need/restore_into_project.sh` | Adapter/Controller | Isaac adapter, motion/data | src/beyondmimic_repro/adapters/ | True | False | False | `6f7e44874c71` |
| `isaac_mp4_need/run_rtx_full_videos.sh` | Adapter/Controller | rollout, Isaac adapter | src/beyondmimic_repro/adapters/ | True | False | False | `2629f1a90356` |
| `isaac_mp4_need/run_rtx_smoke.sh` | Adapter/Controller | Isaac adapter | src/beyondmimic_repro/adapters/ | True | False | True | `633190a3e949` |
| `isaac_weights/README.md` | Adapter/Controller | MuJoCo adapter/probe, Isaac adapter | not uploaded | False | False | False | `226075c6a240` |
| `isaac_weights/dance1_subject1_model29999_official/params/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `762481d7377d` |
| `isaac_weights/dance1_subject1_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `12f761408622` |
| `isaac_weights/dance1_subject2_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `a5057014f172` |
| `isaac_weights/dance1_subject3_model29999_official/params/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `51d759e3cf31` |
| `isaac_weights/dance1_subject3_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `259e29d4474b` |
| `isaac_weights/dance2_subject1_model29999_official/params/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `1f7ec3942e72` |
| `isaac_weights/dance2_subject1_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `4fef2641d7e9` |
| `isaac_weights/dance2_subject2_model29999_official/params/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `9dbee528126c` |
| `isaac_weights/dance2_subject2_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `cebc856713b3` |
| `isaac_weights/fallAndGetUp1_subject1_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `16857b3c0787` |
| `isaac_weights/fightAndSports1_subject1_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `ee19f949a7cd` |
| `isaac_weights/jumps1_subject1_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `f8651045097b` |
| `isaac_weights/run2_subject1_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `7ca1a32ccec0` |
| `isaac_weights/sprint1_subject2_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `bebf0c0d6719` |
| `isaac_weights/walk1_model20000_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `5ab17670c93e` |
| `isaac_weights/walk1_model30000_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `5ab17670c93e` |
| `isaac_weights/walk2_subject3_model29999_official/params/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `ca62112e1574` |
| `isaac_weights/walk2_subject3_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `cd28087fcf23` |
| `isaac_weights/walk2_subject4_model29999_official/params/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `831716f61b38` |
| `isaac_weights/walk2_subject4_model29999_official/params/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `3f3fe6b61c5d` |
| `mujoco_mp4/MIGRATION_AND_RESULTS_SUMMARY.md` | Adapter/Controller | MuJoCo adapter/probe, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `96cf8f1045b4` |
| `mujoco_mp4/README_MUJOCO_MP4_H20.md` | Adapter/Controller | rollout, MuJoCo adapter/probe, Isaac adapter, ONNX/controller export | docs/stage2_stage3_source_inventory.md | False | False | False | `920c6f6dc658` |
| `mujoco_mp4/configs/g1_joint_mapping.yaml` | Adapter/Controller | MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `8300c2eafacd` |
| `mujoco_mp4/mujoco_mp4_manifest.json` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | False | `416b6cafdf78` |
| `mujoco_mp4/run_mujoco_control_videos.sh` | Stage-2 | VAE, latent, MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | False | `2ff95b4bdbd0` |
| `mujoco_mp4/run_mujoco_reference_replay.sh` | Adapter/Controller | MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `87206b77ad6d` |
| `mujoco_mp4/run_mujoco_smoke.sh` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | True | `5a2f7e0b9ce0` |
| `mujoco_mp4/scripts/mujoco_closed_loop_latent_receding_probe.py` | Stage-3 | VAE, latent, diffusion, MuJoCo adapter/probe, Isaac adapter, controller, motion/data | docs/stage2_stage3_source_inventory.md | False | True | False | `b29d4daeb127` |
| `mujoco_mp4/scripts/mujoco_common.py` | Adapter/Controller | MuJoCo adapter/probe | src/beyondmimic_repro/adapters/mujoco/shared_controller_contract.py | True | False | False | `5a069b464f4c` |
| `mujoco_mp4/scripts/mujoco_control_video_summary.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | True | False | `9713408baf79` |
| `mujoco_mp4/scripts/mujoco_g1_asset_inventory.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | True | False | `099022decfa1` |
| `mujoco_mp4/scripts/mujoco_g1_import_smoke.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | True | True | `e3827b4d1d2f` |
| `mujoco_mp4/scripts/mujoco_lafan1_7motion_velocity_guided_diffusion_vae.py` | Stage-3 | VAE, latent, diffusion, guidance, MuJoCo adapter/probe, ONNX/controller export, controller, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `b668d46e22e0` |
| `mujoco_mp4/scripts/mujoco_minimal_video_smoke.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | True | True | `32fd8136f99d` |
| `mujoco_mp4/scripts/mujoco_mp4_manifest.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | True | False | `8aae0978cc04` |
| `mujoco_mp4/scripts/mujoco_pd_control_video.py` | Adapter/Controller | rollout, MuJoCo adapter/probe, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | True | False | `7e8ebc487f5c` |
| `mujoco_mp4/scripts/mujoco_policy_switch_walk_run_walk.py` | Adapter/Controller | MuJoCo adapter/probe, ONNX/controller export, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `f319d9f8cf25` |
| `mujoco_mp4/scripts/mujoco_ppo_adapter_gap_audit.py` | Adapter/Controller | MuJoCo adapter/probe, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | True | False | `dbfd1ea24d0e` |
| `mujoco_mp4/scripts/mujoco_reference_replay_video.py` | Adapter/Controller | rollout, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | True | False | `f0adb59e7871` |
| `mujoco_mp4/scripts/mujoco_render_backend_probe.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | True | False | `528a29dc0bfc` |
| `mujoco_mp4/scripts/mujoco_senlanke_walk1_trained_policy_sim2sim_video.py` | Adapter/Controller | MuJoCo adapter/probe, ONNX/controller export, motion/data | src/beyondmimic_repro/adapters/mujoco/shared_controller_contract.py | True | False | False | `13062bd2b10d` |
| `mujoco_mp4/scripts/mujoco_trace_mesh_video.py` | Adapter/Controller | rollout, MuJoCo adapter/probe, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | True | False | `3d02af77455c` |
| `mujoco_mp4/scripts/mujoco_walk1_official_policy_diffusion_guided_two_waypoints.py` | Stage-3 | rollout, state-latent, latent, diffusion, guidance, waypoint guidance, obstacle guidance, MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | False | `c84d49d8e98c` |
| `mujoco_mp4/scripts/mujoco_walk1_official_policy_diffusion_guided_waypoint_obstacle.py` | Stage-3 | latent, diffusion, guidance, waypoint guidance, obstacle guidance, MuJoCo adapter/probe, controller, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `b9b84c64e495` |
| `mujoco_mp4/scripts/mujoco_walk1_official_policy_guided_waypoint_obstacle.py` | Stage-3 | rollout, guidance, waypoint guidance, obstacle guidance, MuJoCo adapter/probe, ONNX/controller export, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `ae193b202838` |
| `mujoco_mp4/scripts/mujoco_walk1_official_policy_velocity_transition.py` | Stage-3 | waypoint guidance, obstacle guidance, MuJoCo adapter/probe, ONNX/controller export, controller, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `e32802ac1091` |
| `official_mp4/README_OFFICIAL_MP4.md` | Stage-3 | diffusion, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `4c851c6dc09e` |
| `official_mp4/official_mp4_manifest.json` | Adapter/Controller | MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `4e6313146475` |
| `official_mp4/run_official_mp4.sh` | Adapter/Controller | MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `688b30549be5` |
| `official_mp4/scripts/official_dataset_inventory.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | False | `090c59ae7a29` |
| `official_mp4/scripts/official_mp4_manifest.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | False | `69836443dce0` |
| `official_mp4/scripts/render_official_g1_csv_replay.py` | Adapter/Controller | MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `e3061dae736c` |
| `official_mp4/scripts/render_official_mcap_joint_replay.py` | Adapter/Controller | MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | False | `6a7f24fef557` |
| `policy_exports/README.md` | Adapter/Controller | Isaac adapter, ONNX/controller export | not uploaded | False | False | False | `d19ec5b4668d` |
| `policy_exports/dance1_subject1_model29999_official/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `762481d7377d` |
| `policy_exports/dance1_subject1_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `12f761408622` |
| `policy_exports/dance1_subject2_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `a5057014f172` |
| `policy_exports/dance1_subject3_model29999_official/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `51d759e3cf31` |
| `policy_exports/dance1_subject3_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `259e29d4474b` |
| `policy_exports/dance2_subject1_model29999_official/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `1f7ec3942e72` |
| `policy_exports/dance2_subject1_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `4fef2641d7e9` |
| `policy_exports/dance2_subject2_model29999_official/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `9dbee528126c` |
| `policy_exports/dance2_subject2_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `cebc856713b3` |
| `policy_exports/fallAndGetUp1_subject1_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `16857b3c0787` |
| `policy_exports/fightAndSports1_subject1_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `ee19f949a7cd` |
| `policy_exports/jumps1_subject1_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `f8651045097b` |
| `policy_exports/run2_subject1_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `7ca1a32ccec0` |
| `policy_exports/sprint1_subject2_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `bebf0c0d6719` |
| `policy_exports/walk1_model30000_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `5ab17670c93e` |
| `policy_exports/walk2_subject3_model29999_official/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `ca62112e1574` |
| `policy_exports/walk2_subject3_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `cd28087fcf23` |
| `policy_exports/walk2_subject4_model29999_official/agent.yaml` | Adapter/Controller | Isaac adapter, motion/data | not uploaded | False | False | False | `831716f61b38` |
| `policy_exports/walk2_subject4_model29999_official/env.yaml` | Adapter/Controller | Isaac adapter | not uploaded | False | False | False | `3f3fe6b61c5d` |
| `report/README.md` | Stage-3 | VAE, teacher asset/rollout, diffusion, MuJoCo adapter/probe, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `6690440c9290` |
| `report/REPORT_FILE_MAP.md` | Stage-3 | VAE, teacher asset/rollout, rollout, diffusion, guidance, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `f9c45efe7cdc` |
| `report/appendix/environment_summary.md` | Stage-3 | VAE, diffusion, MuJoCo adapter/probe, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `55669b61cc6d` |
| `report/appendix/equations.md` | Stage-3 | VAE, latent, diffusion, denoiser, guidance | docs/stage2_stage3_source_inventory.md | False | False | False | `c1970796101a` |
| `report/appendix/full_code_snippets.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `f93e0cf23a4c` |
| `report/appendix/references.md` | Adapter/Controller | controller, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `7b8280a7e84c` |
| `report/appendix/unresolved_details.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, diffusion, MuJoCo adapter/probe, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `54c1206534e8` |
| `report/audits/model_chain_hard_gate_review_20260624.md` | Stage-3 | VAE, teacher asset/rollout, rollout, diffusion, guidance, MuJoCo adapter/probe, Isaac adapter, controller, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `d5d13cc1bf53` |
| `report/code_review/code_inventory.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `f93e0cf23a4c` |
| `report/code_review/key_code_index.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `f93e0cf23a4c` |
| `report/code_review/key_snippets_diffusion.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, motion/data, trajectory | src/beyondmimic_repro/stage3/ | True | False | False | `f93e0cf23a4c` |
| `report/code_review/key_snippets_guidance.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, guidance, motion/data, trajectory | src/beyondmimic_repro/stage3/ | True | False | False | `f93e0cf23a4c` |
| `report/code_review/key_snippets_mujoco.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, MuJoCo adapter/probe, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `f93e0cf23a4c` |
| `report/code_review/key_snippets_tracking.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `f93e0cf23a4c` |
| `report/code_review/key_snippets_vae.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `f93e0cf23a4c` |
| `report/code_review/pseudocode_all_stages.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `6dc1d83e92e8` |
| `report/code_snippets.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `f93e0cf23a4c` |
| `report/data/data_processing_flow.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, diffusion, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `7e1f74a224ab` |
| `report/data/dataset_inventory.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, diffusion, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `7e1f74a224ab` |
| `report/data_report.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, diffusion, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `7e1f74a224ab` |
| `report/executive_summary.md` | Stage-3 | teacher asset/rollout, diffusion, MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | False | `7284e01440c2` |
| `report/experiment_results.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `d11f5abd0553` |
| `report/experiments/diffusion_analysis.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data | src/beyondmimic_repro/stage3/ | True | False | False | `d11f5abd0553` |
| `report/experiments/experiment_inventory.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `d11f5abd0553` |
| `report/experiments/guidance_analysis.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data | src/beyondmimic_repro/stage3/ | True | False | False | `d11f5abd0553` |
| `report/experiments/metrics_summary.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `d11f5abd0553` |
| `report/experiments/mse_denoising_analysis.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `d11f5abd0553` |
| `report/experiments/tracking_analysis.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `d11f5abd0553` |
| `report/experiments/vae_analysis.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `d11f5abd0553` |
| `report/failure_analysis.md` | Stage-3 | VAE, teacher asset/rollout, latent, diffusion, MuJoCo adapter/probe | docs/stage2_stage3_source_inventory.md | False | False | False | `6ef2b2cdd73a` |
| `report/limitations_and_next_steps.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `6e3aff84a0df` |
| `report/logs/failure_logs/failure_analysis.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `c51389606386` |
| `report/logs_summary/log_inventory.md` | Adapter/Controller | teacher asset/rollout, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `2bc692e4b17d` |
| `report/module_pipeline.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `9cc5f73a03db` |
| `report/next_steps.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `6e3aff84a0df` |
| `report/paper_alignment.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, latent, diffusion, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `053a408f1eaf` |
| `report/paper_vs_project.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, latent, diffusion, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `053a408f1eaf` |
| `report/pipeline/data_flow.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `9cc5f73a03db` |
| `report/pipeline/failure_diagnosis.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `9cc5f73a03db` |
| `report/pipeline/full_pipeline_mermaid.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `9cc5f73a03db` |
| `report/pipeline/mujoco_or_isaac_rendering.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | src/beyondmimic_repro/adapters/ | True | False | False | `9cc5f73a03db` |
| `report/pipeline/pipeline_overview.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `9cc5f73a03db` |
| `report/pipeline/stage1_tracking.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `9cc5f73a03db` |
| `report/pipeline/stage2_vae.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `9cc5f73a03db` |
| `report/pipeline/stage3_diffusion.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | src/beyondmimic_repro/stage3/ | True | False | False | `9cc5f73a03db` |
| `report/pipeline/stage4_guidance.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data, trajectory | src/beyondmimic_repro/stage3/ | True | False | False | `9cc5f73a03db` |
| `report/pseudocode.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `6dc1d83e92e8` |
| `report/report_main.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, waypoint guidance, obstacle guidance, inpainting, MuJoCo adapter/probe, Isaac adapter, controller, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `b05a6229f0b9` |
| `report/reproduction_status.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, latent, diffusion, guidance, MuJoCo adapter/probe, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `053a408f1eaf` |
| `report/video_index.md` | Stage-3 | VAE, teacher asset/rollout, latent, diffusion, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `90b00a420a26` |
| `report/videos/video_index.md` | Stage-3 | VAE, teacher asset/rollout, latent, diffusion, MuJoCo adapter/probe, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `90b00a420a26` |
| `reproduction/PROGRESS.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `52fc1283d7d2` |
| `reproduction/README.md` | Stage-3 | VAE, rollout, diffusion, Isaac adapter, controller | docs/stage2_stage3_source_inventory.md | False | False | False | `db2a9996ecca` |
| `reproduction/RUNBOOK.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `f55583ef168e` |
| `reproduction/docs/chinese_project_report.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, latent, diffusion, guidance, waypoint guidance, obstacle guidance, inpainting, Isaac adapter, ONNX/controller export, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `7a93b6c3a6c2` |
| `reproduction/docs/chinese_reading_report.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, latent, diffusion, guidance, Isaac adapter, motion/data, trajectory | docs/stage2_stage3_source_inventory.md | False | False | False | `90968ed65331` |
| `reproduction/docs/completion_matrix.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `18ad6b25ae9e` |
| `reproduction/docs/current_environment_and_reproduction_status.md` | Stage-3 | VAE, latent, diffusion, guidance | docs/stage2_stage3_source_inventory.md | False | False | False | `9ca873390873` |
| `reproduction/docs/current_goal_baseline_20260622.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, Isaac adapter, ONNX/controller export, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `95f50499ff40` |
| `reproduction/docs/current_project_reproduction_state_20260621.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, diffusion, guidance, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `0c64bfcb635c` |
| `reproduction/docs/current_project_reproduction_state_20260622.md` | Stage-3 | VAE, diffusion, guidance, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `df96ad15fb72` |
| `reproduction/docs/current_project_reproduction_summary_20260622.md` | Stage-3 | VAE, DAgger, teacher asset/rollout, rollout, latent, diffusion, denoiser, guidance, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `f0f387f6d8be` |
| `reproduction/docs/current_reproduction_baseline_20260622.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `8587aa3f1c35` |
| `reproduction/docs/deployment_controller_audit.md` | Adapter/Controller | MuJoCo adapter/probe, controller, motion/data | src/beyondmimic_repro/adapters/ | True | False | False | `a512de9ce635` |
| `reproduction/docs/discrepancy_report.md` | Adapter/Controller | Isaac adapter, controller, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `15f83da830ea` |
| `reproduction/docs/english_reading_report.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, guidance, Isaac adapter, controller, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `87444cdc96c4` |
| `reproduction/docs/environment.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `70b58d2b5b70` |
| `reproduction/docs/environment_plan.md` | Stage-3 | diffusion, Isaac adapter, controller, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `c51956184f05` |
| `reproduction/docs/experiment_protocol.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `66fe10bbb8c7` |
| `reproduction/docs/final_reproduction_report.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `5cc3ddde4658` |
| `reproduction/docs/known_limitations.md` | Adapter/Controller | Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `dbfe9af73cb2` |
| `reproduction/docs/level_b_tracking_protocol.md` | Adapter/Controller | Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `ac3accbee965` |
| `reproduction/docs/level_c_diffusion_plan.md` | Stage-3 | VAE, latent, diffusion, ONNX/controller export, controller, motion/data | src/beyondmimic_repro/stage3/ | True | False | False | `7f5202b4544f` |
| `reproduction/docs/paper_parameter_map.md` | Stage-3 | VAE, rollout, diffusion, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `d3ed1f5c1dec` |
| `reproduction/docs/progress/20260618_163028_baseline_gitignore_and_report_plan.md` | Stage-3 | diffusion, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `e7f8157f2c97` |
| `reproduction/docs/progress/20260618_165051_isaaclab_live_gate_probe.md` | Adapter/Controller | Isaac adapter | src/beyondmimic_repro/adapters/ | True | False | False | `d5a577ed748e` |
| `reproduction/docs/progress/20260618_170858_vulkan_egl_icd_and_cuda_p2p_gate.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `4c3ffef9f7cb` |
| `reproduction/docs/progress/20260618_172145_cuda_p2p_iommu_gate_refinement.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `ae6559ed854f` |
| `reproduction/docs/progress/20260618_173239_gpu_foundation_settings_surface.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `df4746571a2f` |
| `reproduction/docs/progress/20260618_175306_isaaclab_live_gate_and_replay_preflight.md` | Adapter/Controller | Isaac adapter | src/beyondmimic_repro/adapters/ | True | False | False | `7a0ca3eb35b6` |
| `reproduction/docs/progress/20260618_181859_official_replay_conversion_attempt.md` | Adapter/Controller | Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `fa42253f18da` |
| `reproduction/docs/progress/20260618_183003_urdf_conversion_probe.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `25a079f6da81` |
| `reproduction/docs/progress/20260618_185446_urdf_path_tiny_probe.md` | Adapter/Controller | Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | True | `102026ef5b64` |
| `reproduction/docs/progress/20260618_190541_mjcf_stage_probe.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `982a1d5a00f6` |
| `reproduction/docs/progress/20260618_191414_usd_save_policy_probe.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `3514779e368b` |
| `reproduction/docs/progress/20260618_192757_simulationapp_save_policy_probe.md` | Stage-3 | VAE, DAgger, rollout, diffusion, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `6605bbc09991` |
| `reproduction/docs/progress/20260618_194457_usd_stage_export_workaround.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `4caf282d1672` |
| `reproduction/docs/progress/20260618_195555_g1_urdf_stage_export_workaround.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `343310cc2b41` |
| `reproduction/docs/progress/20260618_201452_g1_urdf_layer_save_workaround.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `627fd8e551d7` |
| `reproduction/docs/progress/20260618_204005_g1_urdf_in_memory_import.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `e36d3300f14c` |
| `reproduction/docs/progress/20260618_205623_g1_simulationapp_in_memory_import.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `f7022e1bb350` |
| `reproduction/docs/progress/20260618_210850_g1_in_memory_variant_matrix.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `23398e8d3093` |
| `reproduction/docs/progress/20260618_212122_g1_preconverted_asset_audit.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `02ddd99e4518` |
| `reproduction/docs/progress/20260618_213302_g1_reference_usd_compatibility.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `e622a0ce2e7c` |
| `reproduction/docs/progress/20260618_215000_g1_official_urdf_skeleton_usd.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `35312fe850e3` |
| `reproduction/docs/progress/20260618_220827_g1_urdf_physical_asset_contract.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `eba9e83982d0` |
| `reproduction/docs/progress/20260618_222042_g1_resource_adjusted_enriched_usd.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | True | `8af6d174ab0e` |
| `reproduction/docs/progress/20260618_224743_enriched_usd_replay_gate.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `936ec97bb057` |
| `reproduction/docs/progress/20260618_225219_enriched_gate_explicit_exit.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `f2174b8b2529` |
| `reproduction/docs/progress/20260618_225812_enriched_bounded_replay_metrics.md` | Stage-1/Shared source | motion/data | docs/stage2_stage3_source_inventory.md | False | False | True | `f592c3f9671b` |
| `reproduction/docs/progress/20260618_231035_resource_adjusted_task_smoke.md` | Stage-1/Shared source | motion/data | docs/stage2_stage3_source_inventory.md | False | False | True | `36c9a4d3f526` |
| `reproduction/docs/progress/20260619_001351_resource_adjusted_full_fixture_eval.md` | Stage-2 | DAgger, rollout | docs/stage2_stage3_source_inventory.md | False | False | True | `449d6ffd3d4a` |
| `reproduction/docs/progress/20260619_003039_resource_adjusted_csv_replay.md` | Adapter/Controller | Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | True | `206af6e7c38f` |
| `reproduction/docs/progress/20260619_004056_resource_adjusted_csv_task_eval.md` | Stage-1/Shared source | motion/data | docs/stage2_stage3_source_inventory.md | False | False | True | `dfc0be2a0de5` |
| `reproduction/docs/progress/20260619_005657_resource_adjusted_train_entry_diagnostic.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | True | `ac33c4f6f7c6` |
| `reproduction/docs/progress/20260619_010644_blocked_gate_state_correction.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `0e6ab6ec2785` |
| `reproduction/docs/progress/20260619_012123_g1_urdf_source_equivalence.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `52c3cb360c9f` |
| `reproduction/docs/progress/20260619_013245_official_replay_entry_diagnostic.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `ab77ac8ccc7b` |
| `reproduction/docs/progress/20260619_020706_g1_import_config_probe.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `42294381cb0d` |
| `reproduction/docs/progress/20260619_021821_resource_adjusted_ppo_harness.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | True | `ebae50e45f78` |
| `reproduction/docs/progress/20260619_042556_resource_adjusted_ppo_checkpoint_eval.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | True | `30262c498262` |
| `reproduction/docs/progress/20260619_045331_resource_adjusted_teacher_rollout_dataset.md` | Stage-2 | teacher asset/rollout, rollout | docs/stage2_stage3_source_inventory.md | False | False | True | `0c3e3cae761d` |
| `reproduction/docs/progress/20260619_050453_current_isaaclab_headless_gate.md` | Adapter/Controller | Isaac adapter | src/beyondmimic_repro/adapters/ | True | False | False | `3c3708af1fb0` |
| `reproduction/docs/progress/20260619_051324_official_replay_entry_gpu4.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `e05f55ef7815` |
| `reproduction/docs/progress/20260619_053608_resource_adjusted_teacher_rollout_vae.md` | Stage-2 | VAE, teacher asset/rollout, rollout | docs/stage2_stage3_source_inventory.md | False | False | True | `cc5e2e2b19da` |
| `reproduction/docs/progress/20260619_055350_resource_adjusted_state_latent_diffusion.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser | src/beyondmimic_repro/stage3/datasets/state_latent_builder.py | True | False | True | `8986c6582cf1` |
| `reproduction/docs/progress/20260619_060814_resource_adjusted_state_latent_guidance.md` | Stage-3 | state-latent, latent, denoiser, guidance, Isaac adapter | src/beyondmimic_repro/stage3/datasets/state_latent_builder.py | True | False | True | `8a4ca5a181b3` |
| `reproduction/docs/progress/20260619_062522_official_replay_loop_patch.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `846345a0b56e` |
| `reproduction/docs/progress/20260619_063927_official_csv_to_npz_loop_patch.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `7f88f030e773` |
| `reproduction/docs/progress/20260619_110613_official_csv_loop_ppo_training_eval.md` | Stage-1/Shared source | motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `377a574bdc50` |
| `reproduction/docs/progress/20260619_113142_official_csv_loop_teacher_rollout.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion | docs/stage2_stage3_source_inventory.md | False | False | False | `bc3279c4d551` |
| `reproduction/docs/progress/20260619_114318_official_csv_loop_teacher_rollout_vae.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent | docs/stage2_stage3_source_inventory.md | False | False | False | `f61987eb50b8` |
| `reproduction/docs/progress/20260619_115739_official_csv_loop_state_latent_diffusion.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser | src/beyondmimic_repro/stage3/datasets/state_latent_builder.py | True | False | False | `26bf3d6527a7` |
| `reproduction/docs/progress/20260619_120655_official_csv_loop_guidance_eval.md` | Stage-3 | state-latent, latent, diffusion, denoiser, guidance | src/beyondmimic_repro/stage3/ | True | False | False | `13a0b6187953` |
| `reproduction/docs/progress/20260619_121423_english_reading_report.md` | Stage-3 | state-latent, latent, guidance | docs/stage2_stage3_source_inventory.md | False | False | False | `c4f11783cda7` |
| `reproduction/docs/progress/20260619_122522_guided_decode_report_assets.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, guidance | docs/stage2_stage3_source_inventory.md | False | False | False | `c8549f8fb63a` |
| `reproduction/docs/progress/20260619_124849_headless_replay_gpu47_task_eval.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `c8cddb1fbb65` |
| `reproduction/docs/progress/20260619_125626_ppo_eval_report_assets.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `b1c6709be443` |
| `reproduction/docs/progress/20260619_131539_reference_replay_visual_asset.md` | Adapter/Controller | Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `c025f799f095` |
| `reproduction/docs/progress/20260619_132352_teacher_rollout_report_assets.md` | Stage-2 | DAgger, teacher asset/rollout, rollout | docs/stage2_stage3_source_inventory.md | False | False | False | `9b5ae63627c0` |
| `reproduction/docs/progress/20260619_134243_policy_rollout_video_capture.md` | Stage-1/Shared source | rollout, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `53d9c3aa87a2` |
| `reproduction/docs/progress/20260619_140409_guided_action_rollout_probe.md` | Stage-3 | VAE, rollout, state-latent, latent, guidance, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `a137e0f9b0cc` |
| `reproduction/docs/progress/20260619_144607_vae_closed_loop_rollout_eval.md` | Stage-2 | VAE, teacher asset/rollout, rollout, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `124235f90073` |
| `reproduction/docs/progress/20260619_150403_vae_closed_loop_video_asset.md` | Stage-2 | VAE, teacher asset/rollout, rollout, Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `884dc300018f` |
| `reproduction/docs/progress/20260619_152640_action_guidance_rollout.md` | Stage-3 | VAE, rollout, latent, diffusion, guidance, Isaac adapter, motion/data | src/beyondmimic_repro/stage3/ | True | False | False | `ac763cff7de1` |
| `reproduction/docs/progress/20260619_153439_environment_reproduction_status.md` | Stage-3 | diffusion, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `5af4acebb370` |
| `reproduction/docs/progress/20260619_155833_receding_latent_guidance_rollout.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, guidance | src/beyondmimic_repro/stage3/ | True | False | False | `ab96adcb790c` |
| `reproduction/docs/progress/20260619_164442_task_conditioned_guidance_rollouts.md` | Stage-3 | rollout, latent, guidance | src/beyondmimic_repro/stage3/ | True | False | False | `495fdac6e1be` |
| `reproduction/docs/progress/20260619_165242_task_conditioned_report_assets.md` | Stage-3 | rollout, latent, guidance | docs/stage2_stage3_source_inventory.md | False | False | False | `30a96b237cdc` |
| `reproduction/docs/progress/20260619_170528_onnx_async_deployment_audit.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, ONNX/controller export | src/beyondmimic_repro/adapters/ | True | False | False | `d54642a0f7d0` |
| `reproduction/docs/progress/20260619_175558_official_csv_loop_multiseed_eval.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `6f387797707b` |
| `reproduction/docs/progress/20260619_192353_task_conditioned_guidance_multiseed.md` | Stage-3 | rollout, latent, guidance | src/beyondmimic_repro/stage3/ | True | False | False | `1657856cb5d0` |
| `reproduction/docs/progress/20260619_202806_full_official_csv_loop_conversion.md` | Adapter/Controller | Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `a747cc1a29d8` |
| `reproduction/docs/progress/20260619_212556_full_official_replay_loop_conversion.md` | Adapter/Controller | Isaac adapter, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `c7cc103dbc37` |
| `reproduction/docs/progress/20260619_230538_full_official_csv_task_eval.md` | Stage-1/Shared source | motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `7deba9f5dde3` |
| `reproduction/docs/progress/20260620_032848_full_bundle_ppo.md` | Stage-1/Shared source | motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `dceb13b84873` |
| `reproduction/docs/progress/20260620_033151_full_bundle_policy_rollout_video.md` | Stage-1/Shared source | rollout, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `f4ffb032a9a2` |
| `reproduction/docs/progress/20260620_035457_full_bundle_teacher_rollout.md` | Stage-2 | VAE, teacher asset/rollout, rollout, latent, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `a4ee251c55c4` |
| `reproduction/docs/progress/20260620_041250_full_bundle_downstream.md` | Stage-3 | VAE, teacher asset/rollout, rollout, latent, diffusion, motion/data | docs/stage2_stage3_source_inventory.md | False | False | False | `78ceddd16480` |
| `reproduction/docs/progress/20260620_042310_full_bundle_guidance.md` | Stage-3 | state-latent, latent, denoiser, guidance, motion/data | src/beyondmimic_repro/stage3/ | True | False | False | `1a612b3e9f32` |
| `reproduction/docs/progress/20260620_044502_full_bundle_task_conditioned_multiseed.md` | Stage-3 | rollout, latent, guidance | docs/stage2_stage3_source_inventory.md | False | False | False | `e30a4dce53de` |
| `reproduction/docs/progress/20260620_045238_guided_vs_unguided_matrix.md` | Stage-3 | rollout, guidance | docs/stage2_stage3_source_inventory.md | False | False | False | `606c669971d6` |
| `reproduction/docs/progress/20260620_045830_headless_gate_recheck.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `723933f32458` |
| `reproduction/docs/progress/20260620_051520_full_bundle_receding_guidance_rollout.md` | Stage-3 | VAE, rollout, latent, diffusion, guidance, motion/data | src/beyondmimic_repro/stage3/ | True | False | False | `3abf8a66831f` |
| `reproduction/docs/progress/20260620_052245_visual_evidence_index.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `63b109d88dd7` |
| `reproduction/docs/progress/20260620_052904_full_bundle_onnx_async.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, diffusion, denoiser, ONNX/controller export, motion/data | src/beyondmimic_repro/adapters/ | True | False | False | `d0ac3152d20a` |
| `reproduction/docs/progress/20260620_061539_full_bundle_task_conditioned_guidance.md` | Stage-3 | rollout, latent, guidance | src/beyondmimic_repro/stage3/ | True | False | False | `b3daf77ea072` |
| `reproduction/docs/progress/20260620_125205_current_environment_and_headless_gate.md` | Stage-2 | teacher asset/rollout, rollout, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `99a87bc1cbf2` |
| `reproduction/docs/progress/20260620_125839_visual_report_appendix.md` | Stage-3 | VAE, teacher asset/rollout, rollout, state-latent, latent, Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `38f462a68685` |
| `reproduction/docs/progress/20260620_140639_full_bundle_guidance_5seed.md` | Stage-3 | rollout, latent, guidance | src/beyondmimic_repro/stage3/ | True | False | False | `046ca08f7777` |
| `reproduction/docs/progress/20260620_141440_reading_report_5seed_sync.md` | Stage-3 | latent, guidance | docs/stage2_stage3_source_inventory.md | False | False | False | `7088d026a790` |
| `reproduction/docs/progress/20260620_142443_guidance_success_boundary.md` | Stage-3 | rollout, latent, guidance | src/beyondmimic_repro/stage3/ | True | False | False | `b146188bcfd3` |
| `reproduction/docs/progress/20260620_143445_guidance_video_contact_sheet.md` | Stage-3 | rollout, latent, guidance | src/beyondmimic_repro/stage3/ | True | False | False | `4923528079ca` |
| `reproduction/docs/progress/20260620_145259_g1_urdf_gpu4_export_structure.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `3690db1d4915` |
| `reproduction/docs/progress/20260620_153810_official_importer_export_full_task_eval.md` | Adapter/Controller | Isaac adapter | docs/stage2_stage3_source_inventory.md | False | False | False | `89eaa79649be` |
| `reproduction/docs/progress/20260620_160442_official_importer_export_ppo.md` | Related | related source/config/report | docs/stage2_stage3_source_inventory.md | False | False | False | `0063d49c86e5` |
| `reproduction/docs/progress/20260620_162407_official_importer_teacher_rollout.md` | Stage-2 | DAgger, teacher asset/rollout, rollout | docs/stage2_stage3_source_inventory.md | False | False | False | `d2e2f8a57e20` |
| `reproduction/docs/progress/20260620_163810_official_importer_vae_training.md` | Stage-2 | VAE, teacher asset/rollout, rollout | src/beyondmimic_repro/stage2/models/conditional_action_vae.py | True | False | False | `3f79646a2ee9` |
| `reproduction/docs/progress/20260620_170505_official_importer_vae_closed_loop.md` | Stage-2 | VAE, rollout | docs/stage2_stage3_source_inventory.md | False | False | False | `6c6b627ea8a0` |

Preview truncated to 250 rows. See `docs/stage2_stage3_source_inventory.json` for all 3726 candidates.
