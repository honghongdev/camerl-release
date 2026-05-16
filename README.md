# CaMeRL: Collision-Aware and Memory-Enhanced Reinforcement Learning for UAV Navigation in Multi-Scale Obstacle Environments

**[Project Page](https://honghongdev.github.io/camerl/)** | **[arXiv](https://arxiv.org/abs/2605.14810)** | **[Video](https://www.youtube.com/watch?v=GsZT1YcQKAY)**


In obstacle avoidance navigation of unmanned aerial vehicles (UAVs), variations in obstacle scale have received strangely less attention than obstacle number or density. Existing methods typically extract purely geometric features from single-frame depth observations. Such representations tend to neglect small obstacles and lose spatial context under occlusions caused by large obstacles, leading to noticeable degradation in environments with multi-scale obstacles. To address this issue, we propose CaMeRL, a Collision-aware and Memory-enhanced Reinforcement Learning framework for UAV navigation. The collision-aware latent representation encodes risk-sensitive depth cues to preserve fine-grained obstacle structures, improving its sensitivity to small obstacles. The temporal memory module integrates observations across frames, mitigating partial observability caused by large-obstacle occlusions. We evaluate CaMeRL with multi-scale obstacles, including ultra-small and extra-large obstacle settings. Results show that CaMeRL outperforms state-of-the-art baselines across all scales, with success rate gains of 0.48 and 0.28 in the ultra-small and extra-large settings, respectively. More importantly, CaMeRL shows its capability of reliable navigation in cluttered outdoor environments.

This repository contains the full open-source training pipeline.

<p align="center">
  <img src="assets/images/fig2_architecture.png" alt="CaMeRL architecture" width="100%">
  <br>
  <em>Overview of the CaMeRL architecture and training pipeline.</em>
</p>

## Demonstrations

**Multi-scale obstacle avoidance in simulation.** The policy is trained at a nominal obstacle scale and evaluated zero-shot across six scales ranging from ultra-small (1–5&nbsp;cm) to extra-large (400–500&nbsp;cm).

https://github.com/user-attachments/assets/63f93531-a69a-463a-baf9-287ec0679123

**Real-world deployment.** CaMeRL is deployed fully onboard a 250&nbsp;mm quadrotor (Intel RealSense D435, NVIDIA Jetson Orin NX, Pixhawk 6C mini); the representation modules are fine-tuned on real-world depth data while the policy transfers directly from simulation.

https://github.com/user-attachments/assets/246de680-a529-4754-bfa3-5eb01c9c7600

A higher-resolution gallery with per-scale clips is available on the [project page](https://honghongdev.github.io/camerl/).

# 1. Pipeline Overview

The training pipeline has five stages:

1. **Initial PPO** — train a vision-based PPO in a lightly cluttered environment to get a reasonable starting policy.
2. **Data collection** — roll out the initial policy to collect a large dataset of depth-image sequences.
3. **CA preprocessing** — convert each depth frame to a collision-aware depth frame offline (GPU raycasting through cube meshes placed on detected edges).
4. **VAE (CA target)** — train a VAE whose encoder takes raw depth and whose decoder reconstructs the CA map.
5. **LSTM** — freeze the VAE encoder and train an LSTM head for temporal reasoning using the collected dataset.
6. **PPO retraining** — freeze VAE + LSTM and fine-tune PPO in denser environments.

Each entry script has a `# Key config` block at the top — edit the path constants there (they are relative placeholders) before running.

# 2. Installation

Follow the [AvoidBench](https://github.com/tudelft/AvoidBench) installation instructions first. Then place `camerl` as a ROS package under `AvoidBench/src/`:

```bash
cd AvoidBench/src
git clone <repository-url> camerl
```

Create the conda environment and install:

```bash
cd camerl
conda env create -f environment.yaml
conda activate camerl
pip install .
```

Install the reinforcement learning environment (from AvoidBench):

```bash
cd ../avoidbench/avoidlib/build
cmake ..
make -j
pip install .
```

Build the ROS package (needed only for stage 5 / real-time deployment):

```bash
cd AvoidBench
catkin build
source devel/setup.bash
```

# 3. Training

Unless otherwise noted, run the Unity standalone in a separate terminal before launching any script that talks to the simulator:

```bash
cd AvoidBench/src/avoidbench/unity_scene/
./AvoidBench/AvoidBench.x86_64
```

## 3.1 Stage 1 — Initial PPO

Run (~200 iterations is usually enough):

```bash
python train_policy.py --retrain 0 --train 1 --scene_id 0   # 0: indoor, 1: outdoor forest
```

Checkpoints are saved under `./saved/`. Note the path to the final checkpoint for the next stage.

## 3.2 Stage 2 — Collect depth dataset

Pass the Stage 1 checkpoint via `--weight`:

```bash
python collect_data.py --scene_id 0 --weight ./saved/ppo-init/Policy/iter_00200.pth
```

Rollout files (`rollout_*.npz`) are saved under `./saved/` in a timestamped subdirectory. Note that path for the next two stages.

## 3.3 Stage 3 — CA preprocessing (offline)

```bash
cd ca_proc
python generate.py --input ../saved/dataset/ --output ../saved/dataset-ca/
```

## 3.4 Stage 4 — VAE with CA target

```bash
python trainvae_ca.py --dataset ./saved/dataset-ca --vae-dir ./saved/vae/camerl-1
```

Monitor with TensorBoard:

```bash
tensorboard --logdir ./saved/vae/camerl-1
```

## 3.5 Stage 5 — LSTM (offline, no Unity)

```bash
python train_lstm_without_env.py \
    --vae-path ./saved/vae/camerl-1/best.tar \
    --weight   ./saved/ppo-init/Policy/iter_00200.pth \
    --dataset  ./saved/dataset/ \
    --name     camerl-run1
```

Output is saved to `./saved/lstm/camerl-run1/`.

## 3.6 Stage 6 — PPO retraining

```bash
python train_policy.py --retrain 1 --scene_id 0 --nocontrol 1 \
    --weight ./saved/lstm/camerl-run1/Policy/iter_01950.pth
```

`--nocontrol 1` drops the actor and critic heads from the loaded checkpoint so they are reinitialised for the denser-environment fine-tune.

# 4. Testing

```bash
python test_ppo.py --scene_id 0 --weight ./saved/ppo-final/Policy/iter_02000.pth
```

# 5. Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{hong2025camerl,
  title   = {CaMeRL: Collision-Aware and Memory-Enhanced Reinforcement Learning for UAV Navigation in Multi-Scale Obstacle Environments},
  author  = {Hong, Hong and Liao, Feiyu and Liang, Yongheng and Zhang, Boning and Wang, Haitao and Wu, Hejun},
  journal = {arXiv preprint arXiv:2605.14810},
  year    = {2026},
}
```
