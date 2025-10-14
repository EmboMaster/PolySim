<h1 align="center"> PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via
Multi-Simulator Dynamics Randomization </h1>


<div align="center">


ICRA 2025
[[Arxiv]](https://arxiv.org/abs/2510.01708)
[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym) [![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/index.html) [![IsaacSim](https://img.shields.io/badge/Genesis-0.2.1-b.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/index.html) [![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

</div>

<!-- # Table of Contents -->

## Table of Contents

1. **[Overview](#overview)**  
   - Links: [Arxiv](https://arxiv.org/abs/2510.01708) 

2. **[Installation & Setup](#installation)**  
   2.1 [Base Frameworks](#isaacgym-conda-env)  
   2.2 [IsaacGym Setup](#install-isaacgym)  
   2.3 [HumanoidVerse Setup](#install-humanoidverse)  
   2.4 [IsaacSim + IsaacLab Setup](#isaaclab-environment)  
   2.5 [Genesis Environment Setup](#genesis-environment)  
   2.6 [MuJoCo Environment Setup](#mujoco-environment) 

3. **[Parallel Training Pipelines](#parallel-motion-tracking-training)**  
   3.1 [Setup](#setup)  
   3.2 [Notes](#notes) 

4. **[Citation](#citation)**  

5. **[License](#license)**


## TODO List

- [x] Release code backbone
- [x] Release parallel motion tracking training pipeline
- [x] Release sim2sim in MuJoCo
- [ ] Release sim2real videos with UnitreeSDK on g1

# Installation

PolySim is built on top of [ASAP](https://github.com/LeCAR-Lab/ASAP), leveraging their multi-simulator and humanoid tracking capabilities.

ASAP focuses on studying the transfer of policies across different simulators and real-world systems, providing a platform for robust humanoid learning. It is powered by [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse) (a multi-simulator framework for humanoid learning)
, a modular framework designed to seamlessly train humanoid skills in multiple simulators such as IsaacGym, IsaacSim, and Genesis. These frameworks share a unified design that decouples simulators, tasks, and algorithms, enabling easy transitions between simulators and the real world with minimal effort (often as simple as a single line of code change).

We thank the developers of ASAP and HumanoidVerse for their foundational work that made PolySim possible.

## IsaacGym Conda Env

Create conda environment

```bash
conda create -n hvgym python=3.8
conda activate hvgym
```

### Install IsaacGym

Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and extract:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```

Test installation:

```bash
python 1080_balls_of_solitude.py  # or
python joint_monkey.py
```

For libpython error:

- Check conda path:

  ```bash
  conda info -e
  ```

- Set LD_LIBRARY_PATH:

  ```bash
  export LD_LIBRARY_PATH=</path/to/conda/envs/your_env/lib>:$LD_LIBRARY_PATH
  ```

### Install HumanoidVerse

Install dependencies:

```bash
pip install -e .
pip install -e isaac_utils
pip install -r requirements.txt
```

Test with:

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=1 \
project_name=TestIsaacGymInstallation \
experiment_name=G123dof_loco \
headless=False
```

## IsaacLab Environment

### Install IsaacSim-v4.2.0

1. Download Omniverse Launcher

2. Install Isaac Sim through launcher

   refer to https://isaac-sim.github.io/IsaacLab/v1.4.1/source/setup/installation/pip_installation.html

3. Set environment variables:

```bash
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.2.0"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

### Install IsaacLab-v1.4.1

​	refer to https://isaac-sim.github.io/IsaacLab/v1.4.1/source/setup/installation/pip_installation.html

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --conda hvlab
mamba activate hvlab
sudo apt install cmake build-essential
./isaaclab.sh --install
```

### Setup HumanoidVerse

```bash
pip install -e .
pip install -e isaac_utils
```

## Genesis Environment

```bash
conda create -n hvgen python=3.10
conda activate hvgen
pip install genesis-world==0.2.1 torch
```

Install dependencies:

```bash
pip install -e .
pip install -e isaac_utils
```

#### **Patch for `igl.signed_distance` Issue**

To fix the runtime error, apply the following change to handle the tuple returned by `igl.signed_distance`.

**File to Edit:**

```
.../genesis/engine/entities/rigid_entity/rigid_geom.py
```

**Action:**

Replace the methods `_compute_sd` and `_compute_closest_verts` (around lines 223-228) with the code below.

Python

```
def _compute_sd(self, query_points):
    sd = igl.signed_distance(query_points, self._sdf_verts, self._sdf_faces)[0]
    return sd

def _compute_closest_verts(self, query_points):
    closest_faces = igl.signed_distance(query_points, self._init_verts, self._init_faces)[1]
    return closest_faces
```

This change correctly indexes the output tuple to get the signed distance (`[0]`) and closest faces (`[1]`) respectively.

## MuJoCo Environment

```bash
conda create -n hvmoj python=3.10
conda activate hvgen
pip install mujoco torch
```

Install dependencies:

```bash
pip install -e .
pip install -e isaac_utils
```

# Parallel Motion Tracking Training

This section outlines the process for setting up and running parallel motion tracking training across different simulators. The training involves three simulators: IsaacGym, IsaacSim, and Genesis. These simulators will run on separate terminals and communicate via RPC (Remote Procedure Call), allowing for scalable and parallelized training. You only need to use `polysim_train_agent.py` for routing and parallel training.

### Start Parallel Training with PolySim

Run the following command to initiate parallel training using the `polysim_train_agent.py` script. This will route tasks across the three simulators and train your agent in parallel.

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/polysim_train_agent.py \
+simulator_list=[isaacgym,isaacsim,genesis] \ # choose 1 or 2 or 3 simulators
+exp=motion_tracking \ 
+domain_rand=NO_domain_rand \ # choose domain config
+rewards=motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
num_envs_list=[2048,2048,2048] \ # choose any number of envs of each simulators
project_name=MotionTracking \
experiment_name=MotionTracking_APT \ # choose any names
robot.motion.motion_file="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_APT_level1_amass.pkl" \  # choose any motion files
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination.curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination.curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0 \
master_port=29590 \ # one program need only one specific port
device_list=[5,6,7,7] # the first three for three simulators, the last for the client
```

### Visualize Your Model

To visualize your model, use the following command in a new terminal. This will launch the MuJoCo visualization for your trained agent:

```bash
python humanoidverse/eval_agent.py \
+simulator=mujoco \ 
+headless=False \
+checkpoint=logs/MotionTracking/xxxxxxxx_xxxxxxx-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_10000.pt
```

### Evaluation

To perform the evaluation and test the performance of a trained model, you can use the following command:
```bash
python humanoidverse/eval_agent.py \
+simulator=mujoco \ # you can change the simulator platform
+headless=True \ # To speed up the process, you can disable visualization
+eval=True \
+checkpoint=logs/MotionTracking/xxxxxxxx_xxxxxxx-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_10000.pt
```

We also provide 

## Notes

- Ensure all dependencies and the appropriate versions of each simulator are installed.
- Each terminal should run one of the commands provided above. Make sure you adjust the GPU settings and environment configurations as needed.

# Citation

If our work helps you well, please cite us as:

```bibtex
@article{polysim2025,
  title={PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization},
  author={Zixing Lei1, Zibo Zhou, Sheng Yin, Yueru Chen, Qingyao Xu, Weixin Li, Yunhong Wang, Bowei Tang, Wei Jing, Siheng Chen},
  journal={arXiv preprint arXiv:2510.01708},
  year={2025}
}
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.