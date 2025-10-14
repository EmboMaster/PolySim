# ManiSkill仿真器

## 概述

ManiSkill仿真器是基于ManiSkill框架的机器人操作仿真环境，为HumanoidVerse项目提供机器人操作任务的仿真支持。

## 特性

- **多任务支持**: 支持PickCube、StackCube、Assemble等操作任务
- **多机器人支持**: 支持Panda、UR5、Fetch等机器人平台
- **灵活控制**: 支持多种控制模式（位置控制、速度控制、力控制等）
- **丰富观察**: 支持状态观察、图像观察等多种观察模式
- **域随机化**: 支持摩擦、质量、COM等参数的随机化
- **事件系统**: 完整的事件管理和域随机化系统
- **相机控制**: 智能视角相机控制和跟踪
- **配置管理**: 结构化的配置系统，易于调整和维护

## 安装依赖

```bash
# 安装ManiSkill
pip install mani-skill

# 安装SAPIEN（ManiSkill的物理引擎）
pip install sapien

# 安装其他依赖
pip install torch numpy opencv-python
```

## 配置说明

### 基础配置

```yaml
simulator:
  _target_: humanoidverse.simulator.maniskill.maniskill.ManiSkill
  config:
    name: "maniskill"
    sim:
      fps: 200                    # 仿真频率
      control_decimation: 4       # 控制频率
      substeps: 1                 # 物理子步数
      gravity: [0.0, 0.0, -9.81] # 重力加速度
```

### 任务配置

```yaml
task:
  task_name: "PickCube-v1"           # 任务名称
  control_mode: "pd_ee_delta_pose"   # 控制模式
  obs_mode: "state"                  # 观察模式
  reward_mode: "dense"               # 奖励模式
```

### 机器人配置

```yaml
robot:
  robot_type: "panda"    # 机器人类型
  asset_path: null       # 自定义机器人资产路径
```

## 使用方法

### 1. 基本使用

```python
from humanoidverse.simulator.maniskill.maniskill import ManiSkill
from humanoidverse.simulator.maniskill.maniskill_cfg import ManiSkillCfg
from omegaconf import OmegaConf

# 加载配置
config = OmegaConf.load("config/simulator/maniskill.yaml")

# 创建仿真器
simulator = ManiSkill(config, device="cuda")

# 设置仿真器
simulator.setup()

# 创建环境
envs, handles = simulator.create_envs(
    num_envs=4,
    env_origins=torch.zeros(4, 3),
    base_init_state=torch.zeros(7 + simulator.num_dof),
    env_config=config
)

# 仿真循环
for step in range(1000):
    # 执行动作
    actions = torch.randn(4, simulator.num_dof)
    simulator.step(actions)
    
    # 获取观察
    observations = simulator.get_observations()
    
    # 渲染
    simulator.render()
```

### 2. 使用配置类

```python
from humanoidverse.simulator.maniskill.maniskill import ManiSkill
from omegaconf import OmegaConf

# 加载配置
config = OmegaConf.load("config/simulator/maniskill.yaml")

# 创建仿真器
simulator = ManiSkill(config, device="cuda")

# 设置仿真器
simulator.setup()

# 创建环境
envs, handles = simulator.create_envs(
    num_envs=4,
    env_origins=torch.zeros(4, 3),
    base_init_state=torch.zeros(7 + simulator.num_dof),
    env_config=config
)

# 仿真循环
for step in range(1000):
    # 执行动作
    actions = torch.randn(4, simulator.num_dof)
    simulator.step(actions)
    
    # 获取观察
    observations = simulator.get_observations()
    
    # 渲染
    simulator.render()
```

### 2. 使用配置类

```python
from humanoidverse.simulator.maniskill.maniskill_cfg import ManiSkillCfg

# 创建配置实例
cfg = ManiSkillCfg()
cfg.task_name = "StackCube-v1"
cfg.robot_type = "ur5"
cfg.num_envs = 8

# 使用配置创建仿真器
simulator = ManiSkill(cfg, device="cuda")
```

### 3. 任务切换

```python
# 切换到不同任务
config.simulator.config.task.task_name = "StackCube-v1"
config.simulator.config.task.control_mode = "pd_ee_delta_pose"

# 重新创建环境
simulator.close()
simulator = ManiSkill(config, device="cuda")
simulator.setup()
```

### 4. 机器人切换

```python
# 切换到不同机器人
config.simulator.config.robot.robot_type = "ur5"
config.simulator.config.robot.asset_path = "/path/to/custom/urdf"

# 重新加载资产
simulator.load_assets(config.robot)
```

### 5. 域随机化

```python
# 启用域随机化
config.domain_rand.randomize_friction = True
config.domain_rand.friction_range = [0.3, 1.7]
config.domain_rand.randomize_mass = True
config.domain_rand.mass_range = [0.7, 1.3]

# 重新设置仿真器
simulator.setup()
```

### 6. 相机控制

```python
# 设置相机跟踪机器人
if simulator.viewport_camera_controller:
    simulator.viewport_camera_controller.origin_type = "robot"
    simulator.viewport_camera_controller.robot_name = "robot"
    
# 自定义相机位置
simulator.viewport_camera_controller.set_camera_pose(
    eye=[3.0, 0.0, 3.0],
    lookat=[0.0, 0.0, 1.0]
)
```

## 支持的任务

### 基础操作任务
- `PickCube-v1`: 拾取立方体
- `StackCube-v1`: 堆叠立方体
- `Assemble-v1`: 组装任务
- `Disassemble-v1`: 拆卸任务

### 高级操作任务
- `PegInsertion-v1`: 插销插入
- `ScrewDriver-v1`: 螺丝刀操作
- `Hammer-v1`: 锤击任务

## 支持的控制模式

- `pd_ee_delta_pose`: 末端执行器位置增量控制
- `pd_ee_target_delta_pose`: 末端执行器目标位置增量控制
- `pd_joint_delta_pos`: 关节位置增量控制
- `pd_joint_target_delta_pos`: 关节目标位置增量控制
- `pd_joint_pos`: 关节位置控制
- `pd_joint_vel`: 关节速度控制

## 支持的观察模式

- `state`: 状态观察（关节位置、速度等）
- `rgbd`: RGB-D图像观察
- `pointcloud`: 点云观察
- `state_dict`: 状态字典观察

## 故障排除

### 常见问题

1. **ManiSkill未安装**
   ```bash
   pip install mani-skill
   ```

2. **SAPIEN依赖问题**
   ```bash
   pip install sapien
   ```

3. **CUDA版本不兼容**
   - 检查CUDA版本与PyTorch版本的兼容性
   - 使用CPU版本进行测试

4. **任务加载失败**
   - 检查任务名称是否正确
   - 确认ManiSkill版本支持该任务

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查环境状态
print(f"Number of environments: {len(simulator.envs)}")
print(f"Robot DOFs: {simulator.num_dof}")
print(f"Robot bodies: {simulator.num_bodies}")
```

## 性能优化

1. **使用GPU**: 确保CUDA可用
2. **批量处理**: 使用多个环境并行仿真
3. **减少渲染**: 在训练时使用无头模式
4. **优化物理参数**: 调整物理求解器参数

## 扩展开发

### 添加新任务

1. 在ManiSkill中注册新任务
2. 更新配置文件中的任务选项
3. 实现任务特定的观察和奖励函数

### 添加新机器人

1. 准备机器人URDF文件
2. 更新机器人配置
3. 实现机器人特定的控制接口

## 参考资源

- [ManiSkill官方文档](https://maniskill.readthedocs.io/)
- [SAPIEN物理引擎](https://sapien.ucsd.edu/)
- [HumanoidVerse项目](https://github.com/your-repo/humanoidverse)
