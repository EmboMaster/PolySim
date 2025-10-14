# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class EventTerm:
    """事件项配置"""
    name: str
    func: str
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class SceneEntityCfg:
    """场景实体配置"""
    name: str
    entity_type: str
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class EventCfg:
    """事件配置"""
    name: str
    terms: List[EventTerm] = None
    entities: List[SceneEntityCfg] = None
    
    def __post_init__(self):
        if self.terms is None:
            self.terms = []
        if self.entities is None:
            self.entities = []


# 默认ManiSkill事件配置
DEFAULT_MANISKILL_EVENT_CFG = EventCfg(
    name="default_maniskill_events",
    terms=[
        EventTerm(
            name="randomize_robot_properties",
            func="randomize_maniskill_robot_properties",
            params={
                "mass_range": [0.8, 1.2],
                "friction_range": [0.5, 1.5]
            }
        ),
        EventTerm(
            name="randomize_joint_properties", 
            func="randomize_maniskill_joint_properties",
            params={
                "stiffness_range": [0.8, 1.2],
                "damping_range": [0.8, 1.2]
            }
        ),
        EventTerm(
            name="randomize_task_objects",
            func="randomize_maniskill_task_objects",
            params={
                "pose_noise": 0.01,
                "scale_range": [0.95, 1.05]
            }
        )
    ]
)
