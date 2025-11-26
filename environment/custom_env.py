"""
Dermatology Clinic Triage Environment
Custom Gymnasium environment for reinforcement learning triage optimization.

Author: [Your Name]
Date: 2025-11-21
"""

import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class ClinicEnv(gym.Env):
    """
    Clinic triage environment (vector-based).
    Obs: [age, duration, fever, infection, 8 symptom dims, room_avail, queue_len, time_of_day] (15,)
    Actions (Discrete 8): 0=doctor,1=nurse,2=remote,3=escalate,4=defer,5=idle,6=open_room,7=close_room
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(self, seed: int = 0, max_steps: int = 500):
        super().__init__()
        self.seed(seed)
        self.max_steps = max_steps
        # FIX: Changed shape from (14,) to (15,) to match the actual observation vector
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        # internal state
        self.step_count = 0
        self.num_open_rooms = 1
        self.queue = []
        self.current_patient = None
        self.total_wait = 0.0
        self.last_render = None
        
        # Episode statistics tracking
        self.episode_stats = {
            "correct_triages": 0,
            "incorrect_triages": 0,
            "total_patients": 0,
            "total_reward": 0.0
        }

        # reward tuning defaults
        self.wrong_penalty_base = -2.0
        self.severity_multiplier = {0:1.0, 1:1.2, 2:1.6, 3:2.2}
        self.resource_cost_scale = -0.02
        self.wait_penalty_scale = -0.01
        self.critical_fast_threshold = 5

        # Initialize directly
        self._reset_internal()

    def seed(self, seed=None):
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _sample_patient(self) -> Dict[str, Any]:
        severity = int(np.random.choice([0,1,2,3], p=[0.4,0.35,0.2,0.05]))
        age_norm = float(np.clip(np.random.normal(0.5, 0.15), 0.0, 1.0))
        duration_norm = float(np.clip(np.random.exponential(0.5), 0.0, 1.0))
        fever_flag = 1.0 if (np.random.rand() < (0.05 + 0.2*severity)) else 0.0
        infection_flag = 1.0 if (np.random.rand() < (0.05 + 0.25*severity)) else 0.0
        base = 0.2 + 0.25*severity
        symptom_embed = np.clip(np.random.normal(loc=base, scale=0.08, size=(8,)), 0.0, 1.0)
        return {
            "severity": severity,
            "age_norm": age_norm,
            "duration_norm": duration_norm,
            "fever_flag": fever_flag,
            "infection_flag": infection_flag,
            "symptom_embed": symptom_embed,
            "wait_time": 0.0
        }

    def _form_observation(self, patient: Dict[str,Any]) -> np.ndarray:
        vec = [
            patient["age_norm"],
            patient["duration_norm"],
            patient["fever_flag"],
            patient["infection_flag"],
        ]
        vec += list(patient["symptom_embed"])
        vec += [
            1.0 if self.num_open_rooms > 0 else 0.0,
            np.clip(len(self.queue) / 10.0, 0.0, 1.0),
            np.clip(self.step_count / self.max_steps, 0.0, 1.0)
        ]
        # This vector has length 15 (4 + 8 + 3)
        return np.array(vec, dtype=np.float32)

    def _reset_internal(self):
        self.step_count = 0
        self.num_open_rooms = 1
        self.queue = [self._sample_patient() for _ in range(3)]
        self.current_patient = None
        self.total_wait = 0.0
        self._maybe_spawn_next()
        
        # Reset episode statistics
        self.episode_stats = {
            "correct_triages": 0,
            "incorrect_triages": 0,
            "total_patients": 0,
            "total_reward": 0.0
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._reset_internal()
        obs = self._form_observation(self.current_patient)
        info = {}
        # Gym API: reset must return (obs, info)
        return obs, info

    def _maybe_spawn_next(self):
        if self.current_patient is None and len(self.queue) > 0:
            self.current_patient = self.queue.pop(0)
        elif self.current_patient is None:
            self.current_patient = self._sample_patient()

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        self.step_count += 1

        terminated = False
        truncated = False
        info: Dict[str,Any] = {}
        patient = self.current_patient
        reward = 0.0

        # correct action mapping
        if patient["severity"] == 0:
            correct_action = 2
        elif patient["severity"] == 1:
            correct_action = 1
        elif patient["severity"] == 2:
            correct_action = 0
        else:
            correct_action = 3

        # triage reward / penalty
        if action == correct_action:
            self.episode_stats["correct_triages"] += 1
            if patient["severity"] == 0:
                reward += 1.0
            elif patient["severity"] == 1:
                reward += 1.25
            elif patient["severity"] == 2:
                reward += 2.0
            else:
                reward += 3.0 if patient["wait_time"] < self.critical_fast_threshold else 2.0
        else:
            self.episode_stats["incorrect_triages"] += 1
            mult = self.severity_multiplier.get(patient["severity"], 1.0)
            reward += self.wrong_penalty_base * mult
        
        # Track total patients processed (only for triage actions 0-3)
        if action in [0, 1, 2, 3]:
            self.episode_stats["total_patients"] += 1

        # action effects
        if action == 6:
            self.num_open_rooms += 1
        if action == 7 and self.num_open_rooms > 0:
            self.num_open_rooms -= 1
        if action == 4:
            patient["wait_time"] += 1.0
            self.queue.append(patient)
            self.current_patient = None
        else:
            self.current_patient = None

        # queue wait penalty increment
        wait_increment = 0.01 * len(self.queue)
        for p in self.queue:
            p["wait_time"] += 1.0
        self.total_wait += wait_increment
        reward += self.wait_penalty_scale * wait_increment

        # resource cost
        reward += self.resource_cost_scale * max(0, self.num_open_rooms)

        # clip reward
        reward = float(np.clip(reward, -6.0, 6.0))
        
        # Update total reward
        self.episode_stats["total_reward"] += reward

        # spawn next patient
        self._maybe_spawn_next()

        obs = self._form_observation(self.current_patient)

        # Check termination
        if self.step_count >= self.max_steps:
            truncated = True

        info["current_severity"] = int(patient["severity"])
        info["correct_action"] = int(correct_action)
        info["num_open_rooms"] = int(self.num_open_rooms)
        info["queue_length"] = len(self.queue)
        info["triage_reward_component"] = float(reward)
        info["episode_stats"] = self.episode_stats.copy()

        # Gym API: step must return (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array") -> np.ndarray:
        H, W = 240, 360
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
        sev = self.current_patient["severity"] if self.current_patient else 0
        sev_norm = sev / 3.0
        color = np.array([int(255 * sev_norm), int(180 * (1 - sev_norm)), 60], dtype=np.uint8)
        canvas[20:200, 20:60] = color
        q_len = len(self.queue)
        q_h = min(q_len * 15, 150)
        canvas[20:20+q_h, 80:100] = [80, 80, 255]
        r = self.num_open_rooms
        r_h = min(r * 20, 150)
        canvas[20:20+r_h, 120:140] = [50, 200, 50]
        sym = self.current_patient["symptom_embed"] if self.current_patient else np.zeros(8)
        for i, val in enumerate(sym):
            x0 = 160 + i*20
            height = int(120 * val)
            canvas[150-height:150, x0:x0+12] = [int(100+150*val), int(100*(1-val)), int(50)]
        self.last_render = canvas
        return canvas

# Helper function for creating vectorized environments
def make_env(seed: int = 0, max_steps: int = 500):
    """
    Factory function for creating ClinicEnv instances.

    Args:
        seed: Random seed
        max_steps: Maximum steps per episode

    Returns:
        Callable that creates a ClinicEnv
    """
    def _init():
        env = ClinicEnv(seed=seed, max_steps=max_steps)
        return env
    return _init
