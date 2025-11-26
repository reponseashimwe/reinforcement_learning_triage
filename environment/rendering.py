"""
Enhanced Pygame Rendering for ClinicEnv

Provides high-quality 2D visualization for the dermatology clinic environment.
Features:
- Patient severity visualization with color coding
- Queue management display
- Room occupancy indicators
- Real-time metrics (triage accuracy, wait times)
- Action history
"""

import pygame
import numpy as np
from typing import Optional, Dict, Any
import sys


class ClinicRenderer:
    """
    Pygame-based renderer for ClinicEnv.

    Provides a professional visualization suitable for demonstrations and videos.
    """

    # Color palette
    COLORS = {
        "background": (245, 245, 250),
        "text": (30, 30, 30),
        "panel": (255, 255, 255),
        "border": (200, 200, 200),
        "severity_mild": (100, 200, 100),
        "severity_moderate": (255, 200, 50),
        "severity_severe": (255, 100, 50),
        "severity_critical": (255, 50, 50),
        "queue": (100, 150, 255),
        "room_open": (80, 200, 120),
        "room_closed": (180, 180, 180),
        "correct": (50, 200, 50),
        "incorrect": (255, 50, 50)
    }

    SEVERITY_NAMES = ["MILD", "MODERATE", "SEVERE", "CRITICAL"]
    ACTION_NAMES = [
        "Doctor", "Nurse", "Remote", "Escalate",
        "Defer", "Idle", "Open Room", "Close Room"
    ]

    def __init__(self, width: int = 800, height: int = 600, fps: int = 6):
        """
        Initialize pygame renderer.

        Args:
            width: Window width
            height: Window height
            fps: Frames per second for video recording
        """
        pygame.init()

        self.width = width
        self.height = height
        self.fps = fps

        # Create display surface
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dermatology Clinic Triage Simulation")

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # Clock for FPS control
        self.clock = pygame.time.Clock()

        # Action history for display
        self.action_history = []
        self.max_history = 5

    def render(
        self,
        env_state: Dict[str, Any],
        action: Optional[int] = None,
        reward: float = 0.0
    ) -> np.ndarray:
        """
        Render the current environment state.

        Args:
            env_state: Dictionary containing environment state
            action: Last action taken (optional)
            reward: Last reward received

        Returns:
            RGB array of the rendered frame
        """
        # Fill background
        self.screen.fill(self.COLORS["background"])

        # Draw panels
        self._draw_patient_panel(env_state)
        self._draw_queue_panel(env_state)
        self._draw_metrics_panel(env_state)
        self._draw_resources_panel(env_state)

        # Draw action history if available
        if action is not None:
            self._add_action_to_history(action, reward)
            self._draw_action_history()

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

        # Convert to RGB array
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))  # Pygame uses (width, height, channels)

        return frame

    def _draw_patient_panel(self, state: Dict[str, Any]):
        """Draw current patient information panel."""
        panel_rect = pygame.Rect(20, 20, 360, 240)

        # Draw panel background
        pygame.draw.rect(self.screen, self.COLORS["panel"], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS["border"], panel_rect, 2)

        # Title
        title = self.font_large.render("Current Patient", True, self.COLORS["text"])
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))

        # Get patient info
        severity = state.get("current_severity", 0)

        # Severity indicator
        severity_text = self.SEVERITY_NAMES[severity]
        severity_color_key = f"severity_{severity_text.lower()}"
        severity_color = self.COLORS.get(severity_color_key, self.COLORS["text"])

        # Draw severity bar
        bar_width = 320
        bar_height = 40
        bar_x = panel_rect.x + 20
        bar_y = panel_rect.y + 60

        pygame.draw.rect(
            self.screen,
            severity_color,
            pygame.Rect(bar_x, bar_y, bar_width, bar_height),
            border_radius=5
        )

        # Severity text
        sev_text = self.font_medium.render(severity_text, True, (255, 255, 255))
        text_rect = sev_text.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        self.screen.blit(sev_text, text_rect)

        # Patient features (simplified visualization)
        features_y = bar_y + bar_height + 20

        # Fever indicator
        fever = state.get("fever_flag", 0.0) > 0.5
        fever_text = "🌡️ Fever" if fever else "✓ No Fever"
        fever_color = self.COLORS["incorrect"] if fever else self.COLORS["correct"]
        self._draw_indicator(fever_text, fever_color, bar_x, features_y)

        # Infection indicator
        infection = state.get("infection_flag", 0.0) > 0.5
        infection_text = "⚠️ Infection" if infection else "✓ No Infection"
        infection_color = self.COLORS["incorrect"] if infection else self.COLORS["correct"]
        self._draw_indicator(infection_text, infection_color, bar_x, features_y + 35)

        # Recommended action
        correct_action = state.get("correct_action", 0)
        rec_text = f"Optimal: {self.ACTION_NAMES[correct_action]}"
        rec_label = self.font_small.render(rec_text, True, self.COLORS["text"])
        self.screen.blit(rec_label, (bar_x, features_y + 80))

    def _draw_indicator(self, text: str, color: tuple, x: int, y: int):
        """Draw a small status indicator."""
        indicator_rect = pygame.Rect(x, y, 300, 25)
        pygame.draw.rect(self.screen, color, indicator_rect, border_radius=3)

        ind_text = self.font_small.render(text, True, (255, 255, 255))
        self.screen.blit(ind_text, (x + 10, y + 5))

    def _draw_queue_panel(self, state: Dict[str, Any]):
        """Draw queue visualization panel."""
        panel_rect = pygame.Rect(400, 20, 380, 120)

        # Panel background
        pygame.draw.rect(self.screen, self.COLORS["panel"], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS["border"], panel_rect, 2)

        # Title
        queue_len = state.get("queue_length", 0)
        title = self.font_medium.render(f"Queue: {queue_len} patients", True, self.COLORS["text"])
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))

        # Visual queue representation (circles)
        circle_y = panel_rect.y + 60
        circle_spacing = 40
        max_display = 8

        for i in range(min(queue_len, max_display)):
            circle_x = panel_rect.x + 20 + i * circle_spacing
            pygame.draw.circle(
                self.screen,
                self.COLORS["queue"],
                (circle_x, circle_y),
                15
            )

        if queue_len > max_display:
            overflow_text = self.font_small.render(
                f"+{queue_len - max_display} more",
                True,
                self.COLORS["text"]
            )
            self.screen.blit(overflow_text, (panel_rect.x + 20 + max_display * circle_spacing, circle_y - 10))

    def _draw_resources_panel(self, state: Dict[str, Any]):
        """Draw resource management panel."""
        panel_rect = pygame.Rect(400, 160, 380, 100)

        # Panel background
        pygame.draw.rect(self.screen, self.COLORS["panel"], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS["border"], panel_rect, 2)

        # Title
        title = self.font_medium.render("Resources", True, self.COLORS["text"])
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))

        # Open rooms visualization
        num_rooms = state.get("num_open_rooms", 1)
        room_y = panel_rect.y + 50
        room_size = 30
        room_spacing = 40

        for i in range(10):  # Max 10 rooms displayed
            room_x = panel_rect.x + 20 + i * room_spacing
            color = self.COLORS["room_open"] if i < num_rooms else self.COLORS["room_closed"]

            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(room_x, room_y, room_size, room_size),
                border_radius=3
            )

    def _draw_metrics_panel(self, state: Dict[str, Any]):
        """Draw performance metrics panel."""
        panel_rect = pygame.Rect(20, 280, 760, 160)

        # Panel background
        pygame.draw.rect(self.screen, self.COLORS["panel"], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS["border"], panel_rect, 2)

        # Title
        title = self.font_large.render("Performance Metrics", True, self.COLORS["text"])
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))

        # Get statistics
        stats = state.get("episode_stats", {})
        correct = stats.get("correct_triages", 0)
        incorrect = stats.get("incorrect_triages", 0)
        total = correct + incorrect

        accuracy = (100.0 * correct / total) if total > 0 else 0.0

        # Metrics
        metrics_y = panel_rect.y + 60
        metrics_x = panel_rect.x + 20

        metrics = [
            f"Triage Accuracy: {accuracy:.1f}%",
            f"Correct: {correct}  |  Incorrect: {incorrect}",
            f"Total Patients: {stats.get('total_patients', 0)}",
            f"Total Reward: {stats.get('total_reward', 0.0):.2f}"
        ]

        for i, metric in enumerate(metrics):
            metric_text = self.font_medium.render(metric, True, self.COLORS["text"])
            self.screen.blit(metric_text, (metrics_x, metrics_y + i * 30))

    def _add_action_to_history(self, action: int, reward: float):
        """Add action to history."""
        self.action_history.append((action, reward))
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)

    def _draw_action_history(self):
        """Draw recent action history."""
        panel_rect = pygame.Rect(20, 460, 760, 120)

        # Panel background
        pygame.draw.rect(self.screen, self.COLORS["panel"], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS["border"], panel_rect, 2)

        # Title
        title = self.font_medium.render("Recent Actions", True, self.COLORS["text"])
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))

        # Draw history
        history_y = panel_rect.y + 45
        for i, (action, reward) in enumerate(self.action_history):
            action_name = self.ACTION_NAMES[action]
            reward_text = f"+{reward:.1f}" if reward >= 0 else f"{reward:.1f}"
            reward_color = self.COLORS["correct"] if reward >= 0 else self.COLORS["incorrect"]

            # Action text
            action_label = self.font_small.render(
                f"{i+1}. {action_name}",
                True,
                self.COLORS["text"]
            )
            self.screen.blit(action_label, (panel_rect.x + 20, history_y + i * 20))

            # Reward text
            reward_label = self.font_small.render(reward_text, True, reward_color)
            self.screen.blit(reward_label, (panel_rect.x + 200, history_y + i * 20))

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()


def render_episode_to_video(
    env,
    policy_func,
    filename: str,
    num_episodes: int = 1,
    max_steps: int = 500
):
    """
    Render episodes to video file using pygame rendering.

    Args:
        env: ClinicEnv instance
        policy_func: Function that takes observation and returns action
        filename: Output video filename
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
    """
    import imageio

    renderer = ClinicRenderer()
    writer = imageio.get_writer(filename, fps=6)

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # Get action from policy
            action = policy_func(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # Create state dict for renderer with UPDATED info
            state_dict = {
                "current_severity": info.get("current_severity", 0),
                "queue_length": info.get("queue_length", 0),
                "num_open_rooms": info.get("num_open_rooms", 1),
                "correct_action": info.get("correct_action", 0),
                "episode_stats": info.get("episode_stats", {}),
                "fever_flag": obs[2] if len(obs) > 2 else 0.0,
                "infection_flag": obs[3] if len(obs) > 3 else 0.0
            }

            # Render frame AFTER step with actual data
            frame = renderer.render(state_dict, action, reward)
            writer.append_data(frame)

    writer.close()
    renderer.close()

    print(f"Video saved to: {filename}")
