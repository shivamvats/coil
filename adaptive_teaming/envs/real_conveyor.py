import logging
import os
import re


class RealConveyor:
    def __init__(self):
        pass

    @property
    def pref_space(self):
        return [
            ("Bin0", "Grasp0"),
            ("Bin0", "Grasp1"),
            ("Bin1", "Grasp0"),
            ("Bin1", "Grasp1"),
            ("Bin2", "Grasp0"),
            ("Bin2", "Grasp1"),
        ]

    def step(self, action):
        obs, rew, done, info = {}, 0, False, {}
        return obs, rew, done, info

    def reset(self):
        pass

    def reset_to_state(self, state):
        self.state = state

    @property
    def has_renderer(self):
        return False

    def generate_task_seq(self):
        pass
