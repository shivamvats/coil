import copy
from collections import OrderedDict
from itertools import product
from random import sample
from typing import Dict, List, Union

import gymnasium as gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from adaptive_teaming.utils.object import Fork, Mug
import minigrid
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Goal, Key, Lava, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from typing import Any, Iterable, SupportsFloat, TypeVar
from .objects import ADIKey, Human

class PortableBox(Box):
    def can_pickup(self):
        return True  # Allow the box to be picked up

OBJECT_TYPES = {
    "Key": Key,
    "Ball": Ball,
    "Box": Box,
}

class GridWorld(MiniGridEnv):
    """A simple 2D grid world.
    The agent has to pick up an object and place it in one of the goal locations.

    The user has a hidden preference for the goal location which the agent has to maximize.
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=None,
        goal_cfg=None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        if goal_cfg is None:
            self.goals = OrderedDict(
                G1=lambda x, y: (1, y - 2),
                G2=lambda x, y: (x - 2, y - 2),
            )
        else:
            self.goals = goal_cfg
        self.goal_cfg = goal_cfg
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            # max_steps = 2 * size**2
            max_steps = 10000

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            highlight=False,
            **kwargs,
        )
        self.objects = []
        self.state = None
        # self.device = (
        # torch.device(0) if torch.cuda.device_count() else torch.device("cpu")
        # )

    @property
    def pref_space(self):
        """Set of possible preference parameters. The human's preference is a
        pmf over this space."""
        return list(self.goals.keys())

    @property
    def has_renderer(self):
        return self.render_mode == "human"

    def reset_to_state(self, state):
        """Reset the environment to a specific state."""

        obj = OBJECT_TYPES[state["obj_type"]](state["obj_color"])
        self.state = state
        self.objects = [{"object": obj, "position": state["position"]}]
        obs = super().reset()
        return obs

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            # if fwd_cell is None or fwd_cell.can_overlap():
            if fwd_cell is None or fwd_cell.type != 'wall':
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            # if fwd_cell is not None and fwd_cell.type == "lava":
            #     terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    # def step(
    # self, action: ActType
    # ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

    # obs, reward, terminated, truncated, info = super().step(action)

    # if terminated:
    # if not env.carrying:
    # terminated = False
    # return obs, reward, terminated, truncated, info

    def _place_objects(self, objects):
        for i, obj in enumerate(objects):
            position = obj["position"]
            # self.grid.set(position[0], position[1], obj["object"])
            self.put_obj(obj["object"], position[0], position[1])
            # else:
            # self.grid.set(3, 1 + i, obj)

        # Place the objects
        # self.grid.set(3, 1, ADIKey(COLOR_NAMES[0], scale=1.0))
        # self.grid.set(3, 2, Ball(COLOR_NAMES[1]))
        # self.grid.set(3, 3, Box(COLOR_NAMES[3]))

    def get_goals(self):
        goals = {}
        for goal_name, goal in self.goals.items():
            goals[goal_name] = goal(self.width, self.height)
        return goals

    @staticmethod
    def _gen_mission():
        return "mission"

    # MiniGridEnv._gen_grid
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # sample door location
        # door_y = random.randint(1, height - 2)
        door_y = height // 2

        # Generate vertical separation wall
        lava_x = width // 2
        for i in range(1, height - 1):
            if i not in [door_y]:
                self.grid.set(lava_x, i, Wall())

        # Place the objects
        self._place_objects(self.objects)
        # self.grid.set(3, 1, ADIKey(COLOR_NAMES[0], scale=1.0))
        # self.grid.set(3, 2, Ball(COLOR_NAMES[1]))
        # self.grid.set(3, 3, Box(COLOR_NAMES[3]))

        # Place a goal square in the bottom-right corner
        for goal_name, goal in self.goals.items():
            self.put_obj(Goal(), *goal(width, height))
            # self.put_obj(Box('blue'), *goal(width, height))

        # self.put_obj(Human(), 2, 2)

        # Place the agent
        if self.agent_start_pos is not None:
            if self.agent_start_pos == "random":
                self.agent_pos = (
                    np.random.randint(1, lava_x - 1),
                    np.random.randint(1, height - 2),
                )
            else:
                self.agent_pos = tuple(self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # self.mission = self._gen_mission()

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        # topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        # agent_view_size = agent_view_size or self.agent_view_size

        # grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)
        # grid = self.grid.slice(topX, topY, self.width, self.height)

        # for i in range(self.agent_dir + 1):
        # grid = grid.rotate_left()

        grid = self.grid

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = self.agent_pos
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def get_acceptable_obj_locations(self):
        """
        give a list of acceptable spawn position of objects
        """

        grid_width = self.grid.width
        grid_height = self.grid.height
        easy_grid = np.zeros((grid_width, grid_height))
        type_to_encoding = {minigrid.core.world_object.Wall: 1,
                            type(None): 0,
                            minigrid.core.world_object.Goal: 2,
                            minigrid.core.world_object.Box: 3,
                            minigrid.core.world_object.Key: 3,
                            minigrid.core.world_object.Ball: 3}  # 3 is for objects
        acceptable_locations = []
        for i in range(grid_width):
            for j in range(grid_height):
                cell_type = type(self.grid.get(i, j))
                # print(cell_type)
                # pdb.set_trace()
                easy_grid[j, i] = type_to_encoding[cell_type]
                if type_to_encoding[cell_type] == 0:
                    acceptable_locations.append((j, i))
        return acceptable_locations
