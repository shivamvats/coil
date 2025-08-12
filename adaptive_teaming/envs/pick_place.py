import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import mimicgen
from mimicgen.configs.config import MG_Config
from mimicgen.env_interfaces.base import MG_EnvInterface
from mimicgen.env_interfaces.robosuite import RobosuiteInterface
from mimicgen.models.robosuite.objects import (BlenderObject, DrawerObject,
                                               LongDrawerObject)
from mimicgen.models.robosuite.objects.composite_body.cup import CupObject
from robosuite.controllers import load_controller_config
from robosuite.controllers.interpolators.linear_interpolator import \
    LinearInterpolator
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.objects import (BottleObject, BreadObject, CanObject,
                                      CerealObject, LemonObject, MilkObject)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import (SequentialCompositeSampler,
                                                UniformRandomSampler)

logger = logging.getLogger(__name__)


class PickPlaceSingle(PickPlace):
    """
    Slightly easier task where we limit z-rotation to 0 to 0 degrees for all object initializations (instead of full 360).

    Each task contains only one object which can be changed with every reset.

    Supported objects: ["bread", "milk", "cereal", "can"]
    TODO Add more objects (mugs)
    """

    def __init__(self, **kwargs):
        # initial state placeholder
        obj_type = kwargs.pop("obj_type", "bread")
        self.deterministic_reset = kwargs.pop("deterministic_reset", True)
        self._state = {
            "obj_qpos": [0.1, -0.3, 0.9, 1, 0, 0, 0],
            "obj_type": obj_type,
        }

        ctrl_cfg = load_controller_config(default_controller="OSC_POSE")
        ctrl_cfg["control_delta"] = False
        ctrl_cfg["kp"] = 25
        # ctrl_cfg["interpolator_pos"] = interpolator
        ctrl_cfg["interpolation"] = "linear"
        # ctrl_cfg["input_max"] = 0.05
        super().__init__(
            single_object_mode=2,
            object_type=obj_type,
            z_rotation=(0.0, 0),
            # z_rotation=(0.0, np.pi / 2.0),
            robots="Panda",
            # robots="Sawyer",
            # has_offscreen_renderer=True,
            # ignore_done=True,
            # use_camera_obs=False,
            # control_freq=20,
            controller_configs=ctrl_cfg,
            **kwargs,
        )

    @property
    def pref_space(self):
        # TODO also consider goal orientation
        return ["Bin0", "Bin1", "Bin2", "Bin3"]

    def reset_to_state(self, state):
        self._set_object_type(state["obj_type"])
        # set obj position
        self._state = state
        return self.reset()

    def _reset_internal(self):
        """
        Sets the object position
        """
        # self.deterministic_reset = True
        super()._reset_internal()
        obj = self.objects[self.object_id]
        if self.deterministic_reset:
            # __import__('ipdb').set_trace()
            self.sim.data.set_joint_qpos(
                obj.joints[0], self._state["obj_qpos"])
        else:
            pass
        self.sim.forward()
        # XXX could use them to show ground truth preferences
        # Set the visual object body locations
        # if "visual" in obj.name.lower():
        # self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
        # self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
        # else:
        # Set the collision object joints

    def _set_object_type(self, object_type):
        assert (
            object_type in self.object_to_id.keys()
        ), "invalid @object_type argument - choose one of {}".format(
            list(self.object_to_id.keys())
        )
        # use for convenient indexing
        self.object_id = self.object_to_id[object_type]

    def _check_obj_in_bin(self, bin_id):
        """
        Check if all objects have been successfully placed in their corresponding bins.

        Returns:
            bool: True if all objects are placed correctly
        """
        # remember objects that are in the correct bins
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_in_bins[i] = int(
                (not self.not_in_bin(obj_pos, bin_id)) and r_reach < 0.6
            )

        # returns True if a single object is in the correct bin
        if self.single_object_mode in {1, 2}:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.objects)

    def _construct_objects(self):
        self.objects = []
        obj_types = ["Milk", "Bread", "Cereal",
                     "Can", "Bottle", "Lemon", "Cup",] # "Mug"]
        obj_classes = [
            MilkObject,
            BreadObject,
            CerealObject,
            CanObject,
            BottleObject,
            LemonObject,
            CupObject,
            # MugObject,
        ]
        for obj_cls, obj_name in zip(obj_classes, obj_types):
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)

    def generate_task_seq(self, n_seqs, num_tasks):
        """
        Generate a sequence of tasks.
        """
        # sample objects
        obj_types = [
            "Milk",
            "Bread",
            "Cereal",
            "Can",
            "Bottle",
            "Lemon",
            "Cup",
        ]  # , "Mug"]

        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.05
        z_offset = 0

        dirichlet_prior = np.ones(len(obj_types)) * 2

        task_seqs = []
        for seq_id in range(n_seqs):
            objects = []
            obj_freq = np.random.dirichlet(dirichlet_prior)
            logger.info(f"Sampling tasks with object frequencies: {obj_freq}")
            # TODO sample in more interesting and adversarial ways
            indices = np.random.choice(
                range(len(obj_types)), p=obj_freq, size=num_tasks)
            obj_counts = {obj_index: 0 for obj_index in range(len(obj_types))}
            for index in indices:
                obj_name = obj_types[index] + str(obj_counts[index])
                obj_counts[index] += 1
                objects.append(obj_name)

            task_seq = []
            # resample the placement initializer
            # obs = self.reset()
            obj_X = np.random.uniform(
                low=-bin_x_half, high=bin_x_half, size=num_tasks)
            obj_Y = np.random.uniform(
                low=-bin_y_half, high=bin_y_half, size=num_tasks)
            obj_Z = np.ones(num_tasks) * (self.z_offset + self.bin1_pos[2])
            obj_posx = np.stack([obj_X, obj_Y, obj_Z], axis=1)
            for i, obj in enumerate(objects):
                obj_type = re.findall(r"\D+", obj)[0].lower()
                obj_pos = obj_posx[i]
                obj_quat = np.array([1, 0, 0, 0])
                obj_qpos = np.concatenate([obj_pos, obj_quat])
                task = dict(obj_type=obj_type, obj_qpos=obj_qpos)
                task_seq.append(task)
            fig, axs = plt.subplots(figsize=(15, 5))
            axs.scatter(obj_X, obj_Y)
            fig.savefig(f"task_seq_{seq_id}.png")
            task_seqs.append(task_seq)
        return task_seqs


class PickPlaceTaskSeqGen(PickPlace):
    """
    Generates a sequence of tasks for the PickPlace environment.
    """

    def __init__(self, task_seq_cfg, **kwargs):

        self.task_seq_cfg = task_seq_cfg
        super().__init__(
            single_object_mode=0,
            z_rotation=(0.0, np.pi / 2.0),
            robots="Panda",
            has_offscreen_renderer=True,
            ignore_done=True,
            control_freq=20,
            controller_configs=load_controller_config(
                default_controller="OSC_POSE"),
            hard_reset=True,
            use_object_obs=False,  # to avoid dealing with observables
            use_camera_obs=True,  # for saving images
            camera_names=["frontview", "birdview", "agentview"],
            camera_heights=1024,
            camera_widths=1024,
            **kwargs,
        )

    def generate_task_seq(self, n_seqs):
        """
        Generate a sequence of tasks.
        """
        task_seqs = []
        for seq_id in range(n_seqs):
            task_seq = []
            # resample the placement initializer
            obs = self.reset()
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i, camera_name in enumerate(self.camera_names):
                axs[i].imshow(obs[camera_name + "_image"][::-1])
                axs[i].axis("off")
            fig.savefig(f"task_seq_{seq_id}.png")
            # save each object's position and orientation sequentially
            for obj in self.objects:
                obj_type = re.findall(r"\D+", obj.name)[0].lower()
                obj_qpos = self.sim.data.get_joint_qpos(obj.joints[0])
                task = dict(obj_type=obj_type, obj_qpos=obj_qpos)
                task_seq.append(task)
            task_seqs.append(task_seq)
        return task_seqs

    def _construct_objects(self):
        self.objects = []
        num_tasks = self.task_seq_cfg["num_tasks"]
        self.objects = []
        obj_types = [
            "Milk",
            "Bread",
            "Cereal",
            "Can",
            "Bottle",
            "Lemon",
            "Cup",
        ]  # , "Mug"]

        obj_classes = [
            MilkObject,
            BreadObject,
            CerealObject,
            CanObject,
            BottleObject,
            LemonObject,
            CupObject,
            # MugObject
        ]
        # obj_types = ["Mug"]
        # obj_classes = [MugObject]
        # sample the frequency of each object type
        dirichlet_prior = np.ones(len(obj_classes)) * 2
        # dirichlet_prior[:] = 0.01
        # dirichlet_prior[4] = 1
        # dirichlet_prior /= dirichlet_prior.sum()

        obj_freq = np.random.dirichlet(dirichlet_prior)
        logger.info(f"Sampling tasks with object frequencies: {obj_freq}")
        # TODO sample in more interesting and adversarial ways
        indices = np.random.choice(
            range(len(obj_classes)), p=obj_freq, size=num_tasks)
        obj_counts = {obj_index: 0 for obj_index in range(len(obj_classes))}
        for index in indices:
            obj_cls = obj_classes[index]
            obj_name = obj_types[index] + str(obj_counts[index])
            obj_counts[index] += 1
            obj = obj_cls(name=obj_name)
            self.objects.append(obj)


# def generate_pick_place_task_seq(env, cfg):
# placement_initializer = SequentialCompositeSampler(
# name="ObjectSampler")

# # can sample anywhere in bin
# bin_x_half = env.model.mujoco_arena.table_full_size[0] / 2 - 0.05
# bin_y_half = env.model.mujoco_arena.table_full_size[1] / 2 - 0.05

# # each object should just be sampled in the bounds of the bin (with some tolerance)
# self.placement_initializer.append_sampler(
# sampler=UniformRandomSampler(
# name="CollisionObjectSampler",
# mujoco_objects=self.objects,
# x_range=[-bin_x_half, bin_x_half],
# y_range=[-bin_y_half, bin_y_half],
# rotation=self.z_rotation,
# rotation_axis="z",
# ensure_object_boundary_in_range=True,
# # ensure_valid_placement=True,
# ensure_valid_placement=False,
# reference_pos=self.bin1_pos,
# z_offset=self.z_offset,
# )
# )


class MugObject(BlenderObject):

    def __init__(self, name):
        base_mjcf_path = os.path.join(
            mimicgen.__path__[0], "models/robosuite/assets/shapenet_core/mugs"
        )
        self.mjcf_path = os.path.join(
            base_mjcf_path, "{}/model.xml".format("3143a4ac"))

        super().__init__(
            name=name,
            mjcf_path=self.mjcf_path,
            scale=0.8,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            # friction=(0.95, 0.3, 0.1),
            friction=(1, 1, 1),
            margin=0.001,
        )


class MG_PickPlaceSingle(RobosuiteInterface):
    """
    Corresponds to robosuite PickPlace task and variants.
    """

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # one relevant objects - milk, bread, cereal, can
        object_poses = dict()
        obj = self.env.objects[self.env.object_id]
        object_poses[obj.obj_name] = self.get_object_pose(
            obj_name=obj.root_body, obj_type="body"
        )
        return object_poses

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # checks which objects are in their correct bins and records them in @self.objects_in_bins
        self.env._check_success()

        object_names_in_order = [self.env._state["obj_type"]]
        # assert set(self.env.object_to_id.keys()) == set(object_names_in_order)
        n_obj = len(object_names_in_order)

        # each subtask is a grasp and then a place
        for i, obj_name in enumerate(object_names_in_order):
            obj_id = self.env.object_to_id[obj_name]

            # first subtask for each object is grasping (motion relative to the object)
            signals["grasp_{}".format(obj_name)] = int(
                self.env._check_grasp(
                    gripper=self.env.robots[0].gripper,
                    object_geoms=[
                        g for g in self.env.objects[obj_id].contact_geoms],
                )
            )

            # skip final subtask - unneeded
            if i < (n_obj - 1):
                # second subtask for each object is placement into bin (motion relative to bin)
                signals["place_{}".format(obj_name)] = int(
                    self.env.objects_in_bins[obj_id]
                )

        return signals
