import re
from copy import deepcopy
from time import sleep

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.sim_utils import check_contact


class PickPlaceSkill:
    def __init__(self, params):
        self.params = params

    def step(self, env, pref_params, obs, render=False):
        top_offset = self.params["top_offset"]
        bottom_offset = self.params["bottom_offset"]
        target_bin_pos = self.params["target_pos"]

        # TODO: choose a plan based on the most likely goal
        total_rew = 0
        gamma, discount = 1.0, 1.0
        obj_name = env.obj_to_use
        obj_pos = obs[f"{obj_name}_pos"]
        obj = env.objects[env.object_id]
        # obj_height = obj.get_bounding_box_size()[2]
        # obj_height = obj.top_offset[2]
        bin_height = 0.05 + 0.01  # (buffer)
        bin_z = env.bin1_pos[2]
        done = False

        # self._check_safety(env, obj)

        def get_gripper_pos():
            return deepcopy(env.sim.data.site_xpos[env.robots[0].eef_site_id])

        def get_gripperr_ori():
            ori = T.quat2axisangle(
                T.mat2quat(
                    np.array(
                        env.sim.data.site_xmat[
                            env.sim.model.site_name2id(
                                env.robots[0].gripper.naming_prefix +
                                "grip_site"
                            )
                        ]
                    ).reshape(3, 3)
                )
            )
            return deepcopy(ori)

        ee_ori = get_gripperr_ori()

        # move on top of the object
        target_pos = obj_pos + np.array([0, 0, 0.15])
        for i in range(50):
            if done: break
            gripper_action = [-1]
            action = np.concatenate([target_pos, ee_ori, gripper_action])
            obs, rew, done, info = self._step(env, action, obj, render)

        obj_pos = obs[f"{obj_name}_pos"]

        # go down
        z_offset = max(0, top_offset[2] - 0.05)
        target_pos = obj_pos + np.array([0, 0, z_offset])
        for i in range(50):
            if done: break
            action = np.concatenate([target_pos, ee_ori, [-1]])
            obs, rew, done, info = self._step(env, action, obj, render)

        # pickup the object
        for _ in range(20):
            if done: break
            action = np.concatenate([target_pos, ee_ori, [1]])
            obs, rew, done, info = self._step(env, action, obj, render)

        # go up
        obj_pos = obs[f"{obj_name}_pos"]
        target_pos = get_gripper_pos()
        z_target = obj_pos[2] + abs(bottom_offset[2]) + bin_height + 0.02
        target_pos[2] = z_target
        for _ in range(50):
            if done: break
            action = np.concatenate([target_pos, ee_ori, [1]])
            obs, rew, done, info = self._step(env, action, obj, render)

# check if object is grasped
        _obj = env.objects[env.object_id]
        is_grasped = (
            int(
                env._check_grasp(
                    gripper=env.robots[0].gripper,
                    object_geoms=_obj.contact_geoms
                )
            )
        )
        if not is_grasped:
            done = True
            info["safety_violated"] = True

        # travel to the goal
        target_pos = target_bin_pos
        target_pos[2] = z_target
        for _ in range(80):
            if done: break
            action = np.concatenate([target_pos, ee_ori, [1]])
            obs, rew, done, info = self._step(env, action, obj, render)


        # go down and release the object
        # target_pos = target_bin_pos + np.array([0, 0, 0.12])
        target_pos = get_gripper_pos() + np.array(
            [0, 0, -obj.get_bounding_box_half_size()[2]]
        )
        for _ in range(50):
            if done: break
            action = np.concatenate([target_pos, ee_ori, [1]])
            obs, rew, done, info = self._step(env, action, obj, render)

        for _ in range(20):
            if done: break
            action = np.concatenate([target_pos, ee_ori, [-1]])
            obs, rew, done, info = self._step(env, action, obj, render)

        # go up
        target_pos = get_gripper_pos() + np.array([0, 0, 0.1])
        for _ in range(20):
            if done: break
            action = np.concatenate([target_pos, ee_ori, [-1]])
            obs, rew, done, info = self._step(env, action, obj, render)

        total_rew = 0
        done = True


        # XXX we are not optimizing for the reward here.
        # The true reward is the human's preference which is not available to the robot.
        return obs, total_rew, done, info

    def _step(self, env, action, obj, render):
        obs, rew, done, info = env.step(action)
        if render:
            env.render()
        if self._check_safety(env, obj):
            info["safety_violated"] = True
            done = True
        else:
            info["safety_violated"] = False
        return obs, rew, done, info

    def _check_safety(self, env, obj):
        """
        Ensures that the object is not toppled.
        """
        angle_thresh = np.pi/3
        # ensure that the object is not toppled
        obj_mat = env.sim.data.body_xmat[env.obj_body_id[obj.name]].reshape(3, 3)
        z_world = obj_mat[:, 2]  # Third column of the rotation matrix
        # World up vector
        world_up = np.array([0, 0, 1])
        cos_theta = np.dot(z_world, world_up) / (np.linalg.norm(z_world) * np.linalg.norm(world_up))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clamp for numerical stability
        # Check if the object is toppled
        return theta > angle_thresh

    def _check_collision(self, env, obj):
        """
        TODO: collision checking is not working with bins currently. There is probably a naming issue.
        """
        obj_geoms = obj.contact_geoms
        bin1_geoms = env.model.mujoco_arena.bin1_collision
        bin2_geoms = env.model.mujoco_arena.bin2_collision
        if check_contact(env.sim, bin1_geoms, obj_geoms) or check_contact(
            env.sim, obj_geoms, bin2_geoms
        ):
            return True
        else:
            return False


class PickPlaceExpert:
    @staticmethod
    def get_skill(env, task, pref_params):
        obj = env.objects[env.object_id]
        top_offset = obj.top_offset
        bottom_offset = obj.bottom_offset - 0.05
        target_bin = int(re.findall(r"\d+", pref_params)[0])
        bin_pos = env.target_bin_placements[target_bin]
        params = dict(
            top_offset=top_offset, bottom_offset=bottom_offset, target_pos=bin_pos
        )
        skill = PickPlaceSkill(params)
        return skill
