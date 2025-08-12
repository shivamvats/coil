import logging

import numpy as np
import pandas as pd

from .interaction_env import InteractionEnv
from adaptive_teaming.skills.real_conveyor_skills import RealConveyorExpert

logger = logging.getLogger(__name__)


class RealConveyorInteractionEnv(InteractionEnv):
    def __init__(self, env, human_model_cfg, cost_cfg):
        super().__init__(env, human_model_cfg, cost_cfg)
        self.objects = self._init_objects()

    def robot_step(self, *args, **kwargs):
        return super().robot_step(*args, **kwargs)

    def human_step(self, *args, **kwargs):
        # human_pref = self._get_human_pref_for_object(
            # kwargs["task"])
        return super().human_step(None, task=kwargs["task"])

    def query_skill(self, task, pref):
        """
        If the robot fails to learn the skill after querying the human, then
        the human completes the task.
        """
        obs, rew, done, info = super().query_skill(task, pref)
        # skill is saved to dict on the real robot
        skill= RealConveyorExpert.get_skill(self.env, task, pref)
        info.update({"skill": skill, "pref": pref})

        # robot executes skill
        # rew -= self.cost_cfg["ROBOT"]

        # check if teaching was successful and set teach_success in info
        _done = False
        while not _done:
            inp = input("Was teaching successful? (y/n): ")
            if inp.lower() == "q":
                break

            if inp not in ["y", "n"]:
                print("Please choose y or n")
            else:
                teach_success = int(inp == "y")
                _done = True

        info["teach_success"] = teach_success
        if teach_success:
            logger.info("    Teaching succeeded")
        else:
            logger.info("    Teaching failed")
            # rew -= self.cost_cfg["FAIL"]


        return None, rew, True, info

    def query_pref(self, task):
        """
        Query the human for their preference for task.
        """
        print("Querying human for preference.")
        obs, rew, done, info= super().query_pref(task)

        pref_sample= self._get_human_pref_for_object(task)
        info["pref"]= pref_sample
        return None, rew, True, info

    def _get_human_pref_for_object(self, obj):
        if obj["obj_type"] == "kitchen":
            bin = 0
        elif obj["obj_type"] == "office":
            bin = 1
        elif obj["obj_type"] == "toys":
            bin = 2

        done = False
        while not done:
            try:
                grasp = int(input(f"Provide a preferred grasp for the object {obj}: "))
                done= True
            except ValueError:
                print("Please provide an integer.")

        return (f"Bin{bin}", f"Grasp{grasp}")

    def human_evaluation_of_robot(self, terminal_state=None, traj=None):
        """
        Evaluate the robot's performance. Note that the robot does not have
        access to the evaluation during the interaction but only at the end.
        """
        pref_success= input(
            "Did the robot satisfy user preference? (y/n): ")
        return int(pref_success == "y")

    @ staticmethod
    def task_similarity(task1, task2):
        if task1["obj_name"] == task2["obj_name"]:
            return 1
        else:
            return 0

    @ staticmethod
    def pref_similarity(pref1, pref2):
        return int(pref1 == pref2)

    def load_human_demos(self, demos):
        pass

    @ staticmethod
    def _init_objects():
        # Define the objects
        objects= []  # fruits
        for obj, color in zip(
            [
                "strawberry",
                "banana",
                "lemon",
                "spam",
                "can",
                "mug",
                "mug",
            ],
            [
                "red",
                "yellow",
                "yellow",
                "multi",
                "multi",
                "red",
                "white",
            ],
        ):
            objects.append(
                {"obj_name": obj, "obj_type": "kitchen", "obj_color": color})

        for obj, color in zip(
            ["shoe", "block", "block"],
            ["green", "orange", "blue"],
        ):
            objects.append(
                {"obj_name": obj, "obj_type": "toys", "obj_color": color})

        for obj, color in zip(
            [
                "bottle",
                "bottle",
                "tape",
                "tape",
            ],
            ["green", "pink", "black", "brown"],
        ):
            objects.append(
                {"obj_name": obj, "obj_type": "office", "obj_color": color})

        return objects

    @staticmethod
    def generate_task_seqs(cfg, n_seqs=1):
        objects = RealConveyorInteractionEnv._init_objects()
        task_seqs = []
        high_freq = 5
        low_freq = 1

        freqs = []

        if cfg.task_set == 0:
            # 1 high frequency obj
            freqs = low_freq * np.ones(len(objects))
            high_freq_id = np.random.randint(len(objects))
            logger.info(f"High freq object: {objects[high_freq_id]['obj_name']}")
            freqs[high_freq_id] = high_freq
            freqs /= np.sum(freqs)
            print(f"Frequencies: {freqs}")
            tasks = np.random.choice(objects, p=freqs, size=cfg.num_tasks)
            task_seqs.append(tasks)

        elif cfg.task_set == 1:
            # 2 high frequency objs
            for obj in objects:
                if obj["obj_name"] == "banana" or obj["obj_name"] == "bottle":
                    freqs.append(high_freq)
                else:
                    freqs.append(low_freq)
            freqs /= np.sum(freqs)
            print(f"Frequencies: {freqs}")
            tasks = np.random.choice(objects, p=freqs, size=cfg.num_tasks)
            task_seqs.append(tasks)

        else:
            raise NotImplementedError

        return task_seqs
