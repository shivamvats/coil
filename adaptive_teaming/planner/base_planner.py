import logging
import pdb
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pprint import pformat

import numpy as np

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


class InteractionPlanner(ABC):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        self.interaction_env = interaction_env
        self.belief_estimator = belief_estimator
        self.planner_cfg = planner_cfg
        self.cost_cfg = cost_cfg

    @property
    def action_space(self):
        return ["ROBOT", "HUMAN", "ASK_DEMO", "ASK_PREF"]

    @abstractmethod
    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        raise NotImplementedError

    def rollout_interaction(
        self, task_seq, task_similarity_fn, pref_similarity_fn, opt_planner=None
    ):
        """
        Rollout the interaction between the human and the agent for all the tasks.

        Args:
            opt_planner: if provided, use this planner to compute plan suboptimality
        """
        resultant_objects, resultant_actions, placements = [], [], []
        self.robot_skills = []
        total_rew = 0
        self.interaction_env.reset(task_seq)
        # for task_id, task in enumerate(task_seq):
        executed_task_ids, executed_actions, actual_rews = [], [], []
        rollout_info = {"planner_vs_opt": []}
        executed_beliefs = []
        done = False
        task_id = 0
        while not done:
            task = task_seq[task_id]
            executed_task_ids.append(task_id)
            logger.debug(f"Executing task {task_id}")
            logger.debug(f"  Task: {pformat(task)}")
            pref_beliefs = self.belief_estimator.beliefs

            plan, plan_info = self.plan(
                task_seq,  # [task_id:],
                pref_beliefs,
                task_similarity_fn,
                pref_similarity_fn,
                task_id,
            )
            if opt_planner is not None:
                opt_planner.robot_skills = self.robot_skills
                opt_plan, opt_plan_info = opt_planner.plan(
                    task_seq,  # [task_id:],
                    pref_beliefs,
                    task_similarity_fn,
                    pref_similarity_fn,
                    task_id,
                )
                logger.debug(
                    f"  OPT Precomputation time: {opt_plan_info['precomputation_time']}"
                )
                logger.debug(
                    f"  OPT Facility location solve time: {opt_plan_info['solve_time']}"
                )
                print(f"OPT solve time: {opt_plan_info['solve_time']}")
                rollout_info["planner_vs_opt"].append(
                    {
                        "num_tasks": len(task_seq) - task_id,
                        "plan_cost": plan_info["cost"],
                        "opt_plan_cost": opt_plan_info["cost"],
                        "plan_time": plan_info["solve_time"],
                        "opt_plan_time": opt_plan_info["solve_time"],
                    }
                )

                # plan_cost = sum([a["cost"] for a in plan])
                # opt_plan_cost = sum([a["cost"] for a in opt_plan])
                # plan_info["suboptimality"] = plan_cost / opt_plan_cost

            if "precomputation_time" in plan_info:
                logger.debug(
                    f"  Precomputation time: {plan_info['precomputation_time']}")
                logger.debug(
                    f"  Facility location solve time: {plan_info['solve_time']}")
                print(f"Solver solve time: {plan_info['solve_time']}")
            action = plan[0]
            executed_actions.append(action)
            obs, rew, done, info = self.interaction_env.step(action)

            executed_beliefs.append(
                deepcopy(self.belief_estimator.beliefs[task_id]))

            if action["action_type"] == "ASK_SKILL":
                if info["teach_success"]:
                    self.robot_skills.append(
                        {
                            "task": task,
                            "skill_id": task_id,
                            "pref": info["pref"],
                            "skill": info["skill"],
                        }
                    )
                else:
                    pass
                self.belief_estimator.update_beliefs(task_id, info)

            elif action["action_type"] == "ASK_PREF":
                self.belief_estimator.update_beliefs(task_id, info)
                # logger.debug(
                    # f"  Updated beliefs: {pformat(self.belief_estimator.beliefs)}"
                # )

            elif action["action_type"] == "ROBOT":
                # Execute previously learned skill
                # __import__('ipdb').set_trace()
                pass

            elif action["action_type"] == "HUMAN":
                pass

            else:
                raise NotImplementedError

            actual_rews.append(rew)
            total_rew += rew

            logger.debug(f"  Executing action_type: {action['action_type']}")
            logger.debug(f"    reward: {rew}")

            logger.debug("Actions executed so far:")
            logger.debug("------------------------")
            for i, (action, rew) in enumerate(zip(executed_actions, actual_rews)):
                executed_task = task_seq[executed_task_ids[i]]
                logger.debug(
                    f"  Round {i}: Task {executed_task_ids[i]} {self.get_task_desc(executed_task)} \n \t\t\t\t\t\t\t {action}, belief: {executed_beliefs[i]}, {rew}\n"
                )

            if info["current_task_id"] == task_id:
                resultant_objects.append((task_id))
                resultant_actions.append([action["action_type"]])
                placements.append([info["pref"]])
            else:
                if len(resultant_objects) > 1 and resultant_objects[-1] == task_id:
                    resultant_actions[-1].append(action["action_type"])
                    if action["action_type"] == "HUMAN":
                        placements[-1].append(info["pref"])
                    else:
                        placements[-1].append(action["pref"])
                else:
                    resultant_objects.append((task_id))
                    resultant_actions.append([action["action_type"]])
                    if action["action_type"] == "HUMAN":
                        placements.append([info["pref"]])
                    else:
                        placements.append([action["pref"]])

            task_id = info["current_task_id"]

        logger.info("Interaction statistics:")
        logger.info("-----------------------")
        logger.info(f"  Total reward: {total_rew}")
        logger.info("  Actions executed:")
        logger.info(
            f"   #teach: {sum([1 for a in executed_actions if a['action_type'] == 'ASK_SKILL'])}"
        )
        logger.info(
            f"   #human: {sum([1 for a in executed_actions if a['action_type'] == 'HUMAN'])}"
        )
        logger.info(
            f"   #pref: {sum([1 for a in executed_actions if a['action_type'] == 'ASK_PREF'])}"
        )
        logger.info(
            f"   #robot: {sum([1 for a in executed_actions if a['action_type'] == 'ROBOT'])}"
        )

        resultant_objects = [
            (
                task_seq[i]["obj_color"] + " " + task_seq[i]["obj_type"]
                if "obj_color" in task_seq[i]
                else self.get_task_desc(task_seq[i])
            )
            for i in resultant_objects
        ]
        return total_rew, resultant_objects, resultant_actions, placements, rollout_info

    @staticmethod
    def get_task_desc(task):
        task_desc = task["obj_type"]
        if "obj_color" in task:
            task_desc += "-" + task["obj_color"]
        return task_desc

    def compute_planning_time(self, task_seq, task_similarity_fn, pref_similarity_fn):
        self.robot_skills = []
        self.interaction_env.reset(task_seq)
        task_id = 0
        pref_beliefs = self.belief_estimator.beliefs  # [task_id:]
        plan, plan_info = self.plan(
            task_seq,  # [task_id:],
            pref_beliefs,
            task_similarity_fn,
            pref_similarity_fn,
            task_id,
        )
        return plan_info


class AlwaysHuman(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        """
        Plan
        """
        return [{"action_type": "HUMAN"}] * len(task_seq), {}


class AlwaysLearn(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        """
        Plan
        """
        return [{"action_type": "ASK_SKILL", "pref": "G1"}] * len(task_seq), {}


class LearnThenRobot(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)
        self.iter = 0

    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        """
        Plan
        """

        if self.iter == 0:
            plan = [{"action_type": "ASK_SKILL", "pref": "G1"}]
        else:
            plan = [{"action_type": "ROBOT", "skill_id": 0, "pref": "G1"}]
        plan += [{"action_type": "ROBOT", "skill_id": 0, "pref": "G1"}] * (
            len(task_seq) - 1
        )
        self.iter += 1
        return plan, {}


class FixedPlanner(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)
        self._plan = planner_cfg["plan"]
        self.iter = 0

    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        Plan
        """

        plan = []
        for action_type in self._plan[self.iter:]:
            if action_type == "ROBOT":
                plan.append({"action_type": action_type,
                            "skill_id": 0, "pref": "G1"})
            else:
                plan.append({"action_type": action_type})
        self.iter += 1
        return plan, {}
