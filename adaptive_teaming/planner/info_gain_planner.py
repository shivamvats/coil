import logging
import pdb
from copy import copy, deepcopy
from itertools import product
from pprint import pformat, pprint

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from gurobipy import GRB
from scipy.stats import entropy

from .base_planner import InteractionPlanner

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

# planner_type = "informed_greedy"
# planner_type = "greedy"
planner_type = "facility_location"


class InfoGainPlanner(InteractionPlanner):

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
        task_seq: sequence of tasks each described a vector
        task_similarity_fn: function to compute similarity between tasks
        pref_similarity_fn: function to compute similarity between task preferences

        Output should be a tuple: plan, plan_info
        plan [{'action_type': 'ASK_PREF'}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}]
        plan_info {'cost': 500.0, 'assignments': {(('HUMAN', 'task-0'), 0): 1.0, (('HUMAN', 'task-1'), 1): 1.0, (('HUMAN', 'task-2'), 2): 1.0, (('HUMAN', 'task-3'), 3): 1.0, (('HUMAN', 'task-4'), 4): 1.0, (('HUMAN', 'task-5'), 5): 1.0, (('HUMAN', 'task-6'), 6): 1.0, (('HUMAN', 'task-7'), 7): 1.0, (('HUMAN', 'task-8'), 8): 1.0, (('HUMAN', 'task-9'), 9): 1.0}}

        """

        N = len(task_seq[current_task_id:])
        # some large constant
        M = sum(val for val in self.cost_cfg.values())
        pref_space = self.interaction_env.pref_space
        cost_scale = self.planner_cfg.cost_scale_factor

        # Check if there's anything to be gained from queries (prefs or skills)
        options_to_gain = {}
        pref_option = "ASK_PREF"
        current_obj_type = (
            task_seq[current_task_id]["obj_type"],
            task_seq[current_task_id]["obj_color"],
        )
        # check the gain of preferences
        candidate_info_gains = []
        for pref_idx in range(len(pref_space)):
            total_info_gain_pref_i = 0
            for task_id in range(len(task_seq)):
                next_obj_type = (
                    task_seq[task_id]["obj_type"],
                    task_seq[task_id]["obj_color"],
                )
                if pref_similarity_fn(current_obj_type, next_obj_type) < 1:
                    continue
                print("pref is the same")
                train_pref_belief = pref_beliefs[task_id]
                for g_idx in range(len(train_pref_belief)):
                    distr_gi = [train_pref_belief[g_idx],
                                1 - train_pref_belief[g_idx]]
                    print("distr_gi", distr_gi)
                    init_entropy_gi = entropy(distr_gi)
                    # compute the new belief
                    new_belief_distr_gi = deepcopy(distr_gi)
                    if pref_idx == g_idx:
                        new_belief_distr_gi[0] = 1

                        for other_pref_idx in range(len(new_belief_distr_gi)):
                            if other_pref_idx != pref_idx:
                                new_belief_distr_gi[other_pref_idx] = 0

                    # normalize new belief
                    # new_belief_distr_gi = new_belief_distr_gi / np.sum(new_belief_distr_gi)
                    print("new_belief_distr_gi", new_belief_distr_gi)
                    new_entropy_gi = entropy(new_belief_distr_gi)
                    gain = init_entropy_gi - new_entropy_gi
                    total_info_gain_pref_i += gain
            candidate_info_gains.append(total_info_gain_pref_i)
        best_case_info_gain = max(candidate_info_gains)
        print("candidate_info_gains", candidate_info_gains)
        print("best_case_info_gain", best_case_info_gain)
        options_to_gain[pref_option] = best_case_info_gain

        # check the gain of skills
        skill_option = "ASK_SKILL"
        init_capability = 0
        # print("self.robot_skills", self.robot_skills)
        list_of_skills_type_pref = [
            (self.robot_skills[idx]["task"], self.robot_skills[idx]["pref"])
            for idx in range(len(self.robot_skills))
        ]
        for task_id in range(len(task_seq)):
            # for each preference location
            for pref in pref_space:
                if (task_seq[task_id], pref) in list_of_skills_type_pref:
                    init_capability += 1

        list_case_capacity = []
        for cand_pref in pref_space:
            capability_after_skill = 0
            new_list_of_skills_type_pref = deepcopy(list_of_skills_type_pref)
            new_list_of_skills_type_pref.append(
                (task_seq[current_task_id], cand_pref))
            for task_id in range(len(task_seq)):
                # for each preference location
                for pref in pref_space:
                    if (task_seq[task_id], pref) in new_list_of_skills_type_pref:
                        capability_after_skill += 1
            list_case_capacity.append(capability_after_skill)

        print("list_case_capacity", list_case_capacity)
        best_skill_pref_to_ask_for = pref_space[np.argmax(list_case_capacity)]
        best_case_capacity = max(list_case_capacity)
        best_case_skill_gain = best_case_capacity - init_capability
        print("skill_gain", best_case_skill_gain)
        options_to_gain[skill_option] = best_case_skill_gain
        print("pre options_to_gain", options_to_gain)

        # if any keys in options_to_gain are keys, drop key
        if options_to_gain["ASK_PREF"] == 0:
            options_to_gain.pop("ASK_PREF")
        if options_to_gain["ASK_SKILL"] == 0:
            options_to_gain.pop("ASK_SKILL")

        # if the keys in options_to_gain are not all zeros, then return the learning plan

        # check the gain of robot actions
        robot_option = "ROBOT"
        pref_choice = np.argmax(pref_beliefs[current_task_id])
        prob_correct_on_pref = pref_beliefs[current_task_id][pref_choice]
        prob_success_on_choice = 0
        if (
            task_seq[current_task_id],
            pref_space[pref_choice],
        ) in list_of_skills_type_pref:
            prob_success_on_choice = 1
        failure_cost = self.cost_cfg["FAIL"] * (
            1 - prob_success_on_choice
        ) + self.cost_cfg["PREF_COST"] * (1 - prob_correct_on_pref)
        task_reward = 1 - failure_cost
        print("task_reward", task_reward)
        options_to_gain[robot_option] = (
            task_reward * cost_scale - self.cost_cfg["ROBOT"] * cost_scale
        )

        # check the gain of human actions
        human_option = "HUMAN"
        options_to_gain[human_option] = -self.cost_cfg["HUMAN"] * cost_scale

        # subtract the cost of asking for the preference
        if pref_option in options_to_gain:
            options_to_gain[pref_option] = (
                options_to_gain[pref_option] -
                self.cost_cfg["ASK_PREF"] * cost_scale
            )
        if skill_option in options_to_gain:
            options_to_gain[skill_option] = (
                options_to_gain[skill_option] -
                self.cost_cfg["ASK_SKILL"] * cost_scale
            )

        print("options_to_gain", options_to_gain)
        # choose the key in options_to_gain with highest value
        best_action = max(options_to_gain, key=options_to_gain.get)
        print("best_action", best_action)
        print("current_task_id", current_task_id)
        # pdb.set_trace()

        # update the plan
        plan = [{"action_type": best_action}]
        if best_action == "ASK_SKILL":
            plan[0]["pref"] = best_skill_pref_to_ask_for

        if best_action == "ROBOT":
            plan[0]["pref"] = pref_space[pref_choice]
            plan[0]["task"] = task_seq[current_task_id]
            # find skill id for the task
            for idx in range(len(self.robot_skills)):
                if (
                    self.robot_skills[idx]["task"] == task_seq[current_task_id]
                    and self.robot_skills[idx]["pref"] == pref_space[pref_choice]
                ):
                    plan[0]["skill_id"] = self.robot_skills[idx]["skill_id"]
                    # pdb.set_trace()
                    break

        plan_info = {}

        return plan, plan_info


class TaskRelevantInfoGainPlanner(InteractionPlanner):

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
        task_seq: sequence of tasks each described a vector
        task_similarity_fn: function to compute similarity between tasks
        pref_similarity_fn: function to compute similarity between task preferences

        Output should be a tuple: plan, plan_info
        plan [{'action_type': 'ASK_PREF'}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}, {'action_type': 'HUMAN', 'service_cost': 0}]
        plan_info {'cost': 500.0, 'assignments': {(('HUMAN', 'task-0'), 0): 1.0, (('HUMAN', 'task-1'), 1): 1.0, (('HUMAN', 'task-2'), 2): 1.0, (('HUMAN', 'task-3'), 3): 1.0, (('HUMAN', 'task-4'), 4): 1.0, (('HUMAN', 'task-5'), 5): 1.0, (('HUMAN', 'task-6'), 6): 1.0, (('HUMAN', 'task-7'), 7): 1.0, (('HUMAN', 'task-8'), 8): 1.0, (('HUMAN', 'task-9'), 9): 1.0}}

        """
        N = len(task_seq[current_task_id:])
        # some large constant
        M = sum(val for val in self.cost_cfg.values())
        pref_space = self.interaction_env.pref_space
        cost_scale = self.planner_cfg.cost_scale_factor

        # Check if there's anything to be gained from queries (prefs or skills)
        options_to_gain = {}
        pref_option = "ASK_PREF"
        # current_obj_type = (task_seq[current_task_id]["obj_type"], task_seq[current_task_id]["obj_color"])
        # check the gain of preferences
        candidate_info_gains = []
        for pref_idx in range(len(pref_space)):
            total_info_gain_pref_i = 0
            for task_id in range(len(task_seq)):
                # next_obj_type = (task_seq[task_id]["obj_type"], task_seq[task_id]["obj_color"])
                # if pref_similarity_fn(current_obj_type, next_obj_type) < 1:
                if pref_similarity_fn(task_seq[current_task_id], task_seq[task_id]) < 1:
                    continue
                print("pref is the same\n")
                train_pref_belief = pref_beliefs[task_id]
                distr_gi = [
                    [train_pref_belief[g_idx], 1 - train_pref_belief[g_idx]]
                    for g_idx in range(len(train_pref_belief))
                ]
                new_belief_distr_gi = deepcopy(distr_gi)
                init_entropy_gi = sum(entropy(distr_gi, axis=1))
                for g_idx in range(len(train_pref_belief)):
                    print("distr_gi", distr_gi)
                    # compute the new belief
                    print("pref_idx", pref_idx)
                    print("dist_gi", distr_gi)

                    if pref_idx == g_idx:
                        new_belief_distr_gi[g_idx] = [1, 0]
                    else:
                        new_belief_distr_gi[g_idx] = [0.1, 0.9]

                        # new_belief_distr_gi[0] = 1
                        # new_belief_distr_gi[1] = 0
                        # new_belief_distr_gi[0] = 0.9

                        # for other_pref_idx in range(1, len(new_belief_distr_gi)):
                        # new_belief_distr_gi[other_pref_idx] = 0
                        # new_belief_distr_gi[other_pref_idx] = 0.1

                    # __import__('ipdb').set_trace()
                    # normalize new belief
                    # new_belief_distr_gi = new_belief_distr_gi / np.sum(new_belief_distr_gi)
                print("new_belief_distr_gi", new_belief_distr_gi)
                new_entropy_gi = sum(entropy(new_belief_distr_gi, axis=1))
                gain = init_entropy_gi - new_entropy_gi
                total_info_gain_pref_i += gain
            candidate_info_gains.append(total_info_gain_pref_i)
        best_case_info_gain = max(candidate_info_gains)
        avg_case_info_gain = np.dot(
            candidate_info_gains, pref_beliefs[current_task_id])
        print("candidate_info_gains", candidate_info_gains)
        print("best_case_info_gain", best_case_info_gain)
        print("avg_case_info_gain", avg_case_info_gain)
        # __import__("ipdb").set_trace()
        # options_to_gain[pref_option] = best_case_info_gain
        options_to_gain[pref_option] = avg_case_info_gain
        if sum(pref_beliefs[current_task_id]) == 0:
            options_to_gain.pop(pref_option)

        # __import__('ipdb').set_trace()

        # check the gain of skills
        skill_option = "ASK_SKILL"
        init_capability = 0
        # print("self.robot_skills", self.robot_skills)
        list_of_skills_type_pref = [
            (self.robot_skills[idx]["task"], self.robot_skills[idx]["pref"])
            for idx in range(len(self.robot_skills))
        ]
        for task_id in range(len(task_seq)):
            # for each preference location
            for pref_idx in range(len(pref_space)):
                pref = pref_space[pref_idx]
                # if (task_seq[task_id], pref) in list_of_skills_type_pref:
                    # init_capability += 1 * (pref_beliefs[task_id][pref_idx])
                for pair in list_of_skills_type_pref:
                    if dicts_are_equal(task_seq[task_id], pair[0]) and pref == pair[1]:
                        init_capability += 1 * (pref_beliefs[task_id][pref_idx])
                        break

        list_case_capacity = []
        # cand_pref = pref_space[np.argmax(pref_beliefs[task_id][pref_idx])]
        for cand_pref in pref_space:
            capability_after_skill = 0
            new_list_of_skills_type_pref = deepcopy(list_of_skills_type_pref)
            new_list_of_skills_type_pref.append(
                (task_seq[current_task_id], cand_pref))
            for task_id in range(len(task_seq)):
                # for each preference location
                for pref_idx in range(len(pref_space)):
                    pref = pref_space[pref_idx]
                    # print(task_seq[task_id], pref)
                    # print(new_list_of_skills_type_pref)
                    # print("\n")
                    for pair in new_list_of_skills_type_pref:
                        if (
                            dicts_are_equal(task_seq[task_id], pair[0])
                            and pref == pair[1]
                        ):
                            capability_after_skill += 1 * (
                                pref_beliefs[task_id][pref_idx]
                            )
                            break
            list_case_capacity.append(capability_after_skill)

        print("list_case_capacity", list_case_capacity)
        best_skill_pref_to_ask_for = pref_space[np.argmax(list_case_capacity)]
        # best_skill_pref_to_ask_for = cand_pref
        best_case_capacity = max(list_case_capacity)
        best_case_skill_gain = best_case_capacity - init_capability
        print("skill_gain", best_case_skill_gain)
        options_to_gain[skill_option] = best_case_skill_gain
        print("pre options_to_gain", options_to_gain)
        # pdb.set_trace()

        # if any keys in options_to_gain are keys, drop key
        if "ASK_PREF" in options_to_gain and options_to_gain["ASK_PREF"] == 0:
            options_to_gain.pop("ASK_PREF")
        if options_to_gain["ASK_SKILL"] == 0:
            options_to_gain.pop("ASK_SKILL")

        # if the keys in options_to_gain are not all zeros, then return the learning plan
        # if all(value == 0 for value in options_to_gain.values()):
        # options_to_gain = {}
        # check the gain of robot actions
        robot_option = "ROBOT"
        pref_choice = np.argmax(pref_beliefs[current_task_id])
        prob_correct_on_pref = pref_beliefs[current_task_id][pref_choice]
        prob_success_on_choice = 0
        # if (
            # task_seq[current_task_id],
            # pref_space[pref_choice],
        # ) in list_of_skills_type_pref:
            # prob_success_on_choice = 1
        for pair in list_of_skills_type_pref:
            if (
                dicts_are_equal(task_seq[current_task_id], pair[0])
                and pref_space[pref_choice] == pair[1]
            ):
                prob_success_on_choice = 1
                break
        failure_cost = self.cost_cfg["FAIL"] * (
            1 - prob_success_on_choice
        ) + self.cost_cfg["PREF_COST"] * (1 - prob_correct_on_pref)
        task_reward = 1 - failure_cost
        print("task_reward", task_reward)
        options_to_gain[robot_option] = (
            task_reward * cost_scale - self.cost_cfg["ROBOT"] * cost_scale
        )

        # check the gain of human actions
        human_option = "HUMAN"
        options_to_gain[human_option] = -self.cost_cfg["HUMAN"] * cost_scale

        # else:
        # subtract the cost of asking for the preference
        if pref_option in options_to_gain:
            options_to_gain[pref_option] = (
                options_to_gain[pref_option] -
                self.cost_cfg["ASK_PREF"] * cost_scale
            )
        if skill_option in options_to_gain:
            options_to_gain[skill_option] = (
                options_to_gain[skill_option]
                - self.cost_cfg["ASK_SKILL"] * cost_scale
                - self.cost_cfg["ROBOT"] * cost_scale
            )

        print("options_to_gain", options_to_gain)
        # choose the key in options_to_gain with highest value
        best_action = max(options_to_gain, key=options_to_gain.get)
        print("best_action", best_action)
        # pdb.set_trace()
        print("current_task_id", current_task_id)
        # pdb.set_trace()

        # update the plan
        plan = [{"action_type": best_action}]
        if best_action == "ASK_SKILL":
            plan[0]["pref"] = best_skill_pref_to_ask_for

        if best_action == "ROBOT":
            plan[0]["pref"] = pref_space[pref_choice]
            plan[0]["task"] = task_seq[current_task_id]
            # find skill id for the task
            for idx in range(len(self.robot_skills)):
                if (
                    self.robot_skills[idx]["task"] == task_seq[current_task_id]
                    and self.robot_skills[idx]["pref"] == pref_space[pref_choice]
                ):
                    plan[0]["skill_id"] = self.robot_skills[idx]["skill_id"]
                    # pdb.set_trace()
                    break

        plan_info = {}

        return plan, plan_info


def dicts_are_equal(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        elif val1 != val2:
            return False
    return True


def vis_world(world_state, fig=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    tasks = world_state["tasks"]
    prefs = world_state["prefs"]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    for i, task in enumerate(tasks):
        shape = task["shape"]
        color = task["color"]
        size = task["size"]
        pos = task["pos"]
        ax.scatter(pos[0], pos[1], c=color, s=size, marker=shape)
        ax.text(pos[0], pos[1], i, fontsize=12, color="black")

    for pref_id, pref in prefs.items():
        shape = pref["shape"]
        color = pref["color"]
        size = pref["size"]
        pos = pref["pos"]
        ax.scatter(pos[0], pos[1], c=color, s=size, marker=shape)
        ax.text(pos[0], pos[1], pref_id, fontsize=12, color="black")
    # plt.show()
    return fig, ax


def vis_pref_beliefs(pref_beliefs, fig=None):
    # ax = fig.add_subplot(1, 2, 2)
    fig, axs = plt.subplots(1, len(pref_beliefs),
                            figsize=(3 * len(pref_beliefs), 3))
    # ax.set_xlim(-2, 2)
    for ax in axs:
        ax.set_ylim(0, 1)
        ax.grid(True)
    for i, beliefs in enumerate(pref_beliefs):
        # axs[i].bar(np.arange(len(beliefs)), beliefs, color='blue')
        axs[i].bar(["G1", "G2"], beliefs, color="blue")
        axs[i].set_title(f"Task-{i}")
    return fig, axs


def precond_prediction(train_task, test_task):
    if train_task["shape"] != test_task["shape"]:
        return 0

    if np.linalg.norm(np.array(train_task["pos"]) - np.array(test_task["pos"])) < 0.6:
        return 1
    else:
        return 0


def skill_library_precond(skills, task):
    return max([precond_prediction(skill["train_task"], task) for skill in skills])


if __name__ == "__main__":
    # test code
    cost_cfg = {
        "rob": 10,
        "hum": 90,
        "demo": 150,
        "pref": 20,
        "fail_cost": 100,
        "pref_cost": 50,
        # "pref_cost": 0
    }

    # tasks
    task_seq = [
        {"shape": "s", "color": "red", "size": 300, "pos": [0, 0]},
        {"shape": "s", "color": "blue", "size": 300, "pos": [0, 1]},
        {"shape": "o", "color": "red", "size": 300, "pos": [1, 0]},
        {"shape": "o", "color": "blue", "size": 300, "pos": [1, 0.5]},
        {"shape": "o", "color": "green", "size": 300, "pos": [1.5, 0.5]},
    ]

    pref_params = {
        "G1": {"shape": "s", "color": "gray", "size": 2000, "pos": [-1, 1]},
        "G2": {"shape": "s", "color": "gray", "size": 2000, "pos": [1, -1]},
    }

    # squares together and circles together
    hum_prefs = ["G1", "G1", "G2", "G2", "G2"]

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 2, 1)

    fig1, ax1 = vis_world({"tasks": task_seq, "prefs": pref_params})

    pref_beliefs = [
        np.ones(len(pref_params)) / len(pref_params) for _ in range(len(task_seq))
    ]

    fig2, ax2 = vis_pref_beliefs(pref_beliefs)

    # plt.show()

    # init with a base skill
    skills = [
        {
            "policy": None,
            "train_task": {"shape": "s", "color": "red", "size": 300, "pos": [-1, -1]},
        }
    ]

    # GREEDY Planner
    # ---------------
    if planner_type == "greedy":
        total_cost = 0
        for i in range(len(task_seq)):
            logger.debug(f"Planning for task-{i}")
            tasks = task_seq[i:]

            # options
            # hum
            # -----
            c_hum = cost_cfg["hum"]
            logger.debug(f"  Total HUM cost: {c_hum}")

            # demo
            # -----
            c_demo = cost_cfg["demo"]

            logger.debug(f"  Total DEMO cost: {c_demo}")

            # rob
            # -----
            # pref_belief = pref_beliefs[i]
            # safety cost
            prob_fail = 1 - skill_library_precond(skills, tasks[0])
            c_safe = prob_fail * cost_cfg["fail_cost"]
            c_pref = np.mean(
                [
                    (1 - prob_pref) * cost_cfg["pref_cost"]
                    for prob_pref in pref_beliefs[i]
                ]
            )
            c_rob = cost_cfg["rob"] + c_safe + c_pref

            logger.debug(
                f"  Total ROB cost: {c_rob}, c_safe: {c_safe}, c_pref: {c_pref}"
            )

            # pref
            # -----
            # TODO

            best_action = "hum"
            best_cost = c_hum
            if c_demo < best_cost:
                best_cost = c_demo
                best_action = "demo"

            if c_rob < best_cost:
                best_cost = c_rob
                best_action = "rob"

            logger.debug(
                f"  Best action: {best_action}, Best cost: {best_cost}")

            # simulate the action and get the *actual* cost
            if best_action == "hum":
                total_cost += cost_cfg["hum"]
            elif best_action == "rob":
                prob_fail = 1 - skill_library_precond(skills, tasks[0])
                c_safe = prob_fail * cost_cfg["fail_cost"]
                c_pref = np.mean(
                    [
                        (1 - prob_pref) * cost_cfg["pref_cost"]
                        for prob_pref in pref_beliefs[i]
                    ]
                )
                c_rob = cost_cfg["rob"] + c_safe + c_pref
            elif best_action == "demo":
                skills.append({"policy": None, "train_task": tasks[0]})
            else:
                raise ValueError(f"Unknown action: {best_action}")

        logger.info(f"Total cost: {total_cost}")

    elif planner_type == "informed_greedy":
        # average over all future tasks
        total_cost = 0
        for i in range(len(task_seq)):
            logger.debug(f"Planning for task-{i}")
            tasks = task_seq[i:]

            # options
            # hum
            # -----
            c_hum = cost_cfg["hum"]
            logger.debug(f"  Total HUM cost: {c_hum}")

            # demo
            # -----
            c_demo = cost_cfg["demo"]
            # reduction in future rob costs
            if len(tasks) > 1:
                curr_total_fail_cost = (
                    np.array(
                        [1 - skill_library_precond(skills, task)
                         for task in tasks[1:]]
                    )
                    * cost_cfg["fail_cost"]
                )
                pot_skills = skills + \
                    [{"policy": None, "train_task": tasks[0]}]
                pot_total_fail_cost = (
                    np.array(
                        [
                            1 - skill_library_precond(pot_skills, task)
                            for task in tasks[1:]
                        ]
                    )
                    * cost_cfg["fail_cost"]
                )
                improvement = curr_total_fail_cost - pot_total_fail_cost
                logger.debug(
                    f"  Improvement in future fail costs: {improvement}")
                improvement = np.mean(improvement)
                c_demo -= improvement

            logger.debug(f"  Total DEMO cost: {c_demo}")

            # rob
            # -----
            # pref_belief = pref_beliefs[i]
            # safety cost
            prob_fail = 1 - skill_library_precond(skills, tasks[0])
            c_safe = prob_fail * cost_cfg["fail_cost"]
            c_pref = np.mean(
                [
                    (1 - prob_pref) * cost_cfg["pref_cost"]
                    for prob_pref in pref_beliefs[i]
                ]
            )
            c_rob = cost_cfg["rob"] + c_safe + c_pref

            logger.debug(
                f"  Total ROB cost: {c_rob}, c_safe: {c_safe}, c_pref: {c_pref}"
            )

            # pref
            # -----
            # TODO

            best_action = "hum"
            best_cost = c_hum
            if c_demo < best_cost:
                best_cost = c_demo
                best_action = "demo"

            if c_rob < best_cost:
                best_cost = c_rob
                best_action = "rob"

            logger.debug(
                f"  Best action: {best_action}, Best cost: {best_cost}")

            # simulate the action and get the *actual* cost
            if best_action == "hum":
                total_cost += cost_cfg["hum"]
                # update pref belief
                # rob observes hum pref
                hum_pref = hum_prefs[i]
                # update pref beliefs
                for j, pref_belief in enumerate(pref_beliefs[i:]):
                    # belief estimator assumes shape is important
                    if tasks[j]["shape"] == tasks[0]["shape"]:
                        if hum_pref == "G1":
                            pref_belief[0] = 1
                        elif hum_pref == "G2":
                            pref_belief[1] = 1
                        else:
                            raise ValueError(f"Unknown preference: {hum_pref}")
                vis_pref_beliefs(pref_beliefs)
                plt.show()

            elif best_action == "rob":
                prob_fail = 1 - skill_library_precond(skills, tasks[0])
                c_safe = prob_fail * cost_cfg["fail_cost"]
                c_pref = np.mean(
                    [
                        (1 - prob_pref) * cost_cfg["pref_cost"]
                        for prob_pref in pref_beliefs[i]
                    ]
                )
                c_rob = cost_cfg["rob"] + c_safe + c_pref
            elif best_action == "demo":
                skills.append({"policy": None, "train_task": tasks[0]})
                # TODO update pref belief
            else:
                raise ValueError(f"Unknown action: {best_action}")

        logger.info(f"Total cost: {total_cost}")

    elif planner_type == "facility_location":
        # raise NotImplementedError

        # __import__('ipdb').set_trace()

        # preference parameters

        planner = FacilityLocationPlanner(cost_cfg)
        # task_seq = np.random.rand(10, 2)
        task_seq = np.random.normal(0, 1, (10, 2))

        def task_similarity_fn(x, y):
            return np.exp(-np.linalg.norm(x - y))

        pref_similarity_fn = task_similarity_fn
        planner.plan(task_seq, task_similarity_fn, pref_similarity_fn)
