import logging
from copy import copy, deepcopy
from itertools import product
from pprint import pprint, pformat

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

from .base_planner import InteractionPlanner
import pdb
import scipy
from scipy.stats import entropy

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

# planner_type = "informed_greedy"
# planner_type = "greedy"
planner_type = "facility_location"


class NaiveGreedyPlanner(InteractionPlanner):

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

        """

        N = len(task_seq[current_task_id:])
        # some large constant
        M = sum(val for val in self.cost_cfg.values())
        pref_space = self.interaction_env.pref_space

        # Check if there's anything to be gained from queries (prefs or skills)
        options_to_gain = {}
        pref_option = 'ASK_PREF'
        current_pref_belief = pref_beliefs[current_task_id]
        potential_best_case_rewards_for_pref = []

        list_of_skills_type_pref = [(self.robot_skills[idx]['task'], self.robot_skills[idx]['pref']) for idx in range(len(self.robot_skills))]

        for pref_idx in range(len(pref_space)):
            # consider the probability of the preference
            prob_pref = current_pref_belief[pref_idx]
            # hypothesize receiving a preference query, that this prob_pref goes to 1

            # check if the robot has the skill
            prob_skill = 1 if (task_seq[current_task_id], pref_space[pref_idx]) in list_of_skills_type_pref else 0

            # candidate reward is the likelihood of needing to ask for a skill given that the pref is determined by this query
            future_skill_cost = (-self.cost_cfg["ASK_SKILL"] -self.cost_cfg['ROBOT'])* (1 - prob_skill)
            candidate_reward = -self.cost_cfg["ASK_PREF"] + future_skill_cost

            potential_best_case_rewards_for_pref.append(candidate_reward)

        # take the best case preference query option (best case reward if preference query is asked)
        best_case_reward_for_pref = max(potential_best_case_rewards_for_pref)
        options_to_gain[pref_option] = best_case_reward_for_pref
        if sum(pref_beliefs[current_task_id]) == 0:
            options_to_gain.pop(pref_option)

        # compute the best case skill gain
        skill_option = 'ASK_SKILL'
        current_pref_belief = pref_beliefs[current_task_id]
        potential_best_case_rewards_for_skill = []
        for pref_idx in range(len(pref_space)):
            # consider the probability of the preference
            prob_pref = current_pref_belief[pref_idx]
            prob_wrong_pref = 1 - prob_pref
            # check if the robot has the skill
            prob_skill = 1 if (task_seq[current_task_id], pref_space[pref_idx]) in list_of_skills_type_pref else 0
            # candidate_reward = reward if it were executed correctly - likelihood of wrong goal * pref cost
            candidate_reward = -self.cost_cfg['ASK_SKILL'] - self.cost_cfg['ROBOT'] - prob_wrong_pref * self.cost_cfg['PREF_COST']

            potential_best_case_rewards_for_skill.append(candidate_reward)
        # take the optimistic skill option (best case reward if skill query is asked)
        best_case_reward_for_skill = max(potential_best_case_rewards_for_skill)
        best_skill_pref_to_ask_for = pref_space[np.argmax(potential_best_case_rewards_for_skill)]
        options_to_gain[skill_option] = best_case_reward_for_skill

        # best case for robot
        robot_option = 'ROBOT'
        potential_best_case_rewards_for_robot = []
        for pref_idx in range(len(pref_space)):
            prob_pref = current_pref_belief[pref_idx]
            prob_correct_on_pref = prob_pref
            prob_success_on_choice = 1 if (task_seq[current_task_id], pref_space[pref_idx]) in list_of_skills_type_pref else 0

            # reward is robot cost - failure cost * prob of failure - pref cost * prob of wrong pref
            candidate_reward = -self.cost_cfg['ROBOT'] - self.cost_cfg["FAIL"] * (1 - prob_success_on_choice) - self.cost_cfg["PREF_COST"] * (1 - prob_correct_on_pref)

            potential_best_case_rewards_for_robot.append(candidate_reward)

        # take the best case robot action
        best_case_reward_for_robot = max(potential_best_case_rewards_for_robot)
        pref_choice = np.argmax(potential_best_case_rewards_for_robot)
        options_to_gain[robot_option] = best_case_reward_for_robot

        # best case for human
        options_to_gain['HUMAN'] = -self.cost_cfg['HUMAN']
        print("options_to_gain", options_to_gain)

        # choose the key in options_to_gain with highest value
        best_action = max(options_to_gain, key=options_to_gain.get)

        # update the plan
        plan = [{'action_type': best_action}]
        if best_action == 'ASK_SKILL':

            plan[0]['pref'] = best_skill_pref_to_ask_for

        if best_action == 'ROBOT':
            plan[0]['pref'] = pref_space[pref_choice]
            plan[0]['task'] = task_seq[current_task_id]
            # find skill id for the task
            for idx in range(len(self.robot_skills)):
                if (self.robot_skills[idx]['task'] == task_seq[current_task_id] and
                        self.robot_skills[idx]['pref'] == pref_space[pref_choice]):
                    plan[0]['skill_id'] = self.robot_skills[idx]['skill_id']
                    # pdb.set_trace()
                    break

        plan_info = {}


        return plan, plan_info

