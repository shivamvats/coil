import logging
import pdb
import time
from collections import OrderedDict
from copy import copy, deepcopy
from itertools import product
from pprint import pformat, pprint

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

from .base_planner import InteractionPlanner

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

# planner_type = "informed_greedy"
# planner_type = "greedy"
planner_type = "facility_location"


class FacilityLocationPlanner(InteractionPlanner):

    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    # @profile
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
        """
        start_time = time.time()
        pref_space = self.interaction_env.pref_space
        teach_probs = self.belief_estimator.get_teach_probs()
        ROB = self.cost_cfg["ROBOT"]
        HUM = self.cost_cfg["HUMAN"]
        ASK_SKILL = self.cost_cfg["ASK_SKILL"]
        SKILL_FAIL = self.cost_cfg["FAIL"]
        PREF_FAIL = self.cost_cfg["PREF_COST"]

        def Pi_transfer(task1, task2, pref1, pref2):
            """
            Estimates the probability of transferring the skill from task1 to
            task2 given the preference param with which the skill on task1 was
            trained on.
            """
            # task_similarity_fn(task, test_task)
            # pref_similarity_fn(mle_train_pref, 'G1')
            # TODO for already learned skills, simulate the prob
            task_sim = task_similarity_fn(task1, task2)
            pref_sim = pref_similarity_fn(pref1, pref2)
            return task_sim * pref_sim

        demands = task_seq[current_task_id:]

        facilities = []
        setup_costs = {}
        service_costs = {}
        best_prefs = {}  # for ROBOT and ASK_SKILL actions
        best_skills = {}  # for ROBOT actions
        # create facilities
        # gurobi wants all tuples to be of the same length
        # setup_costs:(type-of-query, train_task_id, demo_task_id)
        # service_costs:(type-of-query, train_task_id, demo_task_id, pref_task_id)
        for task_id, task in enumerate(
            task_seq[current_task_id:], start=current_task_id
        ):
            train_pref_belief = pref_beliefs[task_id]
            mle_train_pref = pref_space[np.argmax(train_pref_belief)]
            # hum
            facility = ("HUMAN", f"task-{task_id}")
            facilities.append(facility)
            setup_costs[facility] = HUM

            # always succeeds
            # Note: I could either set a high cost for all the other tasks or I
            # "think" I can just ignore the other combinations by not creating
            # the corresponding decision variables.
            service_costs[(facility, task_id)] = 0

            # demo
            facility = ("ASK_SKILL", f"task-{task_id}")
            facilities.append(facility)
            setup_costs[facility] = ASK_SKILL

            # demo affects only current and future tasks.
            # cost is a function of Pr(teaching success)
            if self.planner_cfg.teach_adaptive:
                teach_prob = teach_probs[task_id]
            else:
                teach_prob = 1

            teach_fail_prob = 1 - teach_prob

            # task_id
            best_prefs[(facility, task_id)] = mle_train_pref
            service_costs[(facility, task_id)] = teach_fail_prob * HUM + teach_prob * (
                ROB + (1 - max(np.array(train_pref_belief))) * PREF_FAIL
            )

            for test_task_id in range(task_id + 1, len(task_seq)):
                test_task = task_seq[test_task_id]
                # XXX robot specifically asks for a skill with a pref params
                service_costs[(facility, test_task_id)] = ROB
                pref_belief = pref_beliefs[test_task_id]
                # mle_test_pref = pref_space[np.argmax(pref_belief)]
                # argmin over pref params
                pi_transfers = [
                    teach_prob *
                    Pi_transfer(task, test_task, mle_train_pref, pref)
                    for pref in pref_space
                ]
                execution_costs = (1 - np.array(pi_transfers)) * SKILL_FAIL
                pref_costs = (1 - np.array(pref_belief)) * PREF_FAIL

                # if logger.isEnabledFor(logging.DEBUG):
                # logger.debug(f" Belief pref costs: {pref_costs}")

                # save the best pref for use during execution
                # pdb.set_trace()
                best_prefs[(facility, test_task_id)] = pref_space[
                    np.argmin(execution_costs + pref_costs)
                ]
                service_costs[(facility, test_task_id)] += min(
                    execution_costs + pref_costs
                )

            # TODO execute the best previously learned skill in sim
            facility = ("ROBOT", "skill library")
            test_task = task
            # TODO simulate
            best_execution_cost = np.inf
            best_skill, best_pref = None, None
            logger.debug("  Computing best skill for ROBOT facility")
            # print("self, robot skills", self.robot_skills)
            for skill in self.robot_skills:

                for test_pref_id, test_pref in enumerate(pref_space):
                    train_task, train_pref = skill["task"], skill["pref"]
                    execution_cost = (
                        1 - Pi_transfer(train_task, test_task,
                                        train_pref, test_pref)
                    ) * SKILL_FAIL
                    # logger.debug(
                    # f"  train_pref, test_pref: {train_pref}, {test_pref}")
                    pref_belief = pref_beliefs[task_id]
                    pref_cost = (1 - pref_belief[test_pref_id]) * PREF_FAIL
                    if execution_cost + pref_cost < best_execution_cost:
                        best_execution_cost = execution_cost + pref_cost
                        best_skill, best_pref = skill, test_pref
                # save the best pref for use during execution
                best_prefs[(facility, task_id)] = best_pref
                best_skills[(facility, task_id)] = best_skill
            if np.isinf(best_execution_cost):
                logger.debug(
                    "  Skipping ROBOT facility as it has infinite cost")
            else:
                if facility not in facilities:
                    facilities.append(facility)
                    setup_costs[facility] = 0  # already learned
                logger.debug(f"  Adding ROBOT facility  for task {task_id}")
                service_costs[(facility, task_id)] = ROB + best_execution_cost

        precomputation_time = time.time() - start_time

        self.setup_costs, self.service_costs = self._dict_to_arrays(
            demands, facilities, setup_costs, service_costs, current_task_id
        )
        # N tasks = demands
        start_time = time.time()
        solver_info = self._solve_facility_location(
            demands, facilities, setup_costs, service_costs, current_task_id
        )
        fl_solve_time = time.time() - start_time
        assignments = solver_info["assignments"]
        # print("assignments", assignments)
        # construct a plan based on facility location assingnments
        plan = [None] * len(task_seq)
        for key in assignments:
            facility, demand = key[0], int(key[1])
            facility_type, train_task = facility
            plan[demand] = {
                "action_type": facility_type,
                "service_cost": service_costs[key],
            }
            if facility_type == "ASK_SKILL":
                train_task = int(train_task.split("-")[-1])
                if demand != train_task:
                    plan[demand]["skill_id"] = train_task
                    plan[demand]["pref"] = best_prefs[key]
                else:
                    # reqeust skill with specific pref param
                    # pdb.set_trace()
                    # print("key", key)
                    # print("demand", demand)
                    # print("len(keys)", best_prefs.keys())
                    # if key not in best_prefs:
                    #     __import__("ipdb").set_trace()
                    plan[demand]["pref"] = best_prefs[key]
            elif facility_type == "ROBOT":
                try:
                    plan[demand]["skill_id"] = best_skills[key]["skill_id"]
                    plan[demand]["pref"] = best_prefs[key]
                except KeyError:
                    __import__("ipdb").set_trace()
        plan = plan[current_task_id:]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Plan: {pformat(plan)}")
            logger.debug(f"  Plan cost: {solver_info['cost']}")
        plan_info = solver_info
        plan_info["solve_time"] = fl_solve_time
        plan_info["precomputation_time"] = precomputation_time
        # __import__('ipdb').set_trace()
        return plan, plan_info

    def _dict_to_arrays(
        self,
        demands,
        facilities,
        setup_costs_dict,
        service_costs_dict,
        demand_start_index,
    ):
        setup_costs = np.zeros(len(facilities))
        for fac_id, fac in enumerate(facilities):
            setup_costs[fac_id] = setup_costs_dict[fac]

        service_costs = np.zeros(
            (len(facilities), demand_start_index + len(demands)))
        for fac_id, fac in enumerate(facilities):
            for demand_id, demand in enumerate(demands):
                service_costs[fac_id, demand_start_index + demand_id] = (
                    service_costs_dict.get(
                        (fac, demand_start_index + demand_id), np.inf
                    )
                )
        return setup_costs, service_costs

    def _solve_facility_location(
        self, demands, facilities, setup_costs, service_costs, demand_start_index
    ):
        """
        This function uses the Gurobi dictionary API which associates a tuple with each Gurobi variable. The
        variables can then be accessed using the tuple as a key. This is useful
        for managing a large number of variables.

        Main classes: tuplelist and tuple dict
        """
        # cartesian_prod = list(product(range(num_demands),
        # range(num_facilities)))
        # service_costs = {(d, f): service_costs[d, f] for d, f in cartesian_prod}
        start_time = time.time()

        model = gp.Model("facility_location")
        model.setParam("OutputFlag", 0)

        select = model.addVars(facilities, vtype=GRB.BINARY, name="Select")
        assign = model.addVars(service_costs, ub=1,
                               vtype=GRB.CONTINUOUS, name="Assign")

        for key in service_costs:
            f, c = key[0], key[1]
            model.addConstr(assign[key] <= select[f], name="Setup2ship")

        for i, _ in enumerate(demands, start=demand_start_index):
            for f in facilities:
                model.addConstr(
                    gp.quicksum(assign[(f, i)]
                                for f in facilities if (f, i) in assign)
                    == 1,
                    name=f"Demand-{i}",
                )

        model.update()

        model.setObjective(
            select.prod(setup_costs) + assign.prod(service_costs), GRB.MINIMIZE
        )

        model.optimize()

        solver_info = {}

        # print("MIP Time taken for optimization: ", time.time() - start_time)
        if model.status == GRB.OPTIMAL:
            selected_facilities = {
                f: select[f].X for f in facilities if select[f].X > 0.5
            }
            assignments = {
                key: assign[key].X for key in service_costs if assign[key].X > 0.5
            }

            total_cost = model.objVal
            logger.debug("OPTIMAl facility location plan found.")

            solver_info["cost"] = total_cost
            solver_info["assignments"] = assignments
            return solver_info
        else:
            logger.warning("No optimal solution found.")
            __import__("ipdb").set_trace()
            return None


class FacilityLocationGreedyPlanner(FacilityLocationPlanner):
    """
    Implements a greedy version of the facility location planner.
    """

    use_improvement1 = False
    use_improvement2 = True

    def _solve_facility_location(
        self, demands, facilities, setup_costs, service_costs, demand_start_index
    ):
        # future demands
        S = set(range(demand_start_index, demand_start_index + len(demands)))
        # algorithm will update this
        working_setup_costs = deepcopy(setup_costs)
        best_service_costs = {demand: np.inf for demand in S}
        X = []
        while not len(S) == 0:
            # choose best facility and subset of S
            best_ratio = np.inf
            best_fac, best_assignment = None, None
            for fac in facilities:
                logger.debug(f"  Facility: {fac}")
                fac_setup_cost = working_setup_costs[fac]
                # find subset of client with best ratio
                # get service costs
                fac_service_keys = [
                    (fac, demand) for demand in S if (fac, demand) in service_costs
                ]
                fac_service_costs = [service_costs[key]
                                     for key in fac_service_keys]
                # sort the clients in terms of service cost from i
                sorted_indices = np.argsort(fac_service_costs)

                # find optimal prefix
                service_cost_sum = 0

                for i, index in enumerate(sorted_indices):
                    service_cost_sum += fac_service_costs[index]
                    improvement = fac_setup_cost + service_cost_sum
                    # add improvement in the cost of previously opened facility
                    cost_decrement = 0
                    if self.use_improvement1:
                        for d in range(
                            demand_start_index, demand_start_index +
                                len(demands)
                        ):
                            # already assigned
                            if d not in S and (fac, d) in service_costs:
                                # best_assigned_cost = np.min(
                                # [
                                # service_costs[(f, d)]
                                # for f in X
                                # if (f, d) in service_costs
                                # ]
                                # )
                                best_assigned_cost = best_service_costs[d]
                                cost_decrement += max(
                                    0, best_assigned_cost -
                                    service_costs[(fac, d)]
                                )
                                if cost_decrement > 0:
                                    if logger.isEnabledFor(logging.DEBUG):
                                        logger.debug(
                                            f"    Facility {fac} decrs cost of opened facs by {cost_decrement}"
                                        )
                                    logger.debug(
                                        f"    Facility {fac} decrs cost of opened facs by {cost_decrement}"
                                    )
                    ratio = (improvement - cost_decrement) / (i + 1)
                    # logger.debug(f"    Ratio: {ratio}")
                    if ratio < best_ratio:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"      Better ratio found! {best_ratio} -> {ratio}"
                            )
                        best_ratio = ratio
                        best_fac = fac
                        best_assignment = [
                            fac_service_keys[j] for j in sorted_indices[: i + 1]
                        ]

            logger.debug(f"Best facility: {best_fac}")
            assigned_demands = [demand for (_, demand) in best_assignment]
            for d in assigned_demands:
                if service_costs[(best_fac, d)] < best_service_costs[d]:
                    best_service_costs[d] = service_costs[(best_fac, d)]
            # __import__('ipdb').set_trace()
            # open facility
            X.append(best_fac)
            if self.use_improvement2:
                # make setup cost 0 since it is already open
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  Settings setup cost to 0 for {best_fac}")
                working_setup_costs[best_fac] = 0
            # remove assigned demands from S
            S.difference_update(assigned_demands)
            logger.debug(f"  Remaining demands: {S}")

        # compute best assignment for the selected facilities
        assignments = {}
        total_cost = 0
        # setup costs
        for fac in X:
            total_cost += setup_costs[fac]

        # compute assignment and service costs
        for demand in range(demand_start_index, demand_start_index + len(demands)):
            demand_service_costs = [
                service_costs.get((fac, demand), np.inf) for fac in X
            ]
            # pick best facility
            best_fac = X[np.argmin(demand_service_costs)]

            assignments[(best_fac, demand)] = 1
            total_cost += service_costs[(best_fac, demand)]

        solver_info = {"cost": total_cost}
        solver_info["assignments"] = assignments
        return solver_info


class FacilityLocationPrefPlanner(InteractionPlanner):
    """
    Uses the facility location planner to decide among learn, human and robot.
    Decides whether or not to ask for preference by computing expected
    improvement in objective due to belief update.
    """

    def __init__(
        self, fc_planner, interaction_env, belief_estimator, planner_cfg, cost_cfg
    ):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)
        self.fc_planner = fc_planner

    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        start_time = time.time()
        self.fc_planner.robot_skills = self.robot_skills
        plan1, plan_info1 = self.fc_planner.plan(
            task_seq,
            pref_beliefs,
            task_similarity_fn,
            pref_similarity_fn,
            current_task_id,
        )
        logger.debug(f"Time taken for plan1: {time.time() - start_time}")
        # compute plan with updated belief
        pref_belief = pref_beliefs[current_task_id]
        logger.debug(f"  Pref belief: {pref_beliefs[current_task_id:]}")

        possible_costs = []
        for possible_pref_response in self.interaction_env.pref_space:
            # update belief
            belief_estimator = copy(self.belief_estimator)
            belief_estimator.beliefs = deepcopy(pref_beliefs)
            belief_estimator.update_beliefs(
                current_task_id, {"pref": possible_pref_response}
            )
            new_pref_beliefs = belief_estimator.beliefs
            # print("new_pref_beliefs", new_pref_beliefs)
            logger.debug(
                f"  New pref beliefs: {new_pref_beliefs[current_task_id:]}")
            # __import__('ipdb').set_trace()
            self.fc_planner.robot_skills = self.robot_skills
            plan2, plan_info2 = self.fc_planner.plan(
                task_seq,
                new_pref_beliefs,
                task_similarity_fn,
                pref_similarity_fn,
                current_task_id,
            )
            possible_costs.append(plan_info2["cost"])
            logger.debug(
                f"  Considering possible preference response: {possible_pref_response}"
            )
            logger.debug(f"  Plan 2 cost: {plan_info2['cost']}")
        # assume robot asks for preference
        if sum(pref_belief) == 0:
            expected_possible_cost = np.dot(
                possible_costs, np.ones(len(possible_costs))
            )
        else:
            expected_possible_cost = np.dot(possible_costs, pref_belief) / np.sum(
                pref_belief
            )
        logger.debug(
            f"    Expected cost after pref: {expected_possible_cost} vs {plan_info1['cost']} now "
        )
        # if expected_possible_cost + self.cost_cfg["ASK_PREF"] <= plan_info1["cost"]:
        if expected_possible_cost + self.cost_cfg["ASK_PREF"] < plan_info1["cost"]:
            logger.debug("  Asking for preference")
            # __import__('ipdb').set_trace()
            plan = plan1
            plan[0] = {"action_type": "ASK_PREF"}
            plan_info = plan_info1
            plan_info["cost"] = expected_possible_cost + \
                self.cost_cfg["ASK_PREF"]
            return plan, plan_info
        else:
            logger.debug("  Not asking for preference")
            return plan1, plan_info1


class ConfidenceBasedFacilityLocationPlanner(FacilityLocationPlanner):
    """
    Uses the facility location planner to decide among learn, human and robot.
    Decides whether or not to ask for preference based on the confidence in the
    preference belief.
    """

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
        plan, plan_info = super().plan(
            task_seq,
            pref_beliefs,
            task_similarity_fn,
            pref_similarity_fn,
            current_task_id,
        )
        # return plan1, plan_info1
        # compute plan with updated belief
        pref_belief = pref_beliefs[current_task_id]
        logger.debug(f"  Pref belief: {pref_beliefs[current_task_id:]}")
        if (
            sum(pref_belief) > 0
            and np.max(pref_belief) < self.planner_cfg["confidence_threshold"]
        ):
            logger.debug("  Asking for preference")
            plan[0] = {"action_type": "ASK_PREF"}
        return plan, plan_info


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
