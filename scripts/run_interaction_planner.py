import logging
import os
import pdb
import random
import time
from os.path import join
from pprint import pformat, pprint

import hydra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from adaptive_teaming.planner import TaskRelevantInfoGainPlanner
from adaptive_teaming.skills.pick_place_skills import PickPlaceExpert
from adaptive_teaming.utils.collect_demos import collect_demo_in_gridworld
from adaptive_teaming.utils.utils import pkl_dump, pkl_load
from hydra.utils import to_absolute_path
from PIL import Image

# from .experiments import evaluate_planner_perf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(env_name, cfg):
    print(f"Creating environment: {env_name}")
    if env_name == "gridworld":
        from adaptive_teaming.envs import GridWorld

        env = GridWorld(render_mode="human" if cfg.render else "none")

    elif env_name == "med_gridworld":
        from adaptive_teaming.envs import MediumGridWorld

        env = MediumGridWorld(render_mode="human" if cfg.render else "none")

    elif env_name == "pick_place":

        from adaptive_teaming.envs.pick_place import PickPlaceSingle

        env = PickPlaceSingle(
            has_renderer=cfg.render,
            render_camera="frontview",
        )

    elif env_name == "conveyor":

        from adaptive_teaming.envs.real_conveyor import RealConveyor

        env = RealConveyor()

    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env


def make_belief_estimator(cfg, env, task_seq):
    if cfg.env == "gridworld":
        from adaptive_teaming.planner import GridWorldBeliefEstimator

        return GridWorldBeliefEstimator(env, task_seq)

    elif cfg.env == "med_gridworld":
        from adaptive_teaming.planner import GridWorldBeliefEstimator

        return GridWorldBeliefEstimator(env, task_seq)

    elif cfg.env == "pick_place":
        from adaptive_teaming.planner import PickPlaceBeliefEstimator

        return PickPlaceBeliefEstimator(env, task_seq)

    elif cfg.env == "conveyor":
        from adaptive_teaming.planner import RealConveyorBeliefEstimator

        return RealConveyorBeliefEstimator(env, task_seq)

    else:
        raise NotImplementedError(
            f"Belief estimator not implemented for {cfg.env}")


def make_planner(interaction_env, belief_estimator, cfg):
    if cfg.planner == "fc_mip_planner":
        from adaptive_teaming.planner import FacilityLocationPlanner

        planner_cfg = cfg[cfg.planner]
        planner = FacilityLocationPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "info_gain_planner":
        from adaptive_teaming.planner import InfoGainPlanner

        planner_cfg = cfg[cfg.planner]
        planner = InfoGainPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "task_info_gain_planner":
        from adaptive_teaming.planner import TaskRelevantInfoGainPlanner

        planner_cfg = cfg[cfg.planner]
        planner = TaskRelevantInfoGainPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "fc_greedy_planner":
        from adaptive_teaming.planner import FacilityLocationGreedyPlanner

        planner_cfg = cfg[cfg.planner]
        planner = FacilityLocationGreedyPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "fc_pref_planner":
        from adaptive_teaming.planner import FacilityLocationPrefPlanner

        if cfg.fc_planner == "mip":
            from adaptive_teaming.planner import FacilityLocationPlanner

            fc_planner = FacilityLocationPlanner(
                interaction_env, belief_estimator, cfg["fc_mip_planner"], cfg.cost_cfg
            )
        elif cfg.fc_planner == "greedy":
            from adaptive_teaming.planner import FacilityLocationGreedyPlanner

            fc_planner = FacilityLocationGreedyPlanner(
                interaction_env,
                belief_estimator,
                cfg["fc_greedy_planner"],
                cfg.cost_cfg,
            )
        else:
            raise ValueError(
                f"Unknown facility location planner: {cfg.fc_planner}")

        planner_cfg = cfg[cfg.planner]
        planner = FacilityLocationPrefPlanner(
            fc_planner, interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "confidence_based_planner":
        from adaptive_teaming.planner import \
            ConfidenceBasedFacilityLocationPlanner

        planner_cfg = cfg[cfg.planner]
        planner = ConfidenceBasedFacilityLocationPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    elif cfg.planner == "always_human":
        from adaptive_teaming.planner import AlwaysHuman

        planner = AlwaysHuman(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif cfg.planner == "always_learn":
        from adaptive_teaming.planner import AlwaysLearn

        planner = AlwaysLearn(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif cfg.planner == "learn_then_robot":
        from adaptive_teaming.planner import LearnThenRobot

        planner = LearnThenRobot(
            interaction_env, belief_estimator, None, cfg.cost_cfg)
    elif cfg.planner == "fixed_planner":
        from adaptive_teaming.planner import FixedPlanner

        planner_cfg = cfg[cfg.planner]
        planner = FixedPlanner(
            interaction_env, belief_estimator, planner_cfg, cfg.cost_cfg
        )
    else:
        raise ValueError(f"Unknown planner: {cfg.planner}")

    return planner


def init_domain(cfg):
    if cfg.collect_demo:
        env = make_env(cfg.env, cfg)
        env.reset()
        if cfg.env == "gridworld":
            demo_tasks = [
                {
                    "obj_type": "Key",
                    "obj_color": "red",
                    "obj_scale": 1,
                    "position": (3, 1),
                },
                {
                    "obj_type": "Key",
                    "obj_color": "green",
                    "obj_scale": 1,
                    "position": (3, 1),
                },
            ]

        else:
            raise ValueError(f"Unknown environment: {cfg.env}")

        demos = collect_demo_in_gridworld(env, demo_tasks)
        pkl_dump(demos, f"{cfg.env}_demos.pkl")
    else:
        if not cfg.env == "conveyor":
            data_files = join(cfg.data_dir, f"{cfg.env}_demos.pkl")
            data_path = (
                os.path.dirname(os.path.realpath(__file__)) +
                "/../" + cfg.data_dir
            )

            if os.path.isdir(data_path):
                try:
                    demos = pkl_load(data_files)
                except FileNotFoundError:
                    logger.error(f"There's no data in {cfg.data_dir}")
                    exit(
                        f"\nThere's no data in {cfg.data_dir}. Terminating.\n")
            else:
                logger.error(
                    f"The path to the data ({data_path}) is erroneous.")
                exit(f"See if {cfg.data_dir} exists.")

    env = make_env(cfg.env, cfg)
    env.reset()
    if cfg.env == "gridworld":
        from adaptive_teaming.envs import GridWorldInteractionEnv

        interaction_env = GridWorldInteractionEnv(
            env, cfg.human_model, cfg.cost_cfg)

    elif cfg.env == "med_gridworld":
        from adaptive_teaming.envs import GridWorldInteractionEnv

        interaction_env = GridWorldInteractionEnv(
            env, cfg.human_model, cfg.cost_cfg)

    elif cfg.env == "pick_place":
        from adaptive_teaming.envs import PickPlaceInteractionEnv

        interaction_env = PickPlaceInteractionEnv(
            env, cfg.human_model, cfg.cost_cfg)

    elif cfg.env == "conveyor":
        from adaptive_teaming.envs import RealConveyorInteractionEnv

        interaction_env = RealConveyorInteractionEnv(
            env, cfg.human_model, cfg.cost_cfg)
    else:
        raise ValueError(f"Unknown environment: {cfg.env}")

    if not cfg.env == "conveyor":
        interaction_env.load_human_demos(demos)

    return env, interaction_env


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def randomize_gridworld_task_seq(env, cfg, n_objs=10, seed=42, gridsize=8):
    acceptable_locations = env.get_acceptable_obj_locations()
    if env.agent_start_pos in acceptable_locations:
        acceptable_locations.remove(env.agent_start_pos)

    # set the seed
    set_seeds(seed)
    np.random.seed(seed)

    # randomly assign objects to locations
    objs_list = ["Box", "Ball"]
    colors_list = ["red", "green", "blue", "yellow"]

    # create a list of n_objs objects
    objs_present = []
    for i in range(n_objs):
        objs_present.append(
            (random.choice(objs_list), random.choice(colors_list)))

    # randomly assign locations to each object type and color
    unique_objs = list(set(objs_present))
    print(f"Unique objects: {unique_objs}")

    # randomly assign locations to each object type and color
    location_assignments = np.random.choice(
        len(acceptable_locations), len(unique_objs), replace=False
    )

    # create dictionary of object type and color to location
    obj_to_location = {}
    for i, obj in enumerate(unique_objs):
        obj_to_location[obj] = acceptable_locations[location_assignments[i]]

    # create a task sequence
    task_seq = []
    for i, obj in enumerate(objs_present):
        task = {
            "obj_type": obj[0],
            "obj_color": obj[1],
            "obj_scale": 1,
            "position": obj_to_location[(obj[0], obj[1])],
        }
        task_seq.append(task)

    return [task_seq]


def generate_task_seqs(cfg, n_seqs=1, seed=42):
    # set the seed
    set_seeds(seed)
    np.random.seed(seed)

    if cfg.env == "gridworld":
        task_seq = [
            {
                "obj_type": "Key",
                "obj_color": "red",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Key",
                "obj_color": "red",
                "obj_scale": 1,
                "position": (3, 1),
            },
            {
                "obj_type": "Key",
                "obj_color": "yellow",
                "obj_scale": 1,
                "position": (3, 1),
            },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "blue",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Box",
            #     "obj_color": "red",
            #     "obj_scale": 1,
            #     "position": (3, 2),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Ball",
            #     "obj_color": "yellow",
            #     "obj_scale": 1,
            #     "position": (3, 3),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Ball",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 3),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "red",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
            # {
            #     "obj_type": "Key",
            #     "obj_color": "green",
            #     "obj_scale": 1,
            #     "position": (3, 1),
            # },
        ]
        task_seqs = [task_seq]
    elif cfg.env == "pick_place":
        if cfg.use_sim_for_task_gen:
            from adaptive_teaming.envs.pick_place import PickPlaceTaskSeqGen

            env = PickPlaceTaskSeqGen(cfg.task_seq, has_renderer=cfg.render)
            task_seqs = env.generate_task_seq(n_seqs)
        else:
            env = make_env(cfg.env, cfg)
            task_seqs = env.generate_task_seq(n_seqs, cfg.task_seq.num_tasks)

    elif cfg.env == "conveyor":
        from adaptive_teaming.envs import RealConveyorInteractionEnv

        task_seqs = RealConveyorInteractionEnv.generate_task_seqs(
            cfg.task_seq, n_seqs)

    else:
        raise ValueError(f"Unknown environment: {cfg.env}")
    return task_seqs


def vis_tasks(env, task_seq):
    vis_together = False
    if vis_together:
        from adaptive_teaming.env.gridworld import OBJECT_TYPES

        # visualize in the same env
        # env.render_mode = "rgb_array"
        for task in task_seq:
            # env.reset_to_state(task)
            obj = OBJECT_TYPES[task["obj_type"]](task["obj_color"])
            env.objects.append({"object": obj, "position": task["position"]})

        env.reset()
        for _ in range(10):
            env.render()
        __import__("ipdb").set_trace()
    else:
        for task in task_seq:
            env.reset_to_state(task)
            for _ in range(10):
                env.render()


def save_task_imgs(env, task_seq):
    render_mode = env.render_mode
    env.render_mode = "rgb_array"
    os.makedirs("tasks")
    for i, task in enumerate(task_seq):
        env.reset_to_state(task)
        img = Image.fromarray(env.render())
        img.save(f"tasks/task_{i}.png")
    env.render_mode = render_mode


def plot_interaction(objects, actions, placements, plannername):
    """
    Plots a bar chart of actions performed for each object.

    Parameters:
        objects (list of str): List of object names.
        actions (list of tuples): List of actions for each object. Each action is a tuple or string.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Define some colors for actions
    action_colors = {
        "ASK_PREF": "#c9b35b",
        "ASK_SKILL": "#9a4f69",
        "ROBOT": "#b39c85",
        "HUMAN": "#69755c",
    }

    # Define action order
    action_order = ["ASK_PREF", "ASK_SKILL", "ROBOT", "HUMAN"]

    # Create bars for each object and its corresponding actions
    for idx, (obj, action, placement) in enumerate(zip(objects, actions, placements)):
        start = idx  # Bar starts at the index
        width = 1  # Width of the bar spans one unit

        if len(action) == 2:
            # Split the bar if multiple actions are present
            half_width = width / 2
            ax.barh(
                y=action_order.index(action[0]),
                width=half_width,
                left=start,
                color=action_colors[action[0]],
                edgecolor="black",
                align="center",
            )
            ax.barh(
                y=action_order.index(action[1]),
                width=half_width,
                left=start + half_width,
                color=action_colors[action[1]],
                edgecolor="black",
                align="center",
            )
            # Add placement text for each action
            ax.text(
                start + 0.25,
                action_order.index(action[0]),
                placement[0],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )
            ax.text(
                start + 0.75,
                action_order.index(action[1]),
                placement[1],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )

        else:
            # Single action bar
            ax.barh(
                y=action_order.index(action[0]),
                width=width,
                left=start,
                color=action_colors[action[0]],
                edgecolor="black",
                align="center",
            )

            # Add placement text to the bar
            ax.text(
                start + 0.5,
                action_order.index(action[0]),
                placement[0],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                weight="bold",
            )

        # Add a dotted vertical line to indicate the start and end of each object
        ax.axvline(x=start, color="gray", linestyle="dotted", linewidth=0.8)

    # Set the x-axis to show object names
    ax.set_xticks(range(len(objects)))
    ax.set_xticklabels(objects, rotation=90)

    # Set the y-axis to show action names
    ax.set_yticks(range(len(action_order)))
    ax.set_yticklabels(action_order)

    # Set the y-axis label and x-axis label
    ax.set_ylabel("Action Name")
    ax.set_xlabel("Object Name")
    ax.set_title(f"Actions Performed on Objects: {plannername}")

    # Add legend
    # patches = [mpatches.Patch(color=color, label=action) for action, color in action_colors.items()]
    # ax.legend(handles=patches, title="Actions")

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig("interaction.png")
    plt.close()


def sample_human_pref(list_of_goals):
    n_human_prefs = 3
    # create a dictionary of all object, color combinations to random goals
    objs_list = ["key", "box", "ball"]
    colors_list = ["red", "green", "blue", "yellow"]
    all_combinations = [(obj, color)
                        for obj in objs_list for color in colors_list]

    human_pref_1 = {comb: random.choice(list_of_goals)
                    for comb in all_combinations}
    human_pref_2 = {comb: random.choice(list_of_goals)
                    for comb in all_combinations}
    human_pref_3 = {comb: random.choice(list_of_goals)
                    for comb in all_combinations}

    list_of_prefs = [human_pref_1, human_pref_2, human_pref_3]
    return random.choice(list_of_prefs)


def compare_ufl_mip_vs_ptas(cfg, num_objs_min=10, num_objs_max=101):
    """
    Compare planning times for UFL MIP and PTAS and the empirical suboptimality of PTAS.
    """
    cfg.planner = "fc_pref_planner"
    logger.info(f"Computing planning stats for {cfg.planner}")
    stats = []
    # for num_tasks in range(num_objs_min, num_objs_max, 50):
    # for _ in range(5):
    for _ in range(1):
        num_tasks = cfg.task_seq.num_tasks
        logger.info("#tasks: %d", num_tasks)
        cfg.task_seq.num_tasks = num_tasks
        env, interaction_env = init_domain(cfg)
        plan_times = []
        task_seqs = generate_task_seqs(cfg, n_seqs=5, seed=cfg.seed)
        for task_seq in task_seqs:
            belief_estimator = make_belief_estimator(cfg, env, task_seq)
            cfg.fc_planner = "greedy"
            ptas_planner = make_planner(interaction_env, belief_estimator, cfg)
            cfg.fc_planner = "mip"
            opt_planner = make_planner(interaction_env, belief_estimator, cfg)

            (
                total_rew,
                resultant_objects,
                resultant_actions,
                placements,
                rollout_info,
            ) = ptas_planner.rollout_interaction(
                task_seq,
                interaction_env.task_similarity,
                interaction_env.pref_similarity,
                opt_planner=opt_planner,
            )
            stats.extend(rollout_info["planner_vs_opt"])
            print(f"Total reward: {total_rew}")
            # plan_times.append(plan_info["solve_time"])
        # stats["planning_time"].append(plan_times)
        # stats["num_objs"].append(num_tasks)
        # logger.info(f"  Mean Planning Time: {np.mean(plan_times)}")

    pkl_dump(stats, "ufl_mip_vs_ptas.pkl")


def evaluate_planner_perf(cfg):
    """
    WIP
    Compute the total reward using the planner.
    """
    logger.info(f"Evaluating planner {cfg.planner} performance")
    num_objs_min, num_objs_max = 50, 501

    planner_stats = {"planning_time": [], "num_objs": [], "total_reward": []}

    for i, num_tasks in enumerate(range(num_objs_min, num_objs_max, 50)):
        logger.info("  #tasks: %d", num_tasks)
        cfg.task_seq.num_tasks = num_tasks
        env, interaction_env = init_domain(cfg)
        plan_times = []
        task_seqs = generate_task_seqs(cfg, n_seqs=5, seed=cfg.seed + i)
        for task_seq in task_seqs:
            belief_estimator = make_belief_estimator(cfg, env, task_seq)
            planner = make_planner(interaction_env, belief_estimator, cfg)
            (
                total_rew,
                resultant_objects,
                resultant_actions,
                placements,
                rollout_info,
            ) = planner.rollout_interaction(
                task_seq,
                interaction_env.task_similarity,
                interaction_env.pref_similarity,
            )
            print(f"Total reward: {total_rew}")
            print(f"Resultant objects: {resultant_objects}")

            # resultant_actions = [[a["action_type"] for a in actions] for actions in resultant_actions]
            print(f"Resultant actions: {resultant_actions}")
            print(f"Placements: {placements}")
            plot_interaction(
                resultant_objects, resultant_actions, placements, cfg.planner
            )
            print("total_rew", total_rew)

            stats["planning_time"].append(plan_times)
            stats["num_objs"].append(num_tasks)
            logger.info(f"  Mean Planning Time: {np.mean(plan_times)}")
        pkl_dump(stats, "ufl_planning_time.pkl")


@hydra.main(
    config_path="../cfg", config_name="run_interaction_planner", version_base="1.1"
)
def main(cfg):
    logger.info(f"Output directory: {os.getcwd()}")
    set_seeds(cfg.seed)

    if cfg.experiment == "mip_vs_ptas":
        compare_ufl_mip_vs_ptas(cfg)
        return

    env, interaction_env = init_domain(cfg)
    obs = env.reset()

    if cfg.debug.execute_skill:
        skill = PickPlaceExpert.get_skill(env, None, "Bin1")
        # env.deterministic_reset = True
        # objs = ["bread", "milk", "can", "cereal"]
        objs = ["cup"]
        for _ in range(5):
            for obj in objs:
                env._set_object_type(obj)
                obs = env.reset()
                for _ in range(10):
                    env.render()
                skill.step(env, "Bin3", obs, render=cfg.render)

    start_time = time.time()
    if cfg.env == "gridworld":
        task_seqs = randomize_gridworld_task_seq(
            env, cfg, n_objs=10, seed=cfg.seed)
    else:
        task_seqs = generate_task_seqs(
            cfg, n_seqs=cfg.task_seq.num_seqs, seed=cfg.seed)
    logger.info(f"Time to generate task seqs: {time.time() - start_time}")

    if cfg.env == "gridworld":
        true_human_pref = sample_human_pref(cfg.gridworld.list_of_goals)

    # task_seqs = task_seqs[0]

    all_rews, all_actions = [], []
    all_num_actions = {"ASK_SKILL": [],
                       "HUMAN": [], "ASK_PREF": [], "ROBOT": []}
    for task_seq in task_seqs:
        logger.info(f"{pformat(task_seq)}")

        if cfg.env == "gridworld" and cfg.vis_tasks:
            vis_tasks(env, task_seq)
            save_task_imgs(env, task_seq)
            # pdb.set_trace()

        elif cfg.env == "med_gridworld" and cfg.vis_tasks:
            vis_tasks(env, task_seq)
            save_task_imgs(env, task_seq)

        elif cfg.env == "pick_place":
            pass
            # for task in task_seq:
            # env.reset_to_state(task)
            # for _ in range(50):
            # env.render()

        interaction_env.reset(task_seq)
        if cfg.env == "gridworld":
            interaction_env.set_human_pref(true_human_pref)

        belief_estimator = make_belief_estimator(cfg, env, task_seq)
        planner = make_planner(interaction_env, belief_estimator, cfg)
        total_rew, resultant_objects, resultant_actions, placements, rollout_info = (
            planner.rollout_interaction(
                task_seq,
                interaction_env.task_similarity,
                interaction_env.pref_similarity,
            )
        )
        # resultant_actions = [[a["action_type"] for a in actions] for actions in resultant_actions]
        print(f"Resultant actions: {resultant_actions}")
        print(f"Placements: {placements}")
        plot_interaction(resultant_objects, resultant_actions,
                         placements, cfg.planner)
        print("total_rew", total_rew)

        all_rews.append(total_rew)
        # resultant_actions = np.array(resultant_actions).flatten()
        resultant_actions = [
            action for sublist in resultant_actions for action in sublist
        ]
        all_actions.append(resultant_actions)
        all_num_actions["ASK_SKILL"].append(
            sum([1 for a in resultant_actions if a == "ASK_SKILL"])
        )
        all_num_actions["HUMAN"].append(
            sum([1 for a in resultant_actions if a == "HUMAN"])
        )
        all_num_actions["ASK_PREF"].append(
            sum([1 for a in resultant_actions if a == "ASK_PREF"])
        )
        all_num_actions["ROBOT"].append(
            sum([1 for a in resultant_actions if a == "ROBOT"])
        )

        print(f"Total reward: {total_rew}")
        print(f"Resultant objects: {resultant_objects}")

    # compute experiment stats
    all_costs = -np.array(all_rews)
    logger.info(
        f"Experiment statistics averaged over {len(task_seqs)} sequences with {cfg.task_seq.num_tasks} tasks each"
    )
    logger.info(f"  Seed: {cfg.seed}")
    if cfg.planner == "fc_pref_planner":
        logger.info(f"  Planner: {cfg.planner}, MIP solver: {cfg.fc_planner}")
    else:
        logger.info(f"  Planner: {cfg.planner}")
    logger.info(f"    Cost: {np.mean(all_costs)}, +/- {np.std(all_costs)}")
    for key in ["ASK_SKILL", "HUMAN", "ASK_PREF", "ROBOT"]:
        logger.info(
            f"   {key}: {np.mean(all_num_actions[key])}, +/- {np.std(all_num_actions[key])}"
        )


if __name__ == "__main__":
    main()
