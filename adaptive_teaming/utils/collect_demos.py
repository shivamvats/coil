import logging

from minigrid.manual_control import ManualControl
import pygame
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collect_demo_in_gridworld(env, tasks):
    """
    Collects a demonstration from a user.
    """
    manual_control = ManualControl(env)
    trajs = {}
    print("Use the arrow keys to move the agent and TAB to pick-up/drop objects.")
    print("Press the BACKSPACE key to record a demo." )
    for task in tqdm(tasks):
        for pref in env.pref_space:
            print(f"Provide demonstration with pref params : {pref}")
            traj = []
            done_flag = False
            env.reset_to_state(task)
            while not done_flag and not manual_control.closed:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        manual_control.env.close()
                        done_flag = True
                        break
                    if event.type == pygame.KEYDOWN:
                        event.key = pygame.key.name(int(event.key))
                        print(event.key)
                        traj.append(event.key)
                        manual_control.key_handler(event)

                        if event.key == "backspace":
                            # pygame.quit()
                            done_flag = True
                            break

            trajs[task['obj_type'] + "-" + task['obj_color'] + "-" + pref] = traj
            logger.info("Demo collected:")
            logger.info(f"{traj}")
    return trajs
