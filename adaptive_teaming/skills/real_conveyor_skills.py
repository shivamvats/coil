import logging
import re

logger = logging.getLogger(__name__)


class RealConveyorSkill:
    def __init__(self, params):
        self.params = params

    def step(self, env, pref_params, obs, render=False):
        print("Execute skill with pref_params: ", pref_params)
        done = False
        while not done:
            inp = input("Did skill execute safely? (y/n): ")
            if inp.lower() == "q":
                break
            if inp not in ["y", "n"]:
                print("Please provide an integer.")
            else:
                safety_violated = int(inp == "n")
                done = True

        info = {}
        info["safety_violated"] = safety_violated
        return None, 0, done, info


class RealConveyorExpert:
    @staticmethod
    def get_skill(env, task, pref_params):
        logger.info(f"Get skill for {task} with params {pref_params}")
        bin, grasp = pref_params
        target_bin = int(re.findall(r"\d+", bin)[0])
        skill = RealConveyorSkill(pref_params)
        return skill
