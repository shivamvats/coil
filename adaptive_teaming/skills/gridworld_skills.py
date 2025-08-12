from minigrid.core.actions import Actions
import pdb

class PickPlaceSkill:
    def __init__(self, plans):
        # one plan for each goal
        self.plans = plans

    def step(self, env, pref_params, obs):
        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
            "backspace":Actions.drop,
        }

        # TODO: choose a plan based on the most likely goal

        total_rew = 0
        gamma, discount = 1.0, 1.0
        print("plans", self.plans)
        orientation_dict =  {0:'right',
                             1:'down',
                             2:'left',
                             3:'up'} # (0: right, 1: down, 2: left, 3: up)
        # self.plans[0].append("enter")
        for action in self.plans[0]:
            print()
            print("action", action)

            if action not in key_to_action:
                break
            action = key_to_action[action]
            obs, rew, term, trunc, info = env.step(action)
            print("action: ", action)

            # TODO: check if collision happens and break wtih safety violation
            total_rew += discount * rew
            discount *= gamma
            # print("trunc", trunc)
            print('agent pos', (env.agent_pos[1], env.agent_pos[0]))
            print("env.agent_dir", orientation_dict[env.agent_dir])
            done = term # or trunc
            if done:
                break
            # print("env.carrying", env.carrying)
            if env.carrying:
                print("carrying pos: ", env.carrying)
                print("spot picked up", env.carrying.cur_pos)

        # pdb.set_trace()
        info["safety_violated"] = False
        # print("done", done)
        # print("info", info)
        # pdb.set_trace()

        # XXX we are not optimizing for the reward here.
        # The true reward is the human's preference which is not available to the robot.
        return obs, total_rew, done, info
