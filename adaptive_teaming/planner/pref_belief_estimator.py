import logging
from abc import abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class PrefBeliefEstimator:
    def __init__(self, env, task_seq, teach_adaptive=True):
        self.env = env
        self.task_seq = task_seq
        self.teach_adaptive = teach_adaptive

        # indexed by task id
        self.beliefs = [self.prior() for _ in range(len(task_seq))]

        # Beta random variable for the teach success for each task
        # (Fail, Success)
        self.teach_rvs = np.array([[0, 1] for _ in range(len(task_seq))])

    @abstractmethod
    def prior(self):
        raise NotImplementedError

    def update_beliefs(self, task_id, obs):
        """
        Update belief
        """
        pass

    @abstractmethod
    def task_similarity_fn(self, task1, task2):
        """
        Task similarity function
        """
        raise NotImplementedError

    def update_teach_prob(self, task_id, obs):
        """
        Update belief about the success of the teach.
        """
        teach_success = int(obs["teach_success"])
        for task, teach_rv in zip(self.task_seq[task_id:], self.teach_rvs[task_id:]):
            task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            if task_sim:
                teach_rv[teach_success] += 1


    def get_teach_probs(self):
        return self.teach_rvs[:, 1] / np.sum(self.teach_rvs, axis=1)


class GridWorldBeliefEstimator(PrefBeliefEstimator):
    def __init__(self, env, task_seq, teach_adaptive=True):
        super().__init__(env, task_seq, teach_adaptive)

    def prior(self):
        """
        Prior belief

        This is a discrete probability distribution over the preference parameter spaces.
        TODO: figure out how to represent the preference
        """
        return np.ones(len(self.env.pref_space)) / len(self.env.pref_space)

    def update_beliefs(self, task_id, obs):
        """
        Update belief
        """
        # immediately converges to 1 after a preference query
        pref = obs["pref"]
        pref_idx = self._pref_to_idx(pref)
        self.beliefs[task_id][pref_idx] = 1
        # print("self.beliefs[task_id]:", self.beliefs[task_id])
        for other_pref_idx in range(len(self.beliefs[task_id])):
            if other_pref_idx != pref_idx:
                self.beliefs[task_id][other_pref_idx] = 0

        for task, belief in zip(
            self.task_seq[task_id + 1:], self.beliefs[task_id + 1:]
        ):
            task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            if task_sim:
                belief[pref_idx] = 1

                for other_pref_idx in range(len(belief)):
                    if other_pref_idx != pref_idx:
                        belief[other_pref_idx] = 0

            # bayesian update of the belief
            # XXX fix this
            # belief[pref_idx] += task_sim

            # belief /= belief.sum()

        if self.teach_adaptive and "teach_success" in obs:
            self.update_teach_prob(task_id, obs)

    def _pref_to_idx(self, pref):
        return list(self.env.pref_space).index(pref)

    def task_similarity_fn(self, task1, task2):
        """
        Task similarity function
        """
        if task1["obj_type"] == task2["obj_type"]:
            if task1["obj_color"] == task2["obj_color"]:
                return 1
            else:
                return 0
        else:
            return 0


class PickPlaceBeliefEstimator(PrefBeliefEstimator):
    def __init__(self, env, task_seq, teach_adaptive=True):
        super().__init__(env, task_seq, teach_adaptive)

    def prior(self):
        """
        Prior belief

        This is a discrete probability distribution over the preference parameter spaces.
        TODO: figure out how to represent the preference
        """
        return np.ones(len(self.env.pref_space)) / len(self.env.pref_space)

    def update_beliefs(self, task_id, obs):
        """
        Update belief
        """
        # immediately converges to 1 after a preference query
        pref = obs["pref"]
        pref_idx = self._pref_to_idx(pref)
        # self.beliefs[task_id][pref_idx] = 1
        for task, belief in zip(
            self.task_seq[task_id:], self.beliefs[task_id:]
        ):
            task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            if task_sim:
                for other_pref_idx in range(len(belief)):
                    if other_pref_idx != pref_idx:
                        belief[other_pref_idx] = 0.1
                    else:
                        belief[other_pref_idx] = 1

            # bayesian update of the belief
            # XXX fix this
            # belief[pref_idx] += task_sim

            # belief /= belief.sum()

        if self.teach_adaptive and "teach_success" in obs:
            self.update_teach_prob(task_id, obs)

    def _pref_to_idx(self, pref):
        return list(self.env.pref_space).index(pref)

    def task_similarity_fn(self, task1, task2):
        """
        Task similarity function
        """
        if task1["obj_type"] == task2["obj_type"]:
            return 1
        else:
            return 0


class RealConveyorBeliefEstimator(PrefBeliefEstimator):
    def __init__(self, env, task_seq, teach_adaptive=True):
        super().__init__(env, task_seq, teach_adaptive)

    def prior(self):
        """
        Prior belief

        This is a discrete probability distribution over the preference parameter spaces.
        TODO: figure out how to represent the preference
        """
        return np.ones(len(self.env.pref_space)) / len(self.env.pref_space)

    def update_beliefs(self, task_id, obs):
        """
        Update belief
        """
        # immediately converges to 1 after a preference query
        pref = obs["pref"]
        pref_idx = self._pref_to_idx(pref)
        pref_space = self.env.pref_space
        bin, grasp = pref
        self.beliefs[task_id][pref_idx] = 1
        # handle bin preference and grasp perference separately
        # for i, other_pref in enumerate(pref_space):
            # if other_pref[0] != bin:
                # self.beliefs[task_id][i] = 0.1

        for task, belief in zip(
            self.task_seq[task_id + 1:], self.beliefs[task_id + 1:]
        ):
            # task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            if self.task_seq[task_id]["obj_type"] == task["obj_type"]:
                if self.task_seq[task_id]["obj_name"] == task["obj_name"]:
                    belief[pref_idx] = 1
                else:
                    # same bin but grasp can be different
                    belief[self._pref_to_idx((bin, "Grasp0"))] = 0.5
                    belief[self._pref_to_idx((bin, "Grasp1"))] = 0.5

        if self.teach_adaptive and "teach_success" in obs:
            self.update_teach_prob(task_id, obs)

    def _pref_to_idx(self, pref):
        return list(self.env.pref_space).index(pref)

    def task_similarity_fn(self, task1, task2):
        """
        Task similarity function
        """
        human_reward_based_on = "obj_type"
        # human_reward_based_on = "obj_color"

        if task1[human_reward_based_on] == task2[human_reward_based_on]:
            return 1
        else:
            return 0
