from .base_planner import InteractionPlanner


class ConfidenceBasedPlanner(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        Plan
        """
        return [{"action_type": "HUMAN"}] * len(task_seq)
