"""Contains Basic Objects for GymMonitoring"""
from gym import Env
from typing import List

class Predicate:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, obs, gym) -> bool:
        raise NotImplementedError


class Rule:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def evaluate(self, obs, gym) -> bool:
        raise NotImplementedError

    def end_rollout(self, obs, gym) -> bool:
        return True


class Monitor:
    """High Level monitor of a gym env"""
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def evaluate_rules(self, obs, gym) -> bool:
        """Can use either the obs or the env itself to evaluate the rules"""
        evals = [rule.evaluate(obs, gym) for rule in self.rules]
        for i, evaluation in enumerate(evals):
            if not evaluation:
                print("Rule {} has failed.".format(self.rules[i].name))
        return all(evals)

    def evaluate_end_rollout(self, obs, gym) -> bool:
        evals = [rule.end_rollout(obs, gym) for rule in self.rules]
        for i, evaluation in enumerate(evals):
            if not evaluation:
                print("Rule {} has failed.".format(self.rules[i].name))
        return all(evals)
