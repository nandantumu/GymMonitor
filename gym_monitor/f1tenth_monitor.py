"""Specific Rules and Predicates for the F110 Gym Env"""
from .objects import Predicate


class ForwardCollisionZone(Predicate):
    def __init__(self, collision_threshold: float = 0.1):
        super(ForwardCollisionZone, self).__init__()

    def evaluate(self, obs, gym) -> bool:
        """Use a bicycle model to roll forward the car for the time stated"""