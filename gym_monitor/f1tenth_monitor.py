"""Specific Rules and Predicates for the F110 Gym Env"""
import time

import numpy as np
import math

from overtaking.utility import nearest_point_on_trajectory
from .objects import Predicate, Rule, Monitor

# Default FOV and # of beams (4.7, 1080)
# Car width = 0.22
FOV = 4.7
BEAMS = 1080
WIDTH = 0.22


class ForwardCollisionZone(Predicate):
    def __init__(self, collision_threshold: float = 0.1, car_index: int = 1):
        super(ForwardCollisionZone, self).__init__()
        self.collision_threshold = collision_threshold
        self.car_index = car_index

    def evaluate(self, obs, gym) -> bool:
        """Use a bicycle model to roll forward the car for the time stated"""
        """For now, use a roll forward model"""
        rads = np.linspace((np.pi / 2) + (-FOV / 2), (np.pi / 2) + (FOV / 2), BEAMS)
        scans = obs['scans'][self.car_index]
        x_coords = scans * np.cos(rads)
        y_coords = scans * np.sin(rads)
        points = [i for i in zip(x_coords, y_coords) if -WIDTH * .9 < i[0] < WIDTH * .9]
        miny = min([i[1] for i in points])

        # TTC calc
        vel = obs['linear_vels_x'][self.car_index]
        ttc = miny / vel if vel != 0 else np.inf

        if ttc <= self.collision_threshold:
            return True
        else:
            return False


class SafetyBufferViolated(Predicate):
    def __init__(self, safety_barrier_size: float = 0.4, car_index: int = 1):
        super(SafetyBufferViolated, self).__init__()
        self.safety_barrier_size = safety_barrier_size
        self.car_index = car_index

    def evaluate(self, obs, gym) -> bool:
        """Use a bicycle model to roll forward the car for the time stated"""
        """For now, use a roll forward model"""
        rads = np.linspace((np.pi / 2) + (-FOV / 2), (np.pi / 2) + (FOV / 2), BEAMS)
        scans = obs['scans'][self.car_index]
        x_coords = scans * np.cos(rads)
        y_coords = scans * np.sin(rads)
        points = [np.sqrt(i[0]**2 + i[1]**2) for i in zip(x_coords, y_coords)]
        mindist = min([i for i in points])

        if mindist <= self.safety_barrier_size:
            return True
        else:
            return False


class SafetyBufferRule(Rule):
    """Failing Rule: Always eventually have a safety buffer"""
    def __init__(self, car_index: int, safety_barrier_size: float = 0.4,  *args, **kwargs):
        super().__init__(name='Safety Buffer Rule', *args, **kwargs)
        self.safety_predicate = SafetyBufferViolated(safety_barrier_size, car_index=car_index)

    def evaluate(self, obs, gym) -> bool:
        return True

    def end_rollout(self, obs, gym) -> bool:
        return not self.safety_predicate.evaluate(obs, gym)


class CollisionRule(Rule):
    """Rule that says never crash"""
    def __init__(self, ego_idx=0):
        super(CollisionRule, self).__init__(name='Never Crash')
        self.ego_idx = ego_idx

    def evaluate(self, obs, gym) -> bool:
        if gym.collisions[self.ego_idx]:
            return False
        else:
            return True


class LaneChangeRule(Rule):
    """Ensure lane changes only happen with adjacent lanes"""
    def __init__(self, trajectories, ego_lane_switcher):
        super(LaneChangeRule, self).__init__(name='Adjacent Lane Change')
        self.last_lane = None
        self.trajectories = trajectories
        self.ego_lane_switcher = ego_lane_switcher

    def _rotation_matrix(self, angle, direction, point=None):
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = self._unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.array(((cosa, 0.0, 0.0),
                      (0.0, cosa, 0.0),
                      (0.0, 0.0, cosa)),
                     dtype=np.float64)
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array(((0.0, -direction[2], direction[1]),
                       (direction[2], 0.0, -direction[0]),
                       (-direction[1], direction[0], 0.0)),
                      dtype=np.float64)
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M

    @staticmethod
    def _unit_vector(data, axis=None, out=None):
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data * data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data

    def _transform(self, x, y, th, oppx, oppy, oppth):
        """
        Transforms oppponent world coordinates into ego frame

        Args:
            x, y, th (float): ego pose
            oppx, oppy, oppz (float): opponent position

        Returns:
            oppx_new, oppy_new, oppth_new (float): opponent pose in ego frame
        """
        rot = self._rotation_matrix(th, (0, 0, 1))
        homo = np.array([[oppx - x], [oppy - y], [0.], [1.]])
        # inverse transform
        rotated = rot.T @ homo
        rotated = rotated / rotated[3]
        return rotated[0], rotated[1]

    def _pose2lane(self, x, y, include_raceline=True):
        """
        Returns the lane the vehicle is in

        Args:
            oppx, oppy (float): opponent position in world frame

        Returns:
            lane (string): which lane the opponent is on
        """
        nearest_dist = np.inf
        nearest_lane = -1
        for key in [item for item in self.trajectories.keys() if item not in ['race' if not include_raceline else None]]:
            lane_test = self.trajectories[key][:, :2]
            _, test_dist, _, _ = nearest_point_on_trajectory(np.array([x, y]), lane_test)
            if test_dist < nearest_dist:
                nearest_lane = key
                nearest_dist = test_dist
        return nearest_lane

    def evaluate(self, obs, gym) -> bool:
        current_lane = self.ego_lane_switcher.current_lane
        good_switch = None
        if current_lane is None:
            good_switch = True
        elif current_lane == 'race' or current_lane == 'center':
            good_switch = True  # All switches are valid
        elif current_lane == 'left':
            good_switch = True if self.last_lane in ['center', 'race'] else False
        elif current_lane == 'right':
            good_switch = True if self.last_lane in ['center', 'race'] else False
        return good_switch


class ForwardCollisionRule(Rule):
    """Ensure forward collision warning objects don't get too close"""

    def __init__(self, car_index: int, collision_threshold: float = 0.1, *args, **kwargs):
        super().__init__(name='Forward Collision', *args, **kwargs)
        self.collision_predicate = ForwardCollisionZone(collision_threshold, car_index=car_index)
        self.state = True

    def evaluate(self, obs, gym) -> bool:
        collision_status = self.collision_predicate.evaluate(obs, gym)
        self.state = self.state and not collision_status
        return self.state


class TimeBoundedForwardCollisionRule(Rule):
    """Ensure forward collision warning objects exit after a specific time"""
    def __init__(self, car_index: int, collision_threshold: float = 0.5, exit_timer: float = 0.3, *args, **kwargs):
        super().__init__(name='Time Bounded Forward Collision', *args, **kwargs)
        self.collision_predicate = ForwardCollisionZone(collision_threshold, car_index=car_index)
        self.collision_status = False
        self.last_activated = None
        self.exit_time = exit_timer

    def evaluate(self, obs, gym) -> bool:
        self.collision_status = self.collision_predicate.evaluate(obs, gym)
        if self.collision_status and self.last_activated is None:
            self.last_activated = gym.current_time
        elif not self.collision_status:
            if self.last_activated is not None:
                self.last_activated = None  # Reset the timer
        elif gym.current_time-self.last_activated > self.exit_time:
            return False  # Time expired
        return True
