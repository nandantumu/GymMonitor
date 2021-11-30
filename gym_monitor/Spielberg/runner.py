from overtaking.controllers import LaneSwitcherController, PurePursuitController, \
    ITTCAvoider, CollisionAvoider, RandomSwitcher
from overtaking.planners import TripleRacelineLoader, RacelinePlanner
from overtaking.utility import ZoneDefiner, get_start_position
from gym_monitor import Monitor
from gym_monitor.f1tenth_monitor import TimeBoundedForwardCollisionRule, LaneChangeRule, \
    CollisionRule, SafetyBufferRule, ForwardCollisionRule
import time
import yaml
import gym
from argparse import Namespace
import numpy as np

if __name__ == '__main__':
    opponent_cripple = .75
    with open('config.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    COLL_AVOID_LOOKAHEAD = 0.15
    COLL_AVOID_SLOWDOWN = 0.1

    planner = TripleRacelineLoader('spielberg_lanes.csv')
    zd = ZoneDefiner(planner.center, points_per_zone=20)
    ego_switcher = LaneSwitcherController(0, planner.plan('dict'), lookahead=1.3, overtake_lookahead=2.0)
    #ego_switcher = RandomSwitcher(0, planner.plan('dict'), lookahead=1.3, overtake_lookahead=2, switch_time=2)
    ego_collision_avoider = ITTCAvoider(0, ego_switcher, COLL_AVOID_SLOWDOWN, COLL_AVOID_LOOKAHEAD)

    opp_planner = RacelinePlanner('spielberg_lanes.csv', 9, 10, 11)
    OPP_NUMBER = 1
    OPP_OFFSET = 20
    #START_OFFSET = 21*20
    START_OFFSET = 0

    opp_controllers = [PurePursuitController(i+1, opp_planner.plan()) for i in range(OPP_NUMBER)]
    opp_collision_avoiders = [ITTCAvoider(i+1, opp_controllers[i], COLL_AVOID_SLOWDOWN, COLL_AVOID_LOOKAHEAD) for i in range(OPP_NUMBER)]

    collision_rule = CollisionRule()
    ego_lane_change_rule = LaneChangeRule(planner.plan('dict'), ego_switcher)
    ego_safety_buffer_rule = SafetyBufferRule(0, safety_barrier_size=2)
    forward_collision_rule = ForwardCollisionRule(1, collision_threshold=0.05)
    ego_forward_collision_rule = ForwardCollisionRule(0, collision_threshold=0.05)
    time_bound_rule = TimeBoundedForwardCollisionRule(1, exit_timer=1)

    monitor = Monitor([
        #collision_rule,
        #ego_lane_change_rule,
        ego_safety_buffer_rule,
        #forward_collision_rule,
        #time_bound_rule,
                       ])

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=OPP_NUMBER+1)
    obs, step_reward, done, info = env.reset(np.array([get_start_position(planner.race, START_OFFSET + i*OPP_OFFSET)
                                                       for i in range(OPP_NUMBER+1)]))
    monitor.evaluate_rules(obs, env)
    env.render(mode='human_fast')

    step = 0

    while not done:

        controls = list()

        controls.append(list(ego_collision_avoider.plan(obs, zd)))

        opp_controls = [opp_collision_avoiders[i].plan(obs) for i in range(OPP_NUMBER)]
        opp_controls = [[opp_control[0], opp_control[1]*opponent_cripple] for opp_control in opp_controls]

        controls += opp_controls

        obs, step_reward, done, info = env.step(np.array(controls))
        step += 1

        x, y = obs['poses_x'][0], obs['poses_y'][0]
        zone = zd.get_zone(x, y)

        env.renderer.update_waypoints(ego_switcher.path_follower.waypoints[:, :2])
        env.render(mode='human_fast')
        monitor.evaluate_rules(obs, env)

    monitor.evaluate_end_rollout(obs, env)
