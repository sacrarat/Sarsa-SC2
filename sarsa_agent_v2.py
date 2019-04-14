import argparse
import os.path
import pickle
import time
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sc2
from sc2 import run_game, maps, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot
from sc2.unit import Unit
from sc2.position import Point2, Point3

from sarsa_table import SarsaLambdaTable

ATTACK = "attack"
MOVE_FORWARD = "moveforward"
MOVE_BACKWARD = "movebackward"
SPREAD_OUT = "spreadout"

# high level actions taken by sarsa agent
smart_actions = [
    ATTACK,
    MOVE_FORWARD,
    MOVE_BACKWARD,
    SPREAD_OUT
]

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--map", type=str, default="5v5")
parser.add_argument("--num_of_eps", type=int, default="1")
parser.add_argument("--normalize", type=int, default=0)  # DEFAULT - no normalization
parser.add_argument("--advantage", type=int, default=1)  # DEFAULT - true advantage
parser.add_argument("--test_id", type=str)
parser.add_argument("--train", type=int, default=0)  # DEFAULT - no training
parser.add_argument("--aggression", type=float, default=0.5)
args = parser.parse_args()

# naming paths for data saving
DATA_FILE = args.test_id + '_save'
TRACE_FILE = args.test_id + '_trace'
COUNTER_FILE = args.test_id + '_counter'
REWARD_FILE = args.test_id + '_reward'
EPISODE_FILE = args.test_id + '_episode'
REPLAY_FILE = args.test_id + '.SC2Replay'
SAVE_PATH = "save/" + args.test_id + "/"

# global variables
episode = 0
episodes = []
rewards = []
ep_change = False
relative_power_scalar = 0
step = 0

# make saving directory for results if doesnt exist
if args.train:
    try:
        os.makedirs(SAVE_PATH)
    except OSError:
        pass


# bot that extends the base botAI class
class SarsaLambdaAgent(sc2.BotAI):
    def __init__(self):
        # instance variables
        self.sarsa_learn = SarsaLambdaTable(actions=list(range(len(smart_actions))))  # sarsa table
        self.previous_state = None
        self.previous_action = None
        self.unit_group_select_counter = 0
        self.init_max_own_health = 0
        self.init_max_enemy_health = 0
        self.current_own_health = 0
        self.current_enemy_health = 0
        self.current_unit_group = []
        self.closest_enemy = None
        self.actions = []
        self.reward = 0  # cumulative reward for episode
        self.sarsa_learn.eligibility_trace *= 0

        # load q table, eligibility trace, counter, rewards and episodes if exist
        if os.path.isfile(SAVE_PATH + DATA_FILE + '.gz'):
            self.sarsa_learn.q_table = pd.read_pickle(SAVE_PATH + DATA_FILE + '.gz', compression='gzip')
            print(self.sarsa_learn.q_table)
        if os.path.isfile(SAVE_PATH + TRACE_FILE + '.gz'):
            self.sarsa_learn.eligibility_trace = pd.read_pickle(SAVE_PATH + TRACE_FILE + '.gz', compression='gzip')
            # initialise to zero every new episode
            self.sarsa_learn.eligibility_trace *= 0
        if os.path.isfile(SAVE_PATH + COUNTER_FILE):
            self.load_state_counter()
        if os.path.isfile(SAVE_PATH + REWARD_FILE):
            self.load_reward()
        if os.path.isfile(SAVE_PATH + EPISODE_FILE):
            self.load_episode()

        if args.train == 0:
            # to evaluate uses epsilon value 1
            self.sarsa_learn.train = 0

    def load_reward(self):
        global rewards
        pickle_in = open(SAVE_PATH + REWARD_FILE, 'rb')
        rewards = pickle.load(pickle_in)

    def load_episode(self):
        global episodes
        pickle_in = open(SAVE_PATH + EPISODE_FILE, 'rb')
        episodes = pickle.load(pickle_in)

    def load_state_counter(self):
        pickle_in = open(SAVE_PATH + COUNTER_FILE, 'rb')
        self.sarsa_learn.state_counter = pickle.load(pickle_in)
        print(self.sarsa_learn.state_counter)

    async def on_step(self, iteration):
        # step function called every 8 frames by default
        self.actions = []
        global rewards, episodes, episode, ep_change, step, relative_power_scalar

        # episode change handling for custom sarsa maps
        if self.state.own_units.empty or self.state.enemy_units.empty:
            print("Episode changed")
            # to fix the Twice Episode changed bug
            if not ep_change:
                # terminal action learning
                reward = self.calculate_reward(smart_actions[self.previous_action])
                rl_action = self.sarsa_learn.choose_action('terminal')
                self.sarsa_learn.learn(str(self.previous_state), self.previous_action, reward, 'terminal', rl_action)
                print("Cumulative Reward:", self.reward)
                rewards.append(self.reward)
                episodes.append(episode)
                episode += 1
                self.reward = 0

            self.init_setup()
            ep_change = True
        else:
            if ep_change:
                self.init_setup()
            ep_change = False

        if iteration == 0:
            self.init_setup()
            relative_power_scalar = self.calculate_global_relative_scalar()

        # step function called every 8 frames
        current_state = self.retrieve_state()

        rl_action = self.sarsa_learn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        if self.previous_action is not None:
            reward = self.calculate_reward(smart_actions[self.previous_action])
            self.reward += reward
            if args.train:
                self.sarsa_learn.learn(str(self.previous_state), self.previous_action, reward, str(current_state),
                                       rl_action)
                if step % 40 == 0 and args.advantage == 1:
                    self.sarsa_learn.apply_advantage()

        step += 1

        self.previous_action = rl_action
        self.previous_state = current_state

        if smart_action == ATTACK:
            # choose and attack lowest health enemy in range.if none present attack closest enemy
            enemy_units = self.state.enemy_units
            if enemy_units.exists:
                for self_unit in self.current_unit_group:
                    in_range_enemies = enemy_units.filter(lambda u: self_unit.target_in_range(u))
                    if len(in_range_enemies) > 1:
                        # attack lowest health enemy in range else attack the closest if all the same health
                        # account for both health and shield in case of protoss units
                        in_range_enemies_by_health = in_range_enemies.sorted(lambda x: x.health + x.shield)
                        if (in_range_enemies_by_health[0].health + in_range_enemies_by_health[0].shield) != (
                                in_range_enemies_by_health[1].health + in_range_enemies_by_health[1].shield):
                            self.actions.append(self_unit.attack(in_range_enemies_by_health[0]))
                        else:
                            closest_enemy = enemy_units.closest_to(self_unit)
                            self.actions.append(self_unit.attack(closest_enemy))
                    elif len(in_range_enemies) == 1:
                        # one enemy in range, attack that one
                        self.actions.append(self_unit.attack(in_range_enemies[0]))
                    else:
                        # enemy not in range, attack closest enemy
                        closest_enemy = enemy_units.closest_to(self_unit)
                        self.actions.append(self_unit.attack(closest_enemy))

        elif smart_action == MOVE_FORWARD:
            # moves towards the closest enemy unit
            enemy_units = self.state.enemy_units
            if enemy_units.exists:
                target = self.current_unit_group.center.position.towards(enemy_units.center, distance=5)
                for unit in self.current_unit_group:
                    self.actions.append(unit.move(target))


        elif smart_action == MOVE_BACKWARD:
            # roughly moves in the opposite direction of the nearest enemy
            enemy_units = self.state.enemy_units
            if enemy_units.exists:
                target = self.current_unit_group.center.position.towards(enemy_units.center, distance=-7)
                if target.x < 0 or target.y < 0 or target.x > 128 or target.y > 128:
                    retreat_points = self.neighbors8(self.current_unit_group.center.position,
                                                     distance=2) | self.neighbors8(
                        self.current_unit_group.center.position, distance=4)
                    target = random.sample(retreat_points, 1)[0]
                for unit in self.current_unit_group:
                    self.actions.append(unit.move(target))


        elif smart_action == SPREAD_OUT:
            # roughly moves in the opposite direction of the unit group center
            for unit in self.current_unit_group:
                unit_group_center = self.current_unit_group.center
                target = unit.position.towards(unit_group_center, distance=-7)
                self.actions.append(unit.move(target))

        if iteration % 500 == 0:
            # save tables to pickle files
            self.sarsa_learn.q_table.to_pickle(SAVE_PATH + DATA_FILE + '.gz', 'gzip')
            self.sarsa_learn.eligibility_trace.to_pickle(SAVE_PATH + TRACE_FILE + '.gz', 'gzip')
            self.save_state_counter(self.sarsa_learn.state_counter)
            self.save_reward(rewards)
            self.save_episode(episodes)
            # plot graphs
            if args.train == 1:
                if len(episodes) != 0:
                    plot_graph(episodes, rewards, exponential_moving_average=False)
                    plot_graph2(episodes, rewards)
                    plot_graph3(episodes, rewards)

        # dispatch actions to sc2 environment for stepping
        await self.do_actions(self.actions)

    def inPathingGrid(self, pos):
        # taken from https://github.com/Dentosal/python-sc2/blob/master/examples/terran/mass_reaper.py
        # returns True if it is possible for a ground unit to move to pos - doesnt seem to work on ramps or near edges
        assert isinstance(pos, (Point2, Point3, Unit))
        pos = pos.position.to2.rounded
        return self._game_info.pathing_grid[(pos)] != 0

    # stolen and modified from position.py
    def neighbors4(self, position, distance=1):
        # taken from https://github.com/Dentosal/python-sc2/blob/master/examples/terran/mass_reaper.py
        p = position
        d = distance
        return {
            Point2((p.x - d, p.y)),
            Point2((p.x + d, p.y)),
            Point2((p.x, p.y - d)),
            Point2((p.x, p.y + d)),
        }

    # stolen and modified from position.py
    def neighbors8(self, position, distance=1):
        # taken from https://github.com/Dentosal/python-sc2/blob/master/examples/terran/mass_reaper.py
        p = position
        d = distance
        return self.neighbors4(position, distance) | {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d)),
        }

    def on_end(self, game_result):
        # only runs if game ends without crashing
        reward = self.calculate_reward(smart_actions[self.previous_action])
        rl_action = self.sarsa_learn.choose_action('terminal')
        self.sarsa_learn.learn(str(self.previous_state), self.previous_action, reward, 'terminal', rl_action)
        self.sarsa_learn.q_table.to_pickle(SAVE_PATH + DATA_FILE + '.gz', 'gzip')
        self.sarsa_learn.eligibility_trace.to_pickle(SAVE_PATH + TRACE_FILE + '.gz', 'gzip')
        self.save_state_counter(self.sarsa_learn.state_counter)
        self.save_reward(rewards)
        self.save_episode(episodes)
        self.previous_action = None
        self.previous_state = None

    def save_state_counter(self, state_counter):
        def pickle_dump(data, filename):
            with open(filename, 'wb') as output:
                pickle.dump(data, output, 1)

        pickle_dump(state_counter, SAVE_PATH + COUNTER_FILE)

    def save_reward(self, reward):
        def pickle_dump(data, filename):
            with open(filename, 'wb') as output:
                pickle.dump(data, output, 1)

        pickle_dump(reward, SAVE_PATH + REWARD_FILE)

    def save_episode(self, episode):
        def pickle_dump(data, filename):
            with open(filename, 'wb') as output:
                pickle.dump(data, output, 1)

        pickle_dump(episode, SAVE_PATH + EPISODE_FILE)

    def calculate_global_relative_scalar(self):
        own_units = self.state.own_units
        enemy_units = self.state.enemy_units

        # relative power
        own_cost = 0
        for own_unit in own_units:
            unit_cost = 0
            unit_cost += own_unit._type_data.cost.minerals + own_unit._type_data.cost.vespene
            own_cost += unit_cost

        enemy_cost = 0
        for enemy_unit in enemy_units:
            unit_cost = 0
            unit_cost += enemy_unit._type_data.cost.minerals + enemy_unit._type_data.cost.vespene
            enemy_cost += unit_cost

        result = max(own_cost, enemy_cost)

        return result

    # inspired by NiKeys generalised sarsa work
    def retrieve_state(self):
        state = np.zeros(6, dtype=int)

        own_units = self.state.own_units
        enemy_units = self.state.enemy_units

        own_unit_types = []

        # relative power
        own_cost = 0
        for own_unit in own_units:
            unit_cost = 0
            unit_cost += own_unit._type_data.cost.minerals + own_unit._type_data.cost.vespene
            own_cost += unit_cost
            own_unit_type = own_unit.type_id
            if own_unit_type not in own_unit_types:
                own_unit_types.append(own_unit_type)

        enemy_cost = 0
        for enemy_unit in enemy_units:
            unit_cost = 0
            unit_cost += enemy_unit._type_data.cost.minerals + enemy_unit._type_data.cost.vespene
            enemy_cost += unit_cost
        relative_power = own_cost - enemy_cost
        scaled_relative_power = round((relative_power / float(relative_power_scalar)) * 3)  # give within [-3 to +3]

        # determine all available unit groups
        unit_types_available = own_unit_types
        unit_types_available = sorted(unit_types_available, key=lambda x: x.value)

        # create a list of unit groups based on unit_types if the group has units alive

        unit_groups = [own_units.of_type({unit_group_type}) for unit_group_type in unit_types_available if
                       own_units.of_type({unit_group_type}).exists]

        # select one unit group through a wrap around cycle counter
        current_unit_group = self.current_unit_group
        if len(unit_groups) > 0:
            current_unit_group = unit_groups[self.unit_group_select_counter % len(unit_groups)]
            self.current_unit_group = current_unit_group
            self.unit_group_select_counter += 1

        # unit type of currently selected unit subgroup
        current_unit_type = current_unit_group.random.type_id.value

        # unit type and distance to closest enemy to currently selected unit subgroup
        closest_enemy = self.closest_enemy
        if len(self.known_enemy_units):
            closest_enemy = self.known_enemy_units.sorted(lambda x: x.distance_to(current_unit_group.first))[
                0]  # TODO should be particular unit distance or center of subgroup?
            self.closest_enemy = closest_enemy
        closest_enemy_type = closest_enemy.type_id.value
        closest_enemy_distance = current_unit_group.center.distance_to_closest(
            [closest_enemy.position])  # TODO passing in a list? use center?

        # weapon cooldown for currently selected unit subgroup
        cooldown_counter = 0
        for unit in current_unit_group:
            if cooldown_counter > (len(current_unit_group) / 2):
                break
            if unit.weapon_cooldown > 0:
                cooldown_counter += 1
        if cooldown_counter > (len(current_unit_group) / 2):
            is_weapon_cooldown = 1  # majority are on cooldown
        else:
            is_weapon_cooldown = 0

        # clumping check for currently selected unit subgroup
        # cycle through all units in subgroup and check if a majority are away a certain distance from center
        clump_counter = 0
        for unit in current_unit_group:
            clump_decider_value = 5  # TODO tune appropriately
            if clump_counter > (len(current_unit_group) / 2):
                break
            if unit.distance_to(current_unit_group.center) > clump_decider_value:
                clump_counter += 1
        if clump_counter > (len(current_unit_group) / 2):  # TODO simple majority or 3/4th?
            is_clumped = 1  # majority are clumped
        else:
            is_clumped = 0

        state[0] = scaled_relative_power
        state[1] = int(current_unit_type)
        state[2] = int(closest_enemy_type)
        state[3] = int(closest_enemy_distance)
        state[4] = is_weapon_cooldown
        state[5] = is_clumped

        return state

    # inspired by NiKeys generalised sarsa work
    def calculate_reward(self, smart_action):
        # reward calculates damage applied and received from step to step not overall
        reward = 0

        own_units = self.state.own_units
        enemy_units = self.state.enemy_units

        current_own_health = 0
        for unit in own_units:
            current_own_health += unit.health + unit.shield

        # to fix the episode changed issue - unit does not spawn therefore own health was zero
        if self.current_own_health == 0:
            damage_received = 0
        else:
            damage_received = self.current_own_health - current_own_health

        current_enemy_health = 0
        for unit in enemy_units:
            current_enemy_health += unit.health + unit.shield

        # to fix the episode changed issue - enemies already on map before new units spawned
        if self.current_own_health == 0 and self.current_enemy_health > 0:
            damage_applied = 0
        else:
            # to only consider damage applied incase we attacked and not because enemies killed themselves (banelings)
            if smart_action == ATTACK:
                damage_applied = self.current_enemy_health - current_enemy_health
            else:
                damage_applied = 0

        self.current_enemy_health = current_enemy_health
        self.current_own_health = current_own_health

        aggression = args.aggression

        reward = damage_applied - (damage_received * (1 - aggression))

        if args.normalize:
            # normalizing reward to a scale of -1 to 1
            max_reward = current_enemy_health
            min_reward = 0 - (current_own_health * (1 - aggression))
            norm_reward = 2 * ((reward - min_reward) / (max_reward - min_reward)) - 1
            return norm_reward

        print(reward)
        return reward

    def init_setup(self):
        own_units = self.state.own_units
        enemy_units = self.state.enemy_units

        max_own_health = 0
        current_own_health = 0
        for unit in own_units:
            max_own_health += unit.health_max + unit.shield_max
            current_own_health += unit.health + unit.shield

        max_enemy_health = 0
        current_enemy_health = 0
        for unit in enemy_units:
            max_enemy_health += unit.health_max + unit.shield_max
            current_enemy_health += unit.health + unit.shield

        self.init_max_own_health = max_own_health
        self.init_max_enemy_health = max_enemy_health
        self.current_own_health = max_own_health
        self.current_enemy_health = max_enemy_health
        self.previous_state = None
        self.previous_action = None
        self.unit_group_select_counter = 0
        # self.current_unit_group = []
        # self.closest_enemy = None
        self.actions = []
        self.sarsa_learn.eligibility_trace *= 0


def plot_graph3(episode_list, reward_list):
    # rolling mean reward of last 100 episodes
    print("Plotting Graph")
    plt.clf()
    plot_episodes = episode_list
    rews = pd.DataFrame(np.array(reward_list))
    rolling_mean = rews.rolling(100, 1).mean()[0].tolist()
    plt.plot(plot_episodes, rolling_mean, color="blue", label="reward")
    plt.title("Learning Curve")
    plt.xlabel("Episodes"), plt.ylabel("Rolling Reward Averaged over the last 100 episodes"), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(SAVE_PATH + "rolling_graph.svg")


def plot_graph2(episode_list, reward_list):
    # exponential moving mean reward
    print("Plotting Graph")
    plt.clf()
    plot_episodes = episode_list
    rews = pd.DataFrame(np.array(reward_list))
    ewm = rews.ewm(alpha=0.7).mean()[0].tolist()
    plt.plot(plot_episodes, ewm, color="blue", label="reward")
    # plt.set_ylim(bottom=0)
    plt.title("Learning Curve")
    plt.xlabel("Episodes"), plt.ylabel("Exponential Moving Average Reward"), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(SAVE_PATH + "ema_graph.svg")


def plot_graph(episode_list, reward_list, exponential_moving_average=False, interval=30):
    # simple average
    print("Plotting Graph")
    plt.clf()
    plot_episodes = episode_list[::interval]
    plot_episodes.append(plot_episodes[-1] + interval)
    rew = np.array(reward_list)
    # reshape into 100 episode reward lists and pad the end with the extreme value
    shaped_rewards = np.pad(rew, (0, interval - rew.size % interval), mode='constant').reshape(-1, interval)

    if exponential_moving_average:
        # exponential moving average
        ema_rewards = []
        for reward_ep_set in shaped_rewards:
            reward_ep_set = pd.DataFrame(np.array(reward_ep_set))
            ewm = reward_ep_set.ewm(alpha=0.6).mean()[0].iloc[-1]
            if 0 not in reward_ep_set:
                ema_rewards.append(ewm)
        plt.plot(plot_episodes, ema_rewards, color="blue", label="reward")
    else:
        # simple average
        # simple_average_rewards = list(shaped_rewards.sum(1) / (rew.size // interval))
        simple_average_rewards = []
        simple_average_rewards.append(rew[0])
        for reward_ep_set in shaped_rewards:
            if np.count_nonzero(reward_ep_set) > 0:
                sa = sum(reward_ep_set) / np.count_nonzero(reward_ep_set)
                simple_average_rewards.append(sa)
        plt.plot(plot_episodes, simple_average_rewards, color="blue", label="reward")
    plt.title("Learning Curve")
    plt.xlabel("Episodes"), plt.ylabel("Average Reward for the last 100 episodes"), plt.legend(loc="best")
    plt.tight_layout()

    if exponential_moving_average:
        plt.savefig(SAVE_PATH + "ema_graph.svg")
    else:
        plt.savefig(SAVE_PATH + "sa_graph.svg")


def train():
    print("Starting training.")
    print(time.time)
    start = time.time()
    for _ in range(args.num_of_eps):
        run_game(maps.get(args.map), [Bot(Race.Terran, SarsaLambdaAgent())], realtime=False,
                 save_replay_as=SAVE_PATH + REPLAY_FILE, random_seed=1)
    # if args.train == 1:
        # plot_graph(episodes, rewards, exponential_moving_average=True)
        # plot_graph(episodes, rewards, exponential_moving_average=False)
    end = time.time()
    print(end)
    print("Training finished.")


def main():
    train()


if __name__ == '__main__':
    main()
