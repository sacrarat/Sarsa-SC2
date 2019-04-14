"""
bot taken from https://github.com/Dentosal/python-sc2/blob/master/examples/terran/mass_reaper.py with the micro
replaced by our sarsa agent
"""

"""
Bot that stays on 1base, goes 4 rax mass reaper
This bot is one of the first examples that are micro intensive
Bot has a chance to win against elite (=Difficulty.VeryHard) zerg AI
Bot made by Burny
"""
import argparse
import random
import pickle
import os.path
import numpy as np
import sc2
import pandas as pd
from matplotlib import pyplot as plt

from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.player import Bot, Computer
from sc2.player import Human
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

from sarsa_table_bot import SarsaLambdaTable

parser = argparse.ArgumentParser()
parser.add_argument("--test_id", type=str, default="testbot")
parser.add_argument("--train", type=int, default=1)  # DEFAULT - no training
parser.add_argument("--num_of_eps", type=int, default="1")
parser.add_argument("--sparse_reward", type=int, default=1)
parser.add_argument("--difficulty", type=str, default="VeryEasy")
args = parser.parse_args()

DATA_FILE = args.test_id + '_save'
TRACE_FILE = args.test_id + '_trace'
COUNTER_FILE = args.test_id + '_counter'
REWARD_FILE = args.test_id + '_reward'
EPISODE_FILE = args.test_id + '_episode'
REPLAY_FILE = args.test_id + '.SC2Replay'
SAVE_PATH = "save/" + args.test_id + "/"

rewards = []
episodes = []

if args.train:
    try:
        os.makedirs(SAVE_PATH)
    except OSError:
        pass

IDLE = 'idle'
MOVE_FORWARDS = 'moveforwards'
MOVE_BACKWARDS = 'movebackwards'
SCATTER = 'scatter'
CLUMP = 'gather'
ATTACK_ENEMY_BASE = 'attack_enemy_base'
ATTACK_ENEMY_STRUCTURE = 'attack_enemy_structure'
ATTACK_ENEMY_UNIT = 'attack_enemy_unit'
DEFEND = 'defend'

smart_actions = [
    IDLE,
    MOVE_FORWARDS,
    MOVE_BACKWARDS,
    SCATTER,
    CLUMP,
    ATTACK_ENEMY_BASE,
    ATTACK_ENEMY_STRUCTURE,
    ATTACK_ENEMY_UNIT,
    DEFEND
]


class MassReaperBot(sc2.BotAI):
    def __init__(self):
        self.combinedActions = []
        self.current_unit_group = []
        self.closest_enemy = None
        self.previous_state = None
        self.previous_action = None
        self.unit_group_select_counter = 0
        self.sarsa_learn = SarsaLambdaTable(actions=list(range(len(smart_actions))))
        self.sarsa_learn.eligibility_trace *= 0
        self.prev_kill_unit_score = 0
        self.prev_kill_building_score = 0

        if args.sparse_reward:
            self.sparse_reward = True
        else:
            self.sparse_reward = False

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
        self.combinedActions = []

        """
        -  depots when low on remaining supply
        - townhalls contains commandcenter and orbitalcommand
        - self.units(TYPE).not_ready.amount selects all units of that type, filters incomplete units, and then counts the amount
        - self.already_pending(TYPE) counts how many units are queued - but in this bot below you will find a slightly different already_pending function which only counts units queued (but not in construction)
        """
        if self.supply_left < 5 and self.townhalls.exists and self.supply_used >= 14 and self.can_afford(
                UnitTypeId.SUPPLYDEPOT) and self.units(UnitTypeId.SUPPLYDEPOT).not_ready.amount + self.already_pending(
            UnitTypeId.SUPPLYDEPOT) < 1:
            ws = self.workers.gathering
            if ws:  # if workers found
                w = ws.furthest_to(ws.center)
                loc = await self.find_placement(UnitTypeId.SUPPLYDEPOT, w.position, placement_step=3)
                if loc:  # if a placement location was found
                    # build exactly on that location
                    self.combinedActions.append(w.build(UnitTypeId.SUPPLYDEPOT, loc))

        # lower all depots when finished
        for depot in self.units(UnitTypeId.SUPPLYDEPOT).ready:
            self.combinedActions.append(depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER))

        # morph commandcenter to orbitalcommand
        if self.units(UnitTypeId.BARRACKS).ready.exists and self.can_afford(
                UnitTypeId.ORBITALCOMMAND):  # check if orbital is affordable
            for cc in self.units(UnitTypeId.COMMANDCENTER).idle:  # .idle filters idle command centers
                self.combinedActions.append(cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND))

        # expand if we can afford and have less than 2 bases
        if 1 <= self.townhalls.amount < 2 and self.already_pending(UnitTypeId.COMMANDCENTER) == 0 and self.can_afford(
                UnitTypeId.COMMANDCENTER):
            # get_next_expansion returns the center of the mineral fields of the next nearby expansion
            next_expo = await self.get_next_expansion()
            # from the center of mineral fields, we need to find a valid place to place the command center

            if isinstance(next_expo, Point2):
                location = await self.find_placement(UnitTypeId.COMMANDCENTER, next_expo, placement_step=1)
                if location:
                    # now we "select" (or choose) the nearest worker to that found location
                    w = self.select_build_worker(location)
                    if w and self.can_afford(UnitTypeId.COMMANDCENTER):
                        # the worker will be commanded to build the command center
                        error = await self.do(w.build(UnitTypeId.COMMANDCENTER, location))
                        if error:
                            print(error)

        # make up to 4 barracks if we can afford them
        # check if we have a supply depot (tech requirement) before trying to make barracks
        if self.units.of_type([UnitTypeId.SUPPLYDEPOT, UnitTypeId.SUPPLYDEPOTLOWERED,
                               UnitTypeId.SUPPLYDEPOTDROP]).ready.exists and self.units(
            UnitTypeId.BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) < 4 and self.can_afford(
            UnitTypeId.BARRACKS):
            ws = self.workers.gathering
            if ws and self.townhalls.exists:  # need to check if townhalls.amount > 0 because placement is based on townhall location
                w = ws.furthest_to(ws.center)
                # I chose placement_step 4 here so there will be gaps between barracks hopefully
                loc = await self.find_placement(UnitTypeId.BARRACKS, self.townhalls.random.position, placement_step=4)
                if loc:
                    self.combinedActions.append(w.build(UnitTypeId.BARRACKS, loc))

        # build refineries (on nearby vespene) when at least one barracks is in construction
        if self.units(UnitTypeId.BARRACKS).amount > 0 and self.already_pending(UnitTypeId.REFINERY) < 1:
            for th in self.townhalls:
                vgs = self.state.vespene_geyser.closer_than(10, th)
                for vg in vgs:
                    if await self.can_place(UnitTypeId.REFINERY, vg.position) and self.can_afford(UnitTypeId.REFINERY):
                        ws = self.workers.gathering
                        if ws.exists:  # same condition as above
                            w = ws.closest_to(vg)
                            # caution: the target for the refinery has to be the vespene geyser, not its position!
                            self.combinedActions.append(w.build(UnitTypeId.REFINERY, vg))

        # make scvs until 18, usually you only need 1:1 mineral:gas ratio for reapers, but if you don't lose any then you will need additional depots (mule income should take care of that)
        # stop scv production when barracks is complete but we still have a command cender (priotize morphing to orbital command)
        if self.can_afford(UnitTypeId.SCV) and self.supply_left > 0 and self.units(UnitTypeId.SCV).amount < 18 and (
                self.units(UnitTypeId.BARRACKS).ready.amount < 1 and self.units(
            UnitTypeId.COMMANDCENTER).idle.exists or self.units(UnitTypeId.ORBITALCOMMAND).idle.exists):
            for th in self.townhalls.idle:
                self.combinedActions.append(th.train(UnitTypeId.SCV))

        # make reapers if we can afford them and we have supply remaining
        if self.can_afford(UnitTypeId.REAPER) and self.supply_left > 0:
            # loop through all idle barracks
            for rax in self.units(UnitTypeId.BARRACKS).idle:
                self.combinedActions.append(rax.train(UnitTypeId.REAPER))

        # send workers to mine from gas
        if iteration % 25 == 0:
            await self.distribute_workers()

        current_state = self.retrieve_state()

        """
        SARSA AGENT MICRO starts here
        """
        if len(self.current_unit_group) > 0:

            rl_action = self.sarsa_learn.choose_action(str(current_state))
            smart_action = smart_actions[rl_action]

            print(smart_action)

            if self.previous_action is not None:
                reward = self.calculate_rewards(sparse=self.sparse_reward, game_result=None)
                self.sarsa_learn.learn(str(self.previous_state), self.previous_action, reward, str(current_state),
                                       rl_action)

            self.previous_action = rl_action
            self.previous_state = current_state

            if smart_action == IDLE:
                pass
            elif smart_action == MOVE_FORWARDS:
                enemy_units = self.known_enemy_units.not_structure
                if enemy_units.exists:
                    target = self.current_unit_group.center.position.towards(enemy_units.center, distance=5)
                    for unit in self.current_unit_group:
                        self.combinedActions.append(unit.move(target))
                else:
                    target = self.current_unit_group.center.position.towards(
                        random.choice(self.enemy_start_locations).position, distance=5)
                    for unit in self.current_unit_group:
                        self.combinedActions.append(unit.move(target))
            elif smart_action == MOVE_BACKWARDS:
                enemy_units = self.known_enemy_units.not_structure
                if enemy_units.exists:
                    target = self.current_unit_group.center.position.towards(enemy_units.center, distance=-7)
                    if target.x < 0 or target.y < 0 or target.x > 64 or target.y > 64:
                        retreat_points = self.neighbors8(self.current_unit_group.center.position,
                                                         distance=2) | self.neighbors8(
                            self.current_unit_group.center.position, distance=4)
                        target = random.sample(retreat_points, 1)[0]
                    for unit in self.current_unit_group:
                        self.combinedActions.append(unit.move(target))
            elif smart_action == SCATTER:
                for unit in self.current_unit_group:
                    unit_group_center = self.current_unit_group.center
                    target = unit.position.towards(unit_group_center, distance=-7)
                    self.combinedActions.append(unit.move(target))
            elif smart_action == CLUMP:
                for unit in self.current_unit_group:
                    unit_group_center = self.current_unit_group.center
                    target = unit.position.towards(unit_group_center, distance=7)
                    self.combinedActions.append(unit.move(target))
            elif smart_action == ATTACK_ENEMY_BASE:
                target = random.choice(self.enemy_start_locations)
                for unit in self.current_unit_group:
                    self.combinedActions.append(unit.attack(target))
            elif smart_action == ATTACK_ENEMY_STRUCTURE:
                enemy_structures = self.known_enemy_structures
                if enemy_structures.exists:
                    target = enemy_structures.closest_to(self.current_unit_group.first)
                    for unit in self.current_unit_group:
                        self.combinedActions.append(unit.attack(target))
            elif smart_action == ATTACK_ENEMY_UNIT:
                enemy_units = self.known_enemy_units.not_structure
                if enemy_units.exists:
                    for self_unit in self.current_unit_group:
                        in_range_enemies = enemy_units.filter(lambda u: self_unit.target_in_range(u))
                        if len(in_range_enemies) > 1:
                            # attack lowest health enemy in range else attack the closest if all the same health
                            # account for both health and shield in case of protoss units
                            in_range_enemies_by_health = in_range_enemies.sorted(lambda x: x.health + x.shield)
                            if (in_range_enemies_by_health[0].health + in_range_enemies_by_health[0].shield) != (
                                    in_range_enemies_by_health[1].health + in_range_enemies_by_health[1].shield):
                                self.combinedActions.append(self_unit.attack(in_range_enemies_by_health[0]))
                            else:
                                closest_enemy = enemy_units.closest_to(self_unit)
                                self.combinedActions.append(self_unit.attack(closest_enemy))
                        elif len(in_range_enemies) == 1:
                            # one enemy in range, attack that one
                            self.combinedActions.append(self_unit.attack(in_range_enemies[0]))
                        else:
                            # enemy not in range, attack closest enemy
                            closest_enemy = enemy_units.closest_to(self_unit)
                            self.combinedActions.append(self_unit.attack(closest_enemy))
            elif smart_action == DEFEND:
                target = self.start_location
                for unit in self.current_unit_group:
                    self.combinedActions.append(unit.attack(target))

        # manage idle scvs, would be taken care by distribute workers aswell
        if self.townhalls.exists:
            for w in self.workers.idle:
                th = self.townhalls.closest_to(w)
                mfs = self.state.mineral_field.closer_than(10, th)
                if mfs:
                    mf = mfs.closest_to(w)
                    self.combinedActions.append(w.gather(mf))

        # manage orbital energy and drop mules
        for oc in self.units(UnitTypeId.ORBITALCOMMAND).filter(lambda x: x.energy >= 50):
            mfs = self.state.mineral_field.closer_than(10, oc)
            if mfs:
                mf = max(mfs, key=lambda x: x.mineral_contents)
                self.combinedActions.append(oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf))

        # when running out of mineral fields near command center, fly to next base with minerals

        # execute actions
        await self.do_actions(self.combinedActions)

    def calculate_rewards(self, sparse=True, game_result=None):
        reward = 0
        if not sparse:
            kill_unit_reward = 1
            kill_building_reward = 2

            kill_unit_score = self.state.score.killed_value_units
            kill_building_score = self.state.score.killed_value_structures

            if kill_unit_score > self.prev_kill_unit_score:
                reward += kill_unit_reward
            if kill_building_score > self.prev_kill_building_score:
                reward += kill_building_reward

            self.prev_kill_unit_score = kill_unit_score
            self.prev_kill_building_score = kill_building_score

        if game_result is not None:
            if game_result.name == "Victory":
                reward += 100
            elif game_result.name == "Tie":
                reward += 0
            elif game_result.name == "Defeat":
                reward += -100

        return reward

    def calculate_global_relative_scalar(self):
        own_units = self.state.own_units
        enemy_units = self.state.enemy_units

        # relative power
        own_cost = 0
        for own_unit in own_units.not_structure:
            unit_cost = 0
            unit_cost += own_unit._type_data.cost.minerals + own_unit._type_data.cost.vespene
            own_cost += unit_cost

        enemy_cost = 0
        for enemy_unit in enemy_units.not_structure:
            unit_cost = 0
            unit_cost += enemy_unit._type_data.cost.minerals + enemy_unit._type_data.cost.vespene
            enemy_cost += unit_cost

        result = max(own_cost, enemy_cost)

        return result

    def retrieve_state(self):
        """
        :return: [relative_power, closest_enemy_type, self_unit_type, distance_to_closest_enemy, army_supply, weapon_cooldown, is_clumped]
        """
        state = np.zeros(8, dtype=int)

        own_units = self.state.own_units  # includes both army units and structures
        enemy_units = self.state.enemy_units

        own_unit_types = []

        # relative power
        own_cost = 0
        for own_unit in own_units.not_structure:
            unit_cost = 0
            unit_cost += own_unit._type_data.cost.minerals + own_unit._type_data.cost.vespene
            own_cost += unit_cost
            own_unit_type = own_unit.type_id
            if own_unit_type not in own_unit_types:
                own_unit_types.append(own_unit_type)

        enemy_cost = 0
        for enemy_unit in enemy_units.not_structure:
            unit_cost = 0
            unit_cost += enemy_unit._type_data.cost.minerals + enemy_unit._type_data.cost.vespene
            enemy_cost += unit_cost
        relative_power = own_cost - enemy_cost
        relative_power_scalar = self.calculate_global_relative_scalar()
        if relative_power_scalar > 0:
            scaled_relative_power = round((relative_power / float(relative_power_scalar)) * 3)  # give within [-3 to +3]
        else:
            scaled_relative_power = 0
        # determine all available unit groups
        unit_types_available = own_unit_types
        if UnitTypeId.SCV in unit_types_available:
            unit_types_available.remove(UnitTypeId.SCV)
        if UnitTypeId.MULE in unit_types_available:
            unit_types_available.remove(UnitTypeId.MULE)
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
        if len(unit_groups) > 0:
            current_unit_type = current_unit_group.random.type_id.value
        else:
            current_unit_type = 0

        # unit type and distance to closest enemy to currently selected unit subgroup
        closest_enemy = self.closest_enemy
        if len(self.known_enemy_units):
            closest_enemy = self.known_enemy_units.sorted(lambda x: x.distance_to(current_unit_group.first))[
                0]  # TODO should be particular unit distance or center of subgroup?
            self.closest_enemy = closest_enemy
            closest_enemy_type = closest_enemy.type_id.value
            closest_enemy_distance = current_unit_group.center.distance_to_closest(
                [closest_enemy.position])  # TODO passing in a list? use center?
        else:
            closest_enemy = 0
            closest_enemy_type = 0
            closest_enemy_distance = 0

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

        army_supply = own_units.ready.not_structure.of_type(unit_types_available).amount

        enemy_visible = enemy_units.not_structure.exists

        state[0] = scaled_relative_power
        state[1] = int(current_unit_type)
        # state[2] = int(closest_enemy_type)
        # state[3] = int(closest_enemy_distance)
        state[4] = is_weapon_cooldown
        state[5] = is_clumped
        state[6] = army_supply
        state[7] = enemy_visible

        return state

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

    def on_end(self, game_result):
        global rewards
        rl_action = self.sarsa_learn.choose_action('terminal')
        reward = 0
        print(game_result.name)
        reward = self.calculate_rewards(sparse=self.sparse_reward, game_result=game_result)
        rewards.append(reward)
        self.sarsa_learn.learn(str(self.previous_state), self.previous_action, reward, 'terminal', rl_action)
        self.sarsa_learn.q_table.to_pickle(SAVE_PATH + DATA_FILE + '.gz', 'gzip')
        self.sarsa_learn.eligibility_trace.to_pickle(SAVE_PATH + TRACE_FILE + '.gz', 'gzip')
        self.save_state_counter(self.sarsa_learn.state_counter)
        self.save_reward(rewards)
        self.save_episode(episodes)

    # helper functions

    # this checks if a ground unit can walk on a Point2 position

    def inPathingGrid(self, pos):
        # returns True if it is possible for a ground unit to move to pos - doesnt seem to work on ramps or near edges
        assert isinstance(pos, (Point2, Point3, Unit))
        pos = pos.position.to2.rounded
        return self._game_info.pathing_grid[(pos)] != 0

        # stolen and modified from position.py

    def neighbors4(self, position, distance=1):
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
        p = position
        d = distance
        return self.neighbors4(position, distance) | {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d)),
        }

        # already pending function rewritten to only capture units in queue and queued buildings
        # the difference to bot_ai.py alredy_pending() is: it will not cover structures in construction

    def already_pending(self, unit_type):
        ability = self._game_data.units[unit_type.value].creation_ability
        unitAttributes = self._game_data.units[unit_type.value].attributes

        buildings_in_construction = self.units.structure(unit_type).not_ready
        if 8 not in unitAttributes and any(o.ability == ability for w in (self.units.not_structure) for o in w.orders):
            return sum([o.ability == ability for w in (self.units - self.workers) for o in w.orders])
        # following checks for unit production in a building queue, like queen, also checks if hatch is morphing to LAIR
        elif any(o.ability.id == ability.id for w in (self.units.structure) for o in w.orders):
            return sum([o.ability.id == ability.id for w in (self.units.structure) for o in w.orders])
        # the following checks if a worker is about to start a construction (and for scvs still constructing if not checked for structures with same position as target)
        elif any(o.ability == ability for w in self.workers for o in w.orders):
            return sum([o.ability == ability for w in self.workers for o in w.orders]) \
                   - buildings_in_construction.amount
        elif any(egg.orders[0].ability == ability for egg in self.units(UnitTypeId.EGG)):
            return sum([egg.orders[0].ability == ability for egg in self.units(UnitTypeId.EGG)])
        return 0

        # distribute workers function rewritten, the default distribute_workers() function did not saturate gas quickly enough

    async def distribute_workers(self, performanceHeavy=True, onlySaturateGas=False):
        # expansion_locations = self.expansion_locations
        # owned_expansions = self.owned_expansions

        mineralTags = [x.tag for x in self.state.units.mineral_field]
        # gasTags = [x.tag for x in self.state.units.vespene_geyser]
        geyserTags = [x.tag for x in self.geysers]

        workerPool = self.units & []
        workerPoolTags = set()

        # find all geysers that have surplus or deficit
        deficitGeysers = {}
        surplusGeysers = {}
        for g in self.geysers.filter(lambda x: x.vespene_contents > 0):
            # only loop over geysers that have still gas in them
            deficit = g.ideal_harvesters - g.assigned_harvesters
            if deficit > 0:
                deficitGeysers[g.tag] = {"unit": g, "deficit": deficit}
            elif deficit < 0:
                surplusWorkers = self.workers.closer_than(10, g).filter(
                    lambda w: w not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [
                        AbilityId.HARVEST_GATHER] and w.orders[0].target in geyserTags)
                # workerPool.extend(surplusWorkers)
                for i in range(-deficit):
                    if surplusWorkers.amount > 0:
                        w = surplusWorkers.pop()
                        workerPool.append(w)
                        workerPoolTags.add(w.tag)
                surplusGeysers[g.tag] = {"unit": g, "deficit": deficit}

        # find all townhalls that have surplus or deficit
        deficitTownhalls = {}
        surplusTownhalls = {}
        if not onlySaturateGas:
            for th in self.townhalls:
                deficit = th.ideal_harvesters - th.assigned_harvesters
                if deficit > 0:
                    deficitTownhalls[th.tag] = {"unit": th, "deficit": deficit}
                elif deficit < 0:
                    surplusWorkers = self.workers.closer_than(10, th).filter(
                        lambda w: w.tag not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [
                            AbilityId.HARVEST_GATHER] and w.orders[0].target in mineralTags)
                    # workerPool.extend(surplusWorkers)
                    for i in range(-deficit):
                        if surplusWorkers.amount > 0:
                            w = surplusWorkers.pop()
                            workerPool.append(w)
                            workerPoolTags.add(w.tag)
                    surplusTownhalls[th.tag] = {"unit": th, "deficit": deficit}

            if all([len(deficitGeysers) == 0, len(surplusGeysers) == 0,
                    len(surplusTownhalls) == 0 or deficitTownhalls == 0]):
                # cancel early if there is nothing to balance
                return

        # check if deficit in gas less or equal than what we have in surplus, else grab some more workers from surplus bases
        deficitGasCount = sum(
            gasInfo["deficit"] for gasTag, gasInfo in deficitGeysers.items() if gasInfo["deficit"] > 0)
        surplusCount = sum(-gasInfo["deficit"] for gasTag, gasInfo in surplusGeysers.items() if gasInfo["deficit"] < 0)
        surplusCount += sum(-thInfo["deficit"] for thTag, thInfo in surplusTownhalls.items() if thInfo["deficit"] < 0)

        if deficitGasCount - surplusCount > 0:
            # grab workers near the gas who are mining minerals
            for gTag, gInfo in deficitGeysers.items():
                if workerPool.amount >= deficitGasCount:
                    break
                workersNearGas = self.workers.closer_than(10, gInfo["unit"]).filter(
                    lambda w: w.tag not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [
                        AbilityId.HARVEST_GATHER] and w.orders[0].target in mineralTags)
                while workersNearGas.amount > 0 and workerPool.amount < deficitGasCount:
                    w = workersNearGas.pop()
                    workerPool.append(w)
                    workerPoolTags.add(w.tag)

        # now we should have enough workers in the pool to saturate all gases, and if there are workers left over, make them mine at townhalls that have mineral workers deficit
        for gTag, gInfo in deficitGeysers.items():
            if performanceHeavy:
                # sort furthest away to closest (as the pop() function will take the last element)
                workerPool.sort(key=lambda x: x.distance_to(gInfo["unit"]), reverse=True)
            for i in range(gInfo["deficit"]):
                if workerPool.amount > 0:
                    w = workerPool.pop()
                    if len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_RETURN]:
                        self.combinedActions.append(w.gather(gInfo["unit"], queue=True))
                    else:
                        self.combinedActions.append(w.gather(gInfo["unit"]))

        if not onlySaturateGas:
            # if we now have left over workers, make them mine at bases with deficit in mineral workers
            for thTag, thInfo in deficitTownhalls.items():
                if performanceHeavy:
                    # sort furthest away to closest (as the pop() function will take the last element)
                    workerPool.sort(key=lambda x: x.distance_to(thInfo["unit"]), reverse=True)
                for i in range(thInfo["deficit"]):
                    if workerPool.amount > 0:
                        w = workerPool.pop()
                        mf = self.state.mineral_field.closer_than(10, thInfo["unit"]).closest_to(w)
                        if len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_RETURN]:
                            self.combinedActions.append(w.gather(mf, queue=True))
                        else:
                            self.combinedActions.append(w.gather(mf))


def plot_graph3(episode_list, reward_list):
    print("Plotting Graph")
    plt.clf()
    episode_list = np.arange(0, len(reward_list), 1)
    plot_episodes = episode_list
    rews = pd.DataFrame(np.array(reward_list))
    rolling_mean = rews.rolling(10, 1).mean()[0].tolist()
    plt.plot(plot_episodes, rolling_mean, color="blue", label="reward")
    plt.title("Learning Curve")
    plt.xlabel("Episodes"), plt.ylabel("Rolling Reward Averaged over the last 100 episodes"), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(SAVE_PATH + "rolling_graph.svg")


def train():
    global episodes
    print("Starting training.")
    difficulty = Difficulty.VeryEasy
    if args.difficulty == "VeryEasy":
        difficulty = Difficulty.VeryEasy
    elif args.difficulty == "Easy":
        difficulty = Difficulty.Easy
    elif args.difficulty == "Medium":
        difficulty = Difficulty.Medium
    elif args.difficulty == "Hard":
        difficulty = Difficulty.Hard
    elif args.difficulty == "VeryHard":
        difficulty = Difficulty.VeryHard
    for i in range(args.num_of_eps):
        print("Episode: ", i)
        sc2.run_game(sc2.maps.get("Simple64"), [
            Bot(Race.Terran, MassReaperBot()),
            Computer(Race.Zerg, difficulty)
        ], realtime=False, random_seed=1)
        episodes.append(i)
        if i % 25 == 0:
            plot_graph3(episodes, rewards)
    print("Training finished.")


def main():
    train()


if __name__ == '__main__':
    main()
