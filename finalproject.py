import numpy as np
import Space
import itertools
import copy
from abc import abstractmethod, ABC
from typing import Tuple, List, Dict, Any, Union
from collections import defaultdict
import random


random.seed(42)


def enumerate_states(space: Space.Space) -> list:
    """Enumerates the states of a Space object

    Args:
        space (Space.Space): initial representation of state

    Returns:
        list: all states that could be generated from one or no swaps per step
    """
    states = []
    current_states = []
    current_states.append(space)

    while space.steps_taken < space.iterations:
        next_states = []
        for _states in current_states:
            next_states += get_next_states(_states)
            copy_state = copy.deepcopy(_states)
            copy_state._step_()
            next_states.append(copy_state)
        states += next_states
        current_states = next_states
        space._step_()
    return states


def get_next_states(space: Space.Space) -> Space.Space:
    """Calculates all combinations of swaps for a given state

    Args:
        space (Space.Space): the state represented by a Space object

    Yields:
        Space.Space: yields one next state at a time to save some memory
    """
    spaces = []
    # build coordinates
    for row in range(space.rows):
        for col in range(space.cols):
            spaces.append((row, col))
    # find all unique permutations of swaps
    spaces = list(itertools.combinations(spaces, 2))

    # creaetes acopy of space, runs a step and swaps the agents ad the coordinates
    for swap in spaces:
        new_space = copy.deepcopy(space)
        new_space._step_()
        new_space._RL_agent_swap(swap[0][0], swap[0][1], swap[1][0], swap[1][1])
        yield new_space


def reward(state: Space.Space) -> float:
    """Calculates the reward of a given state

    Args:
        state (Space.Space): the state represented by a Space object

    Returns:
        float: the score (reward) of the state
    """
    score = 0
    for row in range(state.rows):
        for col in range(state.cols):
            if state.grid[row][col].infected:
                score -= 2
            if state.grid[row][col].exposed:
                score -= 1

    return score


def _set_state(state: Space.Space, action: tuple):
        """Applies the action

        Args:
            state (Space.Space): state
            action (tuple): action to apply

        Returns:
            Space.Space: the state with the action applied
        """
        next_state = copy.deepcopy(state)
        next_state._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
        next_state._step_()
        return next_state


def reward_generator(states: list):
    """Generator object for calculating rewards

    Args:
        states (list): list of states to calculate rewards for

    Yields:
        float: reward of the current state
    """
    for state in states:
        yield reward(state)


class RL_Agent(ABC):
    """Abstract base agent class

    Args:
        ABC (Object): Python standard abstract base class
    """
    def __init__(self):
        """Initialize the instance
        """
        self.prob_dist = {}
        self.action_to_take = {}

    @abstractmethod
    def get_action(self, env: Space.Space) -> Tuple[int, int, int]:
        pass

    @abstractmethod
    def get_prob_dist(self, env: Space.Space) -> List[Tuple[Tuple[int, int, int], float]]:
        pass

    def get_possible_actions(self, env: Space.Space) -> List:
        """Gets all the actions possible from one state to the next

        Args:
            env (Space.Space): current state

        Returns:
            actions (List): list of all actions that can be taken
        """
        actions = []
        for row in range(env.rows):
            for col in range(env.cols):
                actions.append((row, col))
        # find all unique permutations of swaps
        actions = list(itertools.combinations(actions, 2))
        return actions


class Deterministic_Agent(RL_Agent):
    """RandomAgent class for the covid simulation game

    Args:
        RL_Agent (Object): Base class for interacting with the simulation
    """
    def __init__(self):
        """Initialize the instance
        """
        super().__init__()

    def get_action(self, state: Space.Space):
        """picks an action from the list of actions

        Args:
            state (Space.Space): current representation of state (grid)

        Returns:
            actions (List): action of what swap to make that step of the simulation
        """
        if state in self.action_to_take:
            action = self.action_to_take[state]
            return action

        actions = self.get_possible_actions(state)
        action = self._determine_best_action(state, actions)
        return action

    def _determine_best_action(self, state: Space.Space, actions: list):
        """Corners infected people by swapping

        Args:
            actions (list): possible actions to take
        """
        rewards = []
        for action in actions:
            copy_state = copy.deepcopy(state)
            rewards.append(reward(_set_state(state, action)))

        max_value = max(rewards)
        max_index = rewards.index(max_value)
        return actions[max_index]

    def get_prob_dist(self, state: Space.Space) -> List[Tuple[Tuple[int, int, int], float]]:
        """Generate the probability distribution for the set of actions

        Args:
            state (Space.Space): current state

        Returns:
            List[Tuple[Tuple[int, int, int], float]]: probabilities for each action to be taken
        """
        actions = self.get_possible_actions(state)
        num_actions = len(actions)
        probability = float(1/num_actions)
        return_probs = []

        for action in actions:
            if (state, action) in self.prob_dist:
                return_probs.append((action, self.prob_dist[(state, action)]))
            else:
                self.prob_dist[(state, action)] = probability
                return_probs.append((action, probability))

        return return_probs


class Soft_Deterministic_Agent(Deterministic_Agent):
    def __init__(self):
        super().__init__()
        self.eps = 0.1

    def get_action(self, space: Space.Space):
        # take a random action
        draw = np.random.random()
        if draw < self.eps:
            action = random.choice(self.get_possible_actions(space))
            return action

        # otherwise, take the usual action
        action = self.get_action(space)
        return action


class RandomAgent(RL_Agent):
    """RandomAgent class for the covid simulation game

    Args:
        RL_Agent (Object): Base class for interacting with the simulation
    """
    def __init__(self):
        """Initialize the instance
        """
        super().__init__()

    def get_action(self, state):
        """picks an action from the list of actions

        Args:
            state (Space.Space): current representation of state (grid)

        Returns:
            actions (List): action of what swap to make that step of the simulation
        """
        if state in self.action_to_take:
            action = self.action_to_take[state]
            return action

        action = random.choice(self.get_possible_actions(state))
        return action

    def get_prob_dist(self, state: Space.Space) -> List[Tuple[Tuple[int, int, int], float]]:
        """Generate the probability distribution for the set of actions

        Args:
            state (Space.Space): current state

        Returns:
            List[Tuple[Tuple[int, int, int], float]]: probabilities for each action to be taken
        """
        actions = self.get_possible_actions(state)
        num_actions = len(actions)
        probability = float(1/num_actions)
        return_probs = []

        for action in actions:
            if (state, action) in self.prob_dist:
                return_probs.append((action, self.prob_dist[(state, action)]))
            else:
                self.prob_dist[(state, action)] = probability
                return_probs.append((action, probability))

        return return_probs


class TDAgent(RL_Agent):
    def __init__(self, eps=0.1):
        self.eps = eps
        self.qtable = {'lambda': 0.13579}
        self.novel = set()
        super().__init__()

    def get_possible_actions(self, state: Space.Space) -> List[Action]:
        actions = super.get_possible_actions(state)
        return actions

    def get_max_action(self, state: Space.Space) -> Action:
        """Returns the action that has the max q-value"""
        actions = self.get_possible_actions(state)
        max_q = -1e7
        # initialize with a random action
        if len(actions) == 0:
            return None
        i = np.random.choice(range(len(actions)))
        max_action = actions[i]
        for action in actions:
            q = self.qtable[(state, action)]
            if q > max_q:
                max_q = q
                max_action = action
        return max_action

    def get_action(self, state: Space.Space) -> Tuple[Action, float]:
        actions = self.get_possible_actions(state)
        # to ensure that we don't have some weird ordering bias.
        np.random.shuffle(actions)
        n = len(actions)
        if n == 0:
            return (None, 1.0)
        max_action = self.get_max_action(state)
        p_other_action = self.eps / n
        p_max_action = p_other_action + (1 - self.eps)

        if max_action == 0.13579:
            self.novel.add(state)

        if np.random.random() < self.eps and n > 1:
            i = np.random.choice(range(n))
            action = actions[i]
            return (action, p_max_action if action == max_action else p_other_action)
        else:
            return (max_action, p_max_action)


def policy_evaluation(states: List[Space.Space], policy: RL_Agent, discount: float=0.1) -> Dict[Tuple[Tuple[int], int], int]:
    """Runs the policy evaluation

    Args:
        states (List[Space.Space]): States to look over
        policy (RandomAgent): The policy to use
        discount (float, optional): discount. Defaults to 0.1.

    Returns:
        Dict[Tuple[Tuple[int], int], int]: [description]
    """
    value_dict = {}
    threshold = 0.1

    for state in states:
        # terminal state
        if state.steps_taken == state.iterations:
            value_dict[(state, 1)] = 0
        else:
            value_dict[(state, 1)] = 1

    while True:
        delta = 0
        for state in states:
            v = value_dict[(state, 1)]
            actions_probabilities_state = policy.get_prob_dist(state)
            for action, prob in actions_probabilities_state:
                total_value_sum = 0
                next_state = copy.deepcopy(state)
                next_state._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
                # get all state_primes and their respective probabilities
                state_prime_and_probability = [(next_state, 1)]
                part_two_sum = 0
                for state_prime, state_prime_probability in state_prime_and_probability:
                    r = reward(next_state)
                    if (state_prime, 1) in value_dict:
                        value = value_dict[(state_prime, 1)]
                    else:
                        value_dict[(state_prime, 1)] = 1
                        value = 1
                    part_two_sum += (state_prime_probability * (r + (discount * value)))

            total_value_sum += (prob * part_two_sum)
            value_dict[(state, 1)] = total_value_sum
            delta = max(delta, abs(v - value_dict[(state, 1)]))

        if delta < threshold:
            break

    return value_dict


def value_iteration(states: List[Space.Space], policy: RL_Agent, discount: float=1.0) -> Tuple[RL_Agent, Dict[Tuple[Tuple[int], int], int]]:
    value_dict = {}
    threshold = 0.1

    for state in states:
        # terminal state
        if state.steps_taken == state.iterations:
            value_dict[(state, 1)] = 0
        else:
            value_dict[(state, 1)] = 1

    counter = 0
    while True:
        counter += 1

        delta = 0
        for state in states:
            v = value_dict[(state, 1)]
            # set env to current state to get actions and probabilities for state
            actions_probabilities_state = policy.get_prob_dist(state)

            action_sum_pair = []
            for action, prob in actions_probabilities_state:
                # set env to current state for each loop to get state_prime
                # get all state_primes and their respective probabilities
                state_prime_and_probability = [(_set_state(state, action), 1)]
                for state_prime, state_prime_probability in state_prime_and_probability:
                    r = reward(state_prime)
                    if (state_prime, 1) in value_dict:
                        value = value_dict[(state_prime, 1)]
                    else:
                        value_dict[(state_prime, 1)] = 1
                        value = 1
                    action_sum_pair.append((action, (state_prime_probability * (r + (discount * value)))))

            if len(action_sum_pair) > 0:
                max_action_sum_pair = max(action_sum_pair, key=lambda item: item[1])
                value_dict[(state, 1)] = max_action_sum_pair[1]

                for action, sum in action_sum_pair:
                    if action == max_action_sum_pair[0]:
                        policy.prob_dist[(state, action)] = 1
                    else:
                        policy.prob_dist[(state, action)] = 0

            delta = max(delta, abs(v - value_dict[(state, 1)]))

        if delta < threshold:
            break

    return (policy, value_dict)


def get_trajectory(env: Space.Space, agent: RL_Agent, discount_factor: float):
    """Tells how well a given agent should do

    Args:
        env (Space.Space): Environment
        agent (RL_Agent): Agent who plays the simulation

    Returns:
        list: list of states and their rewards
    """
    reward_tuples = []

    env_curr = copy.deepcopy(env)

    reward_sum = 0
    while env_curr.steps_taken < env_curr.iterations:
        action = agent.get_action(env_curr)
        env_curr._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
        env_curr._step_()
        reward = reward(env_curr)
        reward_tuples.append((env_curr, action, reward))
        reward_sum = (reward) +  discount_factor * reward_sum

    return reward_tuples, reward_sum
