import numpy as np
import Space
import itertools
import copy
from abc import abstractmethod, ABC
from typing import Tuple, List, Dict, Any, Union
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
        next_state.step()
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
        action = _determine_best_action(actions)
        return action

    def _determine_best_action(self, actions: list):
        """Corners infected people by swapping

        Args:
            actions (list): possible actions to take
        """
        action_rewards = reward_generator(actions)
        max_value = max(action_rewards)
        max_index = actions.index(max_value)
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


class RandomAgent(RL_Agent):
    """RandomAgent class for the covid simulation game

    Args:
        RL_Agent (Object): Base class for interacting with the simulation
    """
    def __init__(self):
        """Initialize the instance
        """
        super().__init__()

    def get_action(self, state=None):
        """picks an action from the list of actions

        Args:
            state (Space.Space): current representation of state (grid)

        Returns:
            actions (List): action of what swap to make that step of the simulation
        """
        if state in self.action_to_take:
            action = self.action_to_take[state]
            return action

        action = random.choice(RL_Agent.get_possible_actions(state))
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


def policy_evaluation(states: List[Space.Space], policy: RL_Agent, discount: float = 0.1) -> Dict[Tuple[Tuple[int], int], int]:
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