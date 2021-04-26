import numpy as np
import Space
import itertools
import copy
from abc import abstractmethod, ABC
from typing import Tuple, List, Dict, Any, Union
from collections import defaultdict
import random
import pickle


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
        print("We are on step ", space.steps_taken)

    return states

def enumerate_states_str(space: Space.Space) -> list:
    """Enumerates the states of a Space object

    Args:
        space (Space.Space): initial representation of state

    Returns:
        list: all states that could be generated from one or no swaps per step
    """
    start_board = str(space)
    start_step = str(space.steps_taken)
    start_tup = str((start_board, start_step))
    states = {start_tup}
    current_states = []
    current_states.append(space)

    while space.steps_taken < space.iterations:
        next_states = []
        for _states in current_states:
            next_states += get_next_states(_states)
            copy_state = copy.deepcopy(_states)
            copy_state._step_()
            next_states.append(copy_state)
        for states_to_add in next_states:
            temp_tup = (str(states_to_add), str(states_to_add.steps_taken))
            states.add(str(temp_tup))
        current_states = next_states
        print("Size of Next States is : ", len(next_states))
        space._step_()
        print(space.steps_taken, " steps enumerated")

    return_states = list(states)
    return return_states


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

    def get_possible_actions_str(self, rows, cols):
        actions = []
        for row in range(rows):
            for col in range(cols):
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
        self.type = 'Deterministic_Agent'

    def get_action(self, state: Space.Space):
        """picks an action from the list of actions

        Args:
            state (Space.Space): current representation of state (grid)

        Returns:
            actions (List): action of what swap to make that step of the simulation
        """
        for row in range(state.rows):
            for col in range(state.cols):
                if state.grid[row][col].infected:
                    swap_from = (row, col)
                    safe_spaces = []
                    distances = []
                    for row1 in range(state.rows):
                        for col1 in range(state.cols):
                            if state.grid[row][col].untouched or state.grid[row][col].recovered:
                                safe_spaces.append((row1, col1))
                                distances.append(state._calc_distance_(0, 0, row1, col1))
                    try:
                        swap_to = safe_spaces[distances.index(min(distances))]
                        return (swap_from, swap_to)
                    except:
                        return ((0, 0), (0, 0))
        return ((0, 0), (0, 0))



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
            state_str = str((str(state), str(state.steps_taken)))
            if str((state_str, str(action))) in self.prob_dist:
                return_probs.append((action, self.prob_dist[str((state_str, str(action)))]))
            else:
                self.prob_dist[str((state_str, str(action)))] = probability
                return_probs.append((action, probability))

        return return_probs


class Soft_Deterministic_Agent(Deterministic_Agent):
    def __init__(self):
        super().__init__()
        self.type = 'Soft_Deterministic_Agent'
        self.eps = 0.1

    def get_action(self, space: Space.Space):
        # take a random action
        draw = np.random.random()
        if draw < self.eps:
            action = random.choice(self.get_possible_actions(space))
            return action

        # otherwise, take the usual action
        for row in range(space.rows):
            for col in range(space.cols):
                if space.grid[row][col].infected:
                    swap_from = (row, col)
                    safe_spaces = []
                    distances = []
                    for row1 in range(space.rows):
                        for col1 in range(space.cols):
                            if space.grid[row][col].untouched or space.grid[row][col].recovered:
                                safe_spaces.append((row1, col1))
                                distances.append(space._calc_distance_(0, 0, row1, col1))
                    try:
                        swap_to = safe_spaces[distances.index(min(distances))]
                        return (swap_from, swap_to)
                    except:
                        return ((0, 0), (0, 0))
        return ((0, 0), (0, 0))


class RandomAgent(RL_Agent):
    """RandomAgent class for the covid simulation game

    Args:
        RL_Agent (Object): Base class for interacting with the simulation
    """
    def __init__(self):
        """Initialize the instance
        """
        self.type = 'Random_Agent'
        super().__init__()

    def get_action(self, state):
        """picks an action from the list of actions

        Args:
            state (Space.Space): current representation of state (grid)

        Returns:
            actions (List): action of what swap to make that step of the simulation
        """
        if str((str(state), str(state.steps_taken))) in self.action_to_take:
            action = self.action_to_take[str((str(state), str(state.steps_taken)))]
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
        probability = float(1 / num_actions)
        return_probs = []

        for action in actions:
            state_str = str((str(state), str(state.steps_taken)))
            if str((state_str, str(action))) in self.prob_dist:
                return_probs.append((action, self.prob_dist[str((state_str, str(action)))]))
            else:
                self.prob_dist[str((state_str, str(action)))] = probability
                return_probs.append((action, probability))

        return return_probs


class TDAgent(RL_Agent):
    def __init__(self, eps=0.1):
        self.type = 'TD_Agent'
        self.eps = eps
        self.qtable = defaultdict(lambda: 0.13579)
        self.novel = set()
        super().__init__()

    def get_action_str(self, state, rows, cols):
        actions = self.get_possible_actions_str(rows, cols)
        max_q = -1e7
        # initialize with a random action
        if len(actions) == 0:
            return None
        i = np.random.choice(range(len(actions)))
        max_action = actions[i]
        for action in actions:
            q = self.qtable[str((state, str(action)))]
            if q > max_q:
                max_q = q
                max_action = action
        return max_action

    def get_max_action(self, state: Space.Space):
        """Returns the action that has the max q-value"""
        actions = self.get_possible_actions(state)
        max_q = -1e7
        # initialize with a random action
        if len(actions) == 0:
            return None
        i = np.random.choice(range(len(actions)))
        max_action = actions[i]
        for action in actions:
            state_str = str((str(state), str(state.steps_taken)))
            q = self.qtable[str((state_str, str(action)))]
            if q > max_q:
                max_q = q
                max_action = action
        return max_action

    def get_action(self, state: Space.Space):
        actions = self.get_possible_actions(state)
        # to ensure that we don't have some weird ordering bias.
        np.random.shuffle(actions)
        n = len(actions)
        if n == 0:
            return (None)
        max_action = self.get_max_action(state)
        p_other_action = self.eps / n
        p_max_action = p_other_action + (1 - self.eps)

        if max_action == 0.13579:
            self.novel.add(state)

        if np.random.random() < self.eps and n > 1:
            i = np.random.choice(range(n))
            action = actions[i]
            return action
        else:
            return max_action

    def get_prob_dist(self, env: Space.Space):
        pass


def expected_SARSA(state: Space.Space, maxsteps=10000, gamma=0.5, alpha=0.1):

    temp_TD = []
    TD_error = []

    player = TDAgent(eps=1.0)

    copy_env = copy.deepcopy(state)
    diffs = {}  # track the last diffs, for funsies
    step = 0
    env = state
    # grab the first state and action
    s0 = env
    a0 = player.get_action(env)
    # track the total states encountered
    str_state = str((str(s0), str(s0.steps_taken)))
    states = {str_state:1}
    # track the number of episodes
    num_episodes = 1

    while step < maxsteps and max([abs(v) for v in diffs.values()] or [1.0]) > 0.01:
        if num_episodes > 1000 and step % 5e3 == 0 and player.eps > 0.1:
            # scale back the exploration
            player.eps = player.eps / 2

        old_q_lookup_p1 = str((str(s0),str(s0.steps_taken)))
        old_q_lookup_p2 = str(a0)
        old_q_lookup = str((old_q_lookup_p1,old_q_lookup_p2))
        old_q = player.qtable[old_q_lookup]
        r = reward(_set_state(s0, a0))

        if env.steps_taken == env.iterations:
            num_episodes += 1
            # Need to take one step past game over so we can grab
            # the reward from winning or losing.
            # We can think of this like moving into the terminal state,
            # which has value 0 for every action and only transitions to itself.
            new_q = old_q + alpha * (r - old_q)
            # print(s0, a0, r, old_q, new_q)
            new_q_lookup_p1 = str((str(s0), str(s0.steps_taken)))
            new_q_lookup_p2 = str(a0)
            new_q_lookup = str((new_q_lookup_p1, new_q_lookup_p2))
            player.qtable[new_q_lookup] = new_q
            if str((str(env), str(env.steps_taken))) in states:
                states[str((str(env), str(env.steps_taken)))] += 1
            else:
                states[str((str(env), str(env.steps_taken)))] = 1
            env = copy.deepcopy(copy_env)
            s0 = env
            a0 = player.get_action(s0)
            continue# type: ignore
        # soft det. agent returns tuple
        env._RL_agent_swap(a0[0][0], a0[0][1], a0[1][0], a0[1][1])
        env._step_()
        s1 = env

        # Compute the average q value at the next state
        all_actions = player.get_possible_actions(env)
        # we don't want to divide by 1; if there are no free actions,
        # treat noop as the only available action
        n_actions = len(all_actions) or 1
        # initalize with the max value
        max_action = player.get_max_action(env)
        q_lookup_state = str((str(s1), str(s1.steps_taken)))
        q_avg = player.qtable[str((q_lookup_state, str(max_action)))] * (1 - player.eps)
        for action in all_actions:
            q_avg += player.qtable[str((q_lookup_state, str(action)))] * player.eps * (1.0 / n_actions)
        # print('{0} + {1}[{2} + {3} - {0}]'.format(old_q, alpha, r, q_avg))
        new_q = old_q + alpha * (r + (gamma * q_avg) - old_q)
        curr_TD_error = (r + (gamma * q_avg) - old_q)
        temp_TD.append(curr_TD_error)
        if len(temp_TD) > 50:
            td_mean = np.mean(temp_TD)
            temp_TD = []
            TD_error.append(td_mean)
        # print(s0, a0, r, old_q, new_q)
        q_lookup_state = str((str(s0), str(s0.steps_taken)))
        player.qtable[str((q_lookup_state, str(a0)))] = new_q
        diffs[(q_lookup_state, str(a0))] = (new_q - old_q) / (old_q or 1)
        # shift
        if str(s1) in states:
            states[str((str(s1), str(s1.steps_taken)))] += 1
        else:
            states[str((str(s1), str(s1.steps_taken)))] = 1

        s0 = s1
        # we didn't actually take a move, so grab the next action now
        a0 = player.get_action(env)
        step += 1
        print(step, 'step')
        print(a0)
    player.eps = 0.0
    # print(states)
    return player, diffs, states, num_episodes, step, TD_error

def q_learning(state: Space.Space, maxsteps=10000, gamma=0.5, alpha=0.1):
    temp_TD = []
    TD_error = []

    player = TDAgent(eps=1.0)

    copy_env = copy.deepcopy(state)
    diffs = {}  # track the last diffs, for funsies
    step = 0
    env = state
    # grab the first state and action
    s0 = env
    a0 = player.get_action(env)
    # track the total states encountered
    str_state = str((str(s0), str(s0.steps_taken)))
    states = {str_state:1}
    # track the number of episodes
    num_episodes = 1

    while step < maxsteps and max([abs(v) for v in diffs.values()] or [1.0]) > 0.01:
        if num_episodes > 1000 and step % 5e3 == 0 and player.eps > 0.1:
            # scale back the exploration
            player.eps = player.eps / 2

        old_q_lookup_p1 = str((str(s0),str(s0.steps_taken)))
        old_q_lookup_p2 = str(a0)
        old_q_lookup = str((old_q_lookup_p1,old_q_lookup_p2))
        old_q = player.qtable[old_q_lookup]
        r = reward(_set_state(s0, a0))

        if env.steps_taken == env.iterations:
            num_episodes += 1
            # Need to take one step past game over so we can grab
            # the reward from winning or losing.
            # We can think of this like moving into the terminal state,
            # which has value 0 for every action and only transitions to itself.
            new_q = old_q + alpha * (r - old_q)
            # print(s0, a0, r, old_q, new_q)
            new_q_lookup_p1 = str((str(s0), str(s0.steps_taken)))
            new_q_lookup_p2 = str(a0)
            new_q_lookup = str((new_q_lookup_p1, new_q_lookup_p2))
            player.qtable[new_q_lookup] = new_q
            if str((str(env), str(env.steps_taken))) in states:
                states[str((str(env), str(env.steps_taken)))] += 1
            else:
                states[str((str(env), str(env.steps_taken)))] = 1
            env = copy.deepcopy(copy_env)
            s0 = env
            a0 = player.get_action(s0)
            continue# type: ignore
        # soft det. agent returns tuple
        env._RL_agent_swap(a0[0][0], a0[0][1], a0[1][0], a0[1][1])
        env._step_()
        s1 = env

        # Compute the average q value at the next state
        all_actions = player.get_possible_actions(env)
        # we don't want to divide by 1; if there are no free actions,
        # treat noop as the only available action
        n_actions = len(all_actions) or 1

        # initalize with the max value
        max_action = player.get_max_action(env)
        q_lookup_state = str((str(s1), str(s1.steps_taken)))
        q_max = player.qtable[str((q_lookup_state, str(max_action)))]
        # print('{0} + {1}[{2} + {3} - {0}]'.format(old_q, alpha, r, q_avg))
        new_q = old_q + alpha * (r + (gamma * q_max) - old_q)
        curr_TD_error = (r + (gamma * q_max) - old_q)
        temp_TD.append(curr_TD_error)
        if len(temp_TD) > 50:
            td_mean = np.mean(temp_TD)
            temp_TD = []
            TD_error.append(td_mean)
        # print(s0, a0, r, old_q, new_q)
        q_lookup_state = str((str(s0), str(s0.steps_taken)))
        player.qtable[str((q_lookup_state, str(a0)))] = new_q
        diffs[(q_lookup_state, str(a0))] = (new_q - old_q) / (old_q or 1)
        # shift
        if str(s1) in states:
            states[str((str(s1), str(s1.steps_taken)))] += 1
        else:
            states[str((str(s1), str(s1.steps_taken)))] = 1

        s0 = s1
        # we didn't actually take a move, so grab the next action now
        a0 = player.get_action(env)
        step += 1
        print(step, 'step')
        print(a0)
    player.eps = 0.0
    print(states)
    return player, diffs, states, num_episodes, step, TD_error


def policy_evaluation(states: List[Space.Space], policy: RL_Agent, discount: float=0.1) -> Dict[Tuple[Tuple[int], int], int]:
    """Runs the policy evaluation

    Args:
        states (List[Space.Space]): States to look over
        policy (RandomAgent): The policy to use
        discount (float, optional): discount. Defaults to 0.1.

    Returns:
        Dict[Tuple[Tuple[int], int], int]: [description]
    """
    agent_type = policy.type
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

    with open(f'{agent_type}_policy.pickle', 'wb') as handle:
        pickle.dump(value_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return value_dict


def value_iteration(states: List[Space.Space], policy: RL_Agent, discount: float=0.5) -> Tuple[RL_Agent, Dict[Tuple[Tuple[int], int], int]]:
    value_dict = {}
    threshold = 0.1
    agent_type = policy.type
    for state in states:
        # terminal state
        if state.steps_taken == state.iterations:
            state_str = str((str(state), str(state.steps_taken)))
            value_dict[(state_str)] = 0
        else:
            state_str = str((str(state), str(state.steps_taken)))
            value_dict[(state_str)] = 1

    counter = 0
    while True:
        counter += 1

        delta = 0
        for state in states:
            state_str = str((str(state), str(state.steps_taken)))
            v = value_dict[(state_str)]
            # set env to current state to get actions and probabilities for state
            actions_probabilities_state = policy.get_prob_dist(state)

            print('flag 1')

            action_sum_pair = []
            for action, prob in actions_probabilities_state:
                # set env to current state for each loop to get state_prime
                # get all state_primes and their respective probabilities
                state_prime_and_probability = [(_set_state(state, action), 1)]
                for state_prime, state_prime_probability in state_prime_and_probability:
                    r = reward(state_prime)
                    state_prime_str = str((str(state_prime), str(state_prime.steps_taken)))
                    if (state_prime_str) in value_dict:
                        value = value_dict[(state_prime_str)]
                    else:
                        value_dict[(state_prime_str)] = 1
                        value = 1
                    action_sum_pair.append((action, (state_prime_probability * (r + (discount * value)))))
            print('flag 2')
            if len(action_sum_pair) > 0:
                max_action_sum_pair = max(action_sum_pair, key=lambda item: item[1])
                state_str = str((str(state), str(state.steps_taken)))
                value_dict[(state_str)] = max_action_sum_pair[1]

                for action, sum in action_sum_pair:
                    if action == max_action_sum_pair[0]:
                        policy.prob_dist[(state_str, action)] = 1
                    else:
                        policy.prob_dist[(state_str, action)] = 0
            state_str = str((str(state), str(state.steps_taken)))
            delta = max(delta, abs(v - value_dict[(state_str)]))

        print("delta is ", delta)
        if delta < threshold:
            break
        print(counter)

    with open(f'{agent_type}_value.pickle', 'wb') as handle:
        pickle.dump(value_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
