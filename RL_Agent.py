import numpy as np
import Space
import itertools
import copy


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
            spaces.append([row, col])
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


def reward_generator(states: list):
    """Generator object for calculating rewards

    Args:
        states (list): list of states to calculate rewards for

    Yields:
        float: reward of the current state
    """
    for state in states:
        yield reward(state)