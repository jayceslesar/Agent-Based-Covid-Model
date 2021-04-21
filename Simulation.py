"""
Authors:
---
    Jayce Slesar
    Brandon Lee

Date:
---
    10/15/2020
"""
import pygame, sys
import matplotlib.pyplot as plt
from pygame.locals import *
import time
import Graph
import SocialNetwork as sn
from multiprocessing import Process
import SocialNetworkGrapher as sng
from finalproject import enumerate_states, get_next_states, reward_generator, policy_evaluation, value_iteration, RandomAgent, Deterministic_Agent, Soft_Deterministic_Agent, RL_Agent, expected_SARSA, q_learning
import copy
import pickle
import numpy as np
import json
import Space
import pandas as pd


class Simulation:
    def __init__(self, rows: int, cols: int, num_steps: int, output: bool, seed: int, states):
        self.rows = rows
        self.cols = cols
        self.steps = num_steps
        self.WINDOW_HEIGHT = 800
        self.WINDOW_WIDTH = 800
        self.height_per_block = self.WINDOW_HEIGHT // rows
        self.width_per_block = self.WINDOW_WIDTH // cols
        self.swap_type = swap_type
        self.seed = seed
        self.states = states

        self.output = output
        self.m = Space.Space(self.rows, self.cols, self.steps, self.output, self.seed)
        self.scores = []

    def play_sim(self, agent: RL_Agent, times):
        for i in range(times):
            enum_m = copy.deepcopy(self.m)
            while enum_m.steps_taken < enum_m.iterations:
                action = agent.get_action(enum_m)
                enum_m._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
                enum_m._step_()

            self.scores.append(enum_m.infected_count + enum_m.recovered_count)


        # global SCREEN, CLOCK
        # BLACK = (0, 0, 0)
        # pygame.init()
        # SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        # CLOCK = pygame.time.Clock()
        # SCREEN.fill(BLACK)

        # m = copy.deepcopy(self.m)
        # while m.steps_taken < m.iterations:
        #     action = agent.get_action(m)
        #     m._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
        #     m._step_()
        #     self.draw(m.grid)
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             sys.exit()
        #     pygame.display.update()
        #     time.sleep(3)
        # pygame.quit()

    def draw(self, grid):
        for x, i in enumerate(range(self.rows)):
            for y, j in enumerate(range(self.cols)):
                rect = pygame.Rect(x*self.height_per_block, y*self.height_per_block,
                                   self.height_per_block, self.height_per_block)
                pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)


if __name__ == '__main__':
    rows = 10
    cols = 10
    num_steps = 15
    output = False
    swap_type = 'none'
    seed = 42
    test_space = Space.Space(10, 10, 30, output, seed)
    player, diffs, sarsa_states, num_episodes, step, TD_error = expected_SARSA(copy.deepcopy(test_space))

    print(len(TD_error))

    my_df = pd.DataFrame()
    my_df['TD_error'] = TD_error
    my_df.to_csv('some_csv.csv')

    print(len(sarsa_states))
    print(num_episodes)
    # states = enumerate_states(test_space)
    # print(len(states))
    # states = enumerate_states(copy.deepyop(test_space))
    # times = 100

    # with open(f'states_{4}_{4}_{5}.pickle', 'wb') as handle:
    #     pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # rand = RandomAgent()
    # det = Deterministic_Agent()
    # soft = Soft_Deterministic_Agent()

    # val_iter_rand_agent = RandomAgent()
    # rand_agent_value_iteration = value_iteration(states, val_iter_rand_agent)


    # rand_sim = Simulation(rows, cols, num_steps, output, seed, [])
    # agents_scores_dict = {}
    # # random
    # print('random')
    # rand_sim.play_sim(rand, times)
    # scores = rand_sim.scores
    # print(f'random mean: {np.mean(scores):.2f}, random stdv: {np.std(scores):.2f}')
    # rand_mean = round(np.mean(scores), 3)
    # rand_std = round(np.std(scores), 3)
    # agents_scores_dict['rand_mean'] = rand_mean
    # agents_scores_dict['rand_std'] = rand_std

    # soft_sim = Simulation(rows, cols, num_steps, output, seed, [])
    # # random
    # print('soft')
    # soft_sim.play_sim(soft, times)
    # scores = soft_sim.scores
    # print(f'soft mean: {np.mean(scores):.2f}, soft stdv: {np.std(scores):.2f}')
    # soft_mean = round(np.mean(scores), 3)
    # soft_std = round(np.std(scores), 3)
    # agents_scores_dict['soft_mean'] = soft_mean
    # agents_scores_dict['soft_std'] = soft_std

    # det_sim = Simulation(rows, cols, num_steps, output, seed, [])
    # # random
    # print('det')
    # det_sim.play_sim(det, times)
    # scores = det_sim.scores
    # print(f'det mean: {np.mean(scores):.2f}, det stdv: {np.std(scores):.2f}')
    # det_mean = round(np.mean(scores), 3)
    # det_std = round(np.std(scores), 3)
    # agents_scores_dict['det_mean'] = det_mean
    # agents_scores_dict['det_std'] = det_std

    # with open('agentscores.json', 'w') as json_file:
    #     json.dump(agents_scores_dict, json_file)
