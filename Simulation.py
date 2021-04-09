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
import Space
import time
import Graph
import SocialNetwork as sn
from multiprocessing import Process
import SocialNetworkGrapher as sng
from finalproject import enumerate_states, get_next_states, reward_generator, policy_evaluation, RandomAgent, Deterministic_Agent, RL_Agent
import copy

class Simulation:
    def __init__(self, rows: int, cols: int, num_steps: int, output: bool, swap_type: str, seed: int):
        self.rows = rows
        self.cols = cols
        self.steps = num_steps
        self.WINDOW_HEIGHT = 800
        self.WINDOW_WIDTH = 800
        self.height_per_block = self.WINDOW_HEIGHT // rows
        self.width_per_block = self.WINDOW_WIDTH // cols
        self.swap_type = swap_type

        self.output = output
        self.m = Space.Space(self.rows, self.cols, self.steps, self.output, self.swap_type, seed)

    def play_sim(self, agent: RL_Agent):
        enum_m = copy.deepcopy(self.m)
        states = enumerate_states(enum_m)
        print('states calculated....')
        v1 = policy_evaluation(states, agent)

        global SCREEN, CLOCK
        BLACK = (0, 0, 0)
        pygame.init()
        SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        CLOCK = pygame.time.Clock()
        SCREEN.fill(BLACK)

        while self.m.steps_taken < self.m.iterations:
            m = self.m
            action = agent.get_action(m)
            m._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
            m._step_()
            self.draw(m.grid)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()
            time.sleep(3)
        pygame.quit()

    def draw(self, grid):
        for x, i in enumerate(range(self.rows)):
            for y, j in enumerate(range(self.cols)):
                rect = pygame.Rect(x*self.height_per_block, y*self.height_per_block,
                                   self.height_per_block, self.height_per_block)
                pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)


if __name__ == '__main__':
    rows = 2
    cols = 2
    num_steps = 3
    output = False
    swap_type = 'none'
    seed = 42
    rand_sim = Simulation(rows, cols, num_steps, output, swap_type, seed)
    # random
    print('random')
    rand = RandomAgent()
    rand_sim.play_sim(rand)
    # det
    det_sim = Simulation(rows, cols, num_steps, output, swap_type, seed)
    print('deterministic')
    det = Deterministic_Agent()
    det_sim.play_sim(det)