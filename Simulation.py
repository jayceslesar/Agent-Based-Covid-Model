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
import Space
import SocialNetwork as sn
from multiprocessing import Process
import SocialNetworkGrapher as sng
from RL_Agent import enumerate_states, get_next_states, reward_generator, policy_evaluation, RandomAgent
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

        enum_m = copy.deepcopy(self.m)
        states = enumerate_states(enum_m)
        v1 = policy_evaluation(states, RandomAgent())
        for s, r in v1:
            print(r)


    def viz(self):
        global SCREEN, CLOCK
        BLACK = (0, 0, 0)
        pygame.init()
        SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        CLOCK = pygame.time.Clock()
        SCREEN.fill(BLACK)

        while self.m.steps_taken < self.m.iterations:
            model = self.m
            model._step_()
            self.draw(self.m.grid)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()
            time.sleep(0.2)
        pygame.quit()

    def draw(self, grid):
        for x, i in enumerate(range(self.rows)):
            for y, j in enumerate(range(self.cols)):
                rect = pygame.Rect(x*self.height_per_block, y*self.height_per_block,
                                   self.height_per_block, self.height_per_block)
                pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)

    def run(self):
        # p = Process(target=Graph.graph_process, args=(self.file_name, self.rows*self.cols, self.steps))
        # p.start()
        self.viz()
        # p.join()

if __name__ == '__main__':
    rows = 3
    cols = 3
    num_steps = 3
    output = False
    swap_type = 'none'
    seed = 42
    sim = Simulation(rows, cols, num_steps, output, swap_type, seed)
    sim.run()