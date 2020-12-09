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

class Simulation:
    def __init__(self, rows: int, cols: int, num_steps: int, output: bool, swap_type: str, seed: int, file_name: str, fig: plt.figure, pos_a: int, pos_b: int, pos_c: int):
        self.rows = rows
        self.cols = cols
        self.steps = num_steps
        self.WINDOW_HEIGHT = 800
        self.WINDOW_WIDTH = 800
        self.height_per_block = self.WINDOW_HEIGHT // rows
        self.width_per_block = self.WINDOW_WIDTH // cols
        self.swap_type = swap_type
        self.file_name = file_name
        self.fig = fig
        self.pos_a = pos_a
        self.pos_b = pos_b
        self.pos_c = pos_c

        self.output = output
        self.m = Space.Space(self.rows, self.cols, self.steps, self.output, self.swap_type, seed)

    def viz(self):
        global SCREEN, CLOCK
        BLACK = (0, 0, 0)
        # pygame.init()
        # SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        # CLOCK = pygame.time.Clock()
        # SCREEN.fill(BLACK)

        while self.m.steps_taken < self.m.iterations:
            model = self.m
            model._step_()
            # self.draw(self.m.grid)
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             sys.exit()
        #
        #     pygame.display.update()
        #     time.sleep(0.2)
        # pygame.quit()

        # create social network
        print("making social network...")
        self.m.social_network = sn.SocialNetwork(self.m.initial_agent, self.m.agents)
        # Prints out Social network
        for key in self.m.social_network.network:
            self.m.social_network.tracer(key)

        # Finds R0
        total_infected = 0
        total_spreaders = 0
        for i in range(self.m.rows):
            for j in range(self.m.cols):
                curr_agent = self.m.grid[i][j]
                total_infected += curr_agent.total_infected
                if curr_agent.infected or curr_agent.recovered:
                    total_spreaders += 1

        r0 = total_infected / total_spreaders
        print("Total infected was " + str(total_infected))
        print("Total spreaders were " + str(total_spreaders))
        print("The R0 for this run was " + str(r0))

        title = self.swap_type + " swap"
        sn_grapher = sng.SocialNetworkGrapher(title, self.fig, self.pos_a, self.pos_b, self.pos_c)
        sn_grapher.print_graph(self.m.social_network_log, self.m.agents)


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

