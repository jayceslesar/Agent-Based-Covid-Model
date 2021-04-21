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
from ipykernel.pickleutil import can
from pygame.locals import *
import Space
import time
import Graph
import Space
import SocialNetwork as sn
from multiprocessing import Process
import SocialNetworkGrapher as sng
import Simulation
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt


BLACK = (0, 0, 0)
rows = 18
cols = 18
steps = 40
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
height_per_block = WINDOW_HEIGHT // rows
width_per_block = WINDOW_WIDTH // cols

output = False
m = Space.Space(rows, cols, steps, output, "random", 1)


def viz():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    while m.steps_taken < m.iterations:
        m._step_()
        draw(m.grid)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        time.sleep(0.2)
    pygame.quit()

    # create social network
    print("making social network...")
    m.social_network = sn.SocialNetwork(m.initial_agent, m.agents)
    # Prints out Social network
    for key in m.social_network.network:
        m.social_network.tracer(key)

    # Finds R0
    total_infected = 0
    total_spreaders = 0
    for i in range(m.rows):
        for j in range(m.cols):
            curr_agent = m.grid[i][j]
            total_infected += curr_agent.total_infected
            if curr_agent.infected or curr_agent.recovered:
                total_spreaders += 1

    r0 = total_infected / total_spreaders
    print("Total infected was " + str(total_infected))
    print("Total spreaders were " + str(total_spreaders))
    print("The R0 for this run was " + str(r0))

    sn_grapher = sng.SocialNetworkGrapher("Title")
    sn_grapher.print_graph(m.social_network_log, m.agents)


def draw(grid):
    for x, i in enumerate(range(rows)):
        for y, j in enumerate(range(cols)):
            rect = pygame.Rect(x*height_per_block, y*height_per_block,
                               height_per_block, height_per_block)
            pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)

def sim():
    fig = plt.figure(figsize=plt.figaspect(0.5))
    rows_s = 8
    cols_s = 8
    steps_s = 20
    output_s = False
    run1 = Simulation.Simulation(rows_s, cols_s, steps_s, output_s, "random", 1, "output1.csv", fig, 1, 2, 1)
    run2 = Simulation.Simulation(rows_s, cols_s, steps_s, output_s, "specific", 1, "output3.csv", fig, 1, 2, 2)
    run3 = Simulation.Simulation(rows_s, cols_s, steps_s, output_s, "smart", 1, "output3.csv", fig, 1, 3, 2)
    run1.run()
    run2.run()
    run3.run()
    plt.show()
    # p1 = Process(target=run1.run, args=())
    # # p2 = Process(target=run2.run, args=())
    # p3 = Process(target=run3.run, args=())
    # p1.start()
    # # p2.start()
    # p3.start()
    # p1.join()
    # # p2.join()
    # p3.join()


if __name__ == "__main__":
    p = Process(target=Graph.graph_process, args=("output.csv", rows*cols, steps))
    p.start()
    viz()
    p.join()

    # # can comment out above and uncomment below to run 3 simultaneous simulations of different types
    # sim()

