import pygame, sys
from pygame.locals import *
import Space
import time
import Graph
import Space
import SocialNetwork as sn
from multiprocessing import Process
import SocialNetworkGrapher as sng


BLACK = (0, 0, 0)
rows = 5
cols = 5
steps = 10
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
height_per_block = WINDOW_HEIGHT // rows
width_per_block = WINDOW_WIDTH // cols

output = False
seed = 42
# TODO:: add smart functionality and then add smart
swap_types = ['none', 'random','specific']
models = []
for swap_type in swap_types:
    models.append(Space.Space(rows, cols, steps, output, swap_type, seed))


def headless(m):
    while m.steps_taken < m.iterations:
        m._step_()

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


def viz(m):
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

    sn_grapher = sng.SocialNetworkGrapher()
    sn_grapher.print_graph(m.social_network_log, m.agents)


def draw(grid):
    for x, i in enumerate(range(rows)):
        for y, j in enumerate(range(cols)):
            rect = pygame.Rect(x*height_per_block, y*height_per_block,
                               height_per_block, height_per_block)
            pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)

if __name__ == "__main__":
    p = Process(target=Graph.graph_process, args=("output.csv", rows*cols, steps))
    p.start()
    for model in models:
        headless(model)
    p.join()
