import pygame, sys
from pygame.locals import *
import Space
import time
import Graph
import Space
from multiprocessing import Process

BLACK = (0, 0, 0)
rows = 20
cols = 20
steps = 100
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
height_per_block = WINDOW_HEIGHT // rows
width_per_block = WINDOW_WIDTH // cols

output = False
m = Space.Space(rows, cols, steps, output)


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

def draw(grid):
    for x, i in enumerate(range(rows)):
        for y, j in enumerate(range(cols)):
            rect = pygame.Rect(x*height_per_block, y*height_per_block,
                               height_per_block, height_per_block)
            pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)

if __name__ == "__main__":
    p = Process(target=Graph.graph_process, args=("output.csv", rows*cols, steps))
    p.start()
    viz()
    p.join()

