import matplotlib.pyplot as plt
from multiprocessing import Process
import matplotlib.animation as animation
from matplotlib import style
import csv


class DataSaver:
    def __init__(self, initial_uninfected, initial_infected, initial_exposed, initial_recovered, initial_dead, file_name, num_iterations):
        self.current_iteration = 0
        self.uninfected = [initial_uninfected]
        self.infected = [initial_infected]
        self.exposed = [initial_exposed]
        self.recovered = [initial_recovered]
        self.dead = [initial_dead]
        self.iterations = [0]
        self.file_name = file_name
        self.total_iterations = num_iterations

        # self.p = Process(target=self.draw_graph, args=())
        # self.p.start()
        file = open(file_name, 'w')
        row = str(self.current_iteration) + ',' + str(initial_uninfected) + ',' + str(initial_infected) + ',' + str(initial_exposed) + ',' + str(initial_recovered) + ',' + str(initial_dead) + '\n'
        file.write(row)
        file.close()


    def update_graph(self, updated_uninfected, updated_infected, updated_exposed, updated_recovered, updated_dead):
        file = open(self.file_name, 'a')
        self.current_iteration += 1
        self.iterations.append(self.current_iteration)
        self.uninfected.append(updated_uninfected)
        self.infected.append(updated_infected)
        self.exposed.append(updated_exposed)
        self.recovered.append(updated_recovered)
        self.dead.append(updated_dead)
        row = str(self.current_iteration) + ',' + str(updated_uninfected) + ',' + str(updated_infected) + ',' + str(updated_exposed) + ',' + str(updated_recovered) + ',' + str(updated_dead) + '\n'
        file.write(row)
        file.close()


class Grapher:
    def __init__(self, file_name, num_agents, iterations):
        style.use('fivethirtyeight')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.file_name = file_name
        self.num_agents = num_agents
        self.iterations = iterations


    def animate(self, i):
        with open(self.file_name) as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')

            iterations = []
            u = []
            i = []
            e = []
            r = []
            d = []

            for row in readCSV:
                iterations.append(int(row[0]))
                u.append(int(row[1]))
                i.append(int(row[2]))
                e.append(int(row[3]))
                r.append(int(row[4]))
                d.append(int(row[5]))

        self.ax.clear()
        self.ax.plot(iterations, u)
        self.ax.plot(iterations, i)
        self.ax.plot(iterations, e)
        self.ax.plot(iterations, r)
        plt.title("Live Graph")
        plt.xlabel("Iterations")
        plt.ylabel("Number of People")
        plt.legend(["Succeptible", "Infected", "Exposed", "Recovered"])
        plt.ylim(0,self.num_agents)
        plt.xlim(0,self.iterations)


    def draw_graph(self):
        # interval is the delay between frames
        ani = animation.FuncAnimation(self.fig, self.animate, interval = 20)
        plt.show()


def graph_process(file_name, num_agents, iterations):
    graph = Grapher(file_name, num_agents, iterations)
    graph.draw_graph()
