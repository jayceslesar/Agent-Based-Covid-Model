import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import SocialNetwork
import Agent
import SocialNetwork as sn
from multiprocessing import Process
import matplotlib.animation as animation
from matplotlib import style



class SocialNetworkGrapher:
    def __init__(self):
        self.xline = []
        self.yline = []
        self.zline = []
        self.fig = plt.figure(figsize=(20, 20))
        self.ax = plt.axes(projection='3d')
        # note: agent positions stored as (z,y)
        self.agent_position = {}

    def find_agent(self, agent):
        # returns the (z,y) coordinates of the agent
        return self.agent_position[agent]


    def print_graph(self, net_array, agents):
        # initial outline of graph
        # x = iteration number
        # z and y position of the agent is based on the agent number and the number of agents
        # z and y positions of an agent are stored in a dictionary

        max_row_num = int(math.sqrt(len(agents)))
        x_array = []
        y_array = []
        z_array = []

        # fills x_array with iteration values
        for i in range (0,len(net_array)):
            iteration = i + 1
            for c in range (0,len(agents)):
                x_array.append(iteration)

        # fills y and z arrays with relevant values for agent positions
        for agent in agents:
            result = divmod(agent.number,max_row_num)
            z_array.append(result[0])
            y_array.append(result[1])
            self.agent_position[agent] = result
        copy_z = z_array.copy()
        copy_y = y_array.copy()
        for i in range(0, len(net_array) - 1):
            z_array.extend(copy_z)
            y_array.extend(copy_y)

        for agent in agents:
            if agent.iteration_infected is not None:

                # draws a red line from step infected to step recovered
                x_infected_len = [agent.iteration_infected,agent.iteration_infected + agent.days_infected]
                position = self.find_agent(agent)
                y_pos = position[1]
                z_pos = position[0]
                y_infected_len = [y_pos, y_pos]
                z_infected_len = [z_pos, z_pos]
                self.ax.plot3D(x_infected_len, y_infected_len, z_infected_len, 'red')

                # draws a line from infected agent to the victims they infected at relevant steps
                for i_tuple in agent.agents_infected_iterations:
                    victim = i_tuple[0]
                    step = i_tuple[1]
                    victim_coor = self.find_agent(victim)
                    x_victim = [step - 1, step]
                    y_victim = [y_pos, victim_coor[1]]
                    z_victim = [z_pos, victim_coor[0]]
                    self.ax.plot3D(x_victim, y_victim, z_victim, 'red')

        self.ax.scatter(x_array, y_array, z_array, c='g', marker='o')
        self.ax.set_xlabel('Iteration Number')
        plt.show()


