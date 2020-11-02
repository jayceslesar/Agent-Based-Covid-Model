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

        testx = [1,1,1]
        testy = [1,1,1]
        testz = [1,1,1]

        zdata = [1, 2, 3, 4, 5, 6, 7]
        xdata = [1, 2, 3, 4, 5, 6, 7]
        ydata = [1, 2, 3, 4, 5, 6, 7]
        self.ax.scatter(x_array, y_array, z_array, c='r', marker='o');
        plt.show()
        print(len(x_array))
        print(len(y_array))
        print(len(z_array))
