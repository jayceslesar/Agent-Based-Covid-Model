import Agent
from copy import deepcopy
import random as rd
import numpy as np
import Graph
import SocialNetwork as sn


class Space:
    def __init__(self, rows: int, cols: int, num_steps: int, output: bool):
        """
        Space constructor

        Args:
            rows (int): how many rows in the grid
            cols (int): how many cols in the grid
            num_steps (int): total number of steps to take

            O(n)
        """
        self.rows = rows
        self.cols = cols
        self.steps_taken = 0  # tracker to tell what step the model is currently on
        self.iterations = num_steps
        self.log = []  # log to keep track of each time the grid/space is changed
        self.swapped_agents = []  # keeps track of swapped agents so no agents are double swapped
        self.num_agents = [i for i in range(rows*cols)]  # total number of agents
        self.initial_infected = rd.choice(self.num_agents)  # sets one agent to start infected
        self.agents = []
        self.output = output
        self.data = Graph.DataSaver(0, 0, 0, 0, 0, 'output.csv', self.iterations)
        self.curr_number_of_infections = 0  # current number of infections at each step


        # distributions to pick from when building each agent below
        # TODO:: needs a much heavier tail mathematically but it works (normal at mean 2 and sd of 1.5, but take the absolute value and it works out nicely)
        self.INCUBATION_PERIOD_DISTRIBUTION = list(np.absolute(np.around(np.random.normal(loc=2, scale=1.5, size=(rows*cols))).astype(int)))
        # infective length
        self.INFECTIVE_LENGTH_DISTRUBUTION = list(np.around(np.random.normal(loc=10.5, scale=3.5, size=(rows*cols))).astype(int))
        # choose agents to have a pre existing health condition float
        pre_exist_agents = deepcopy(self.num_agents)
        rd.shuffle(pre_exist_agents)
        self.AGENT_NUMBERS_WITH_PRE_EXISTING_CONDITIONS = deepcopy(pre_exist_agents)[:int(rd.uniform(40, 60))]

        # initialized later in program
        self.social_network = None
        self.initial_agent = None


        rows = []
        n = 0
        for i in range(self.rows):
            col = []
            for j in range(self.cols):
                agent = Agent.Agent(n, i, j)
                agent.INCUBATION_PERIOD = self.INCUBATION_PERIOD_DISTRIBUTION.pop(rd.randint(0, len(self.INCUBATION_PERIOD_DISTRIBUTION) - 1))
                agent.INFECTIVE_LENGTH = self.INFECTIVE_LENGTH_DISTRUBUTION.pop(rd.randint(0, len(self.INFECTIVE_LENGTH_DISTRUBUTION) - 1))
                if n in self.AGENT_NUMBERS_WITH_PRE_EXISTING_CONDITIONS:
                    agent.pre_existing_float = rd.uniform(0, 1)
                if n == self.initial_infected:
                    agent.infected = True
                    col.append(agent)
                    self.agents.append(agent)
                    self.initial_agent = agent
                else:
                    col.append(agent)
                    self.agents.append(agent)
                n += 1
            rows.append(col)
        # initiate the grid variable
        self.grid = rows
        # update the log
        self.log.append(self.grid)
        # build a distances dict for distances with neighborhoods
        self.distance_dict = self.calc_distance_dict()


    def calc_distance_dict(self) -> dict:
        """
        creates the distance dictionary for a neighborhood run

        Returns:
            dict: distances for each agent to each agent, where distance from
            a -> b = distance_dict[a.num][b.num]

        O(n^2)
        """
        distance_dict = {}
        for i in range(self.rows):
            for j in range(self.cols):
                agent_distances = {}
                for k in range(self.rows):
                    for l in range(self.cols):
                        dist = self._calc_distance_(i, j, k, l)
                        agent_distances[self.grid[k][l].number] = dist
                distance_dict[self.grid[i][j].number] = agent_distances
        return distance_dict


    def _calc_distance_(self, x1, y1, x2, y2) -> float:
        """
        distance between two points on a linear plane

        Args:
            x1
            y1
            x2
            y2

        Returns:
            float: distance

        O(1)
        """
        return ((x2-x1)**2 + (y2-y1)**2)**0.5


    def _random_swap_(self):
        """
        The function swap will randomly swap two agents with each other

        O(1)
        """
        # find one of the agents to swap
        init_swap_row = rd.randint(0, self.rows - 1)
        init_swap_col = rd.randint(0, self.cols - 1)
        # find the other agent to swap
        to_swap_row = rd.randint(0, self.rows - 1)
        to_swap_col = rd.randint(0, self.cols - 1)
        # make copies of objects
        init_agent = self.grid[init_swap_row][init_swap_col]
        to_agent = self.grid[to_swap_row][to_swap_col]
        # swap the object
        self.grid[init_swap_row][init_swap_col] = to_agent
        self.grid[to_swap_row][to_swap_col] = init_agent

        # update distances dict
        if not self.adjacency:
            self.distance_dict = self.calc_distance_dict()
            if self.output:
                print("distances updated")

        # output
        if self.output:
            print("swapped agent number " + str(self.grid[init_swap_row][init_swap_col].number) +
                " with agent number " + str(self.grid[to_swap_row][to_swap_col].number))


    def _specifc_swap_(self):
        """
        Finds an agent who has yet to be infected and an agent who has recovered and swaps them
        in hopes to reduce or remove the chances that the yet to be infected agent gets infected

        O(n^2)
        """
        safe_spots = []
        untouched_agents = []
        for i in range(self.rows):
            for j in range(self.cols):
                curr_agent = self.grid[i][j]
                # find all untouched agents
                if curr_agent.untouched:
                    untouched_agents.append(curr_agent)
        # find all safe spots -> a point on the grid such that no neighborhoods of currently infected or exposed agents reach
        # must be a recovered spot so that the recovered induvidual can swap with them
        for i in range(self.rows):
            for j in range(self.cols):
                curr_agent = self.grid[i][j]
                if curr_agent.recovered:
                    # assume it is a safe spot
                    safe_spot = True
                    # check each agent against that potential safe spot
                    for k in range(self.rows):
                        for l in range(self.cols):
                            next_curr_agent = self.grid[k][l]
                            # if the agent is infected or exposed
                            if next_curr_agent.infected or next_curr_agent.exposed:
                                # if the distance from exposed/infected agent to recovered agent < neighborhood of exposed/infected agent
                                if self.distance_dict[next_curr_agent.number][curr_agent.number] < next_curr_agent.neighborhood_size:
                                    safe_spot = False
                    if safe_spot:
                        safe_spots.append(curr_agent)
        # make the swaps
        # TODO:: refactor untouched_agents[random_index] with pop
        if len(untouched_agents) > 0:
            for recovered_agent in safe_spots:
                # if we haven't swapped this agent yet
                random_index = rd.randint(0, len(untouched_agents) - 1)
                if recovered_agent not in self.swapped_agents and untouched_agents[random_index] not in self.swapped_agents:
                    # swap the recovered agent with a random untouched agent
                    self.grid[recovered_agent.row][recovered_agent.col] = untouched_agents[random_index]
                    self.grid[untouched_agents[random_index].row][untouched_agents[random_index].col] = recovered_agent
                    # keep track of what agents are swapped for consistency
                    self.swapped_agents.append(untouched_agents[random_index])
                    self.swapped_agents.append(recovered_agent)
                    del untouched_agents[random_index]
        # update distances
        self.distance_dict = self.calc_distance_dict()


    def __str__(self):
        out = ""
        for row in self.grid:
            for agent in row:
                out += str(agent) + " "
            out += "\n"
        return out


    def _neighborhood_infect_(self):
        """
        the neighborhood and radius defined spread run

        O(n^2)
        """
        self.curr_number_of_infections = 0
        new_grid = deepcopy(self.grid)
        for i in range(self.rows):
            for j in range(self.cols):
                agent = self.grid[i][j]
                for k in range(self.rows):
                    for l in range(self.cols):
                        curr_agent = self.grid[k][l]
                        # check against each agent
                        if agent.infected and not agent.recovered:
                            if agent.neighborhood_size >= self.distance_dict[agent.number][curr_agent.number]:
                                if curr_agent.number != agent.number and not curr_agent.exposed and not curr_agent.infected and not curr_agent.recovered:
                                    curr_agent.agent_who_exposed_me = agent
                                    # print(curr_agent.name + " got exposed by " + curr_agent.agent_who_exposed_me.name)
                                    curr_agent.exposed = True
                                    curr_agent.untouched = False
                                    agent.num_infected += 1
                                    self.curr_number_of_infections += 1
                        new_grid[k][l] = curr_agent
        self.grid = new_grid


    def _step_(self):
        """
        runs an iteration of the model

        O(n)
        """
        self._neighborhood_infect_()
        # update based off of infections
        for i in range(self.rows):
            for j in range(self.cols):
                curr_agent = self.grid[i][j]
                # current agent
                if curr_agent.infected:
                    curr_agent.days_infected += 1
                if curr_agent.exposed:
                    curr_agent.days_exposed += 1
                if curr_agent.days_exposed > curr_agent.INCUBATION_PERIOD and not curr_agent.infected and not curr_agent.recovered:
                    curr_agent.infected = True
                    curr_agent.agent_who_infected_me = curr_agent.agent_who_exposed_me
                    # (curr_agent.name + " got infected by " + curr_agent.agent_who_infected_me.name)
                    curr_agent.agent_who_infected_me.total_infected += 1
                    curr_agent.agent_who_infected_me.agents_infected.append(curr_agent)
                    curr_agent.exposed = False
                if curr_agent.days_infected > curr_agent.INFECTIVE_LENGTH:
                    curr_agent.infected = False
                    curr_agent.recovered = True
                self.grid[i][j] = curr_agent

        # reset counts for output
        self.suceptible_count = 0
        self.infected_count = 0
        self.uninfected_count = 0
        self.exposed_count = 0
        self.recovered_count = 0

        # get stats after run
        for i in range(self.rows):
            for j in range(self.cols):
                curr_agent = self.grid[i][j]
                if curr_agent.infected:
                    self.infected_count += 1
                if curr_agent.untouched:
                    self.suceptible_count += 1
                if curr_agent.exposed and not curr_agent.infected and not curr_agent.recovered:
                    self.exposed_count += 1
                if curr_agent.recovered:
                    self.recovered_count += 1


        #update data
        self.data.update_graph(self.suceptible_count, self.infected_count, self.exposed_count, self.recovered_count, 0)

        # swap
        if self.recovered_count > 0:
            self._specifc_swap_()

        # step complete
        self.steps_taken += 1
        # print
        print("step " + str(self.steps_taken) + "/" + str(self.iterations))
        if self.output:
            print("step " + str(self.steps_taken))
            print("suceptible:", self.suceptible_count)
            print("infected:", self.infected_count)
            print("exposed:", self.exposed_count)
            print("recovered:", self.recovered_count)
            print(self.__str__())
        # add to the log of grids in this instance
        self.log.append(self.grid)


    def run(self):
        """
        runs the model until iteration cap
        """
        # print initial state
        if self.output:
            print("starting grid")
            print(self.__str__())
        while self.steps_taken < self.iterations:
            self._step_()

        #create social network
        if self.output:
            print("making social network...")
        sn.SocialNetwork(self.initial_agent, self.agents)
        #Prints out Social network
        for key in self.social_network:
            self.social_network.tracer(key)

        #Finds R0
        total_infected = 0
        total_spreaders = 0
        for i in range(self.rows):
            for j in range(self.cols):
                curr_agent = self.grid[i][j]
                total_infected += curr_agent.num_infected
                if (curr_agent.infected or curr_agent.recovered):
                    total_spreaders += 1

        # r0 = total_infected/total_spreaders
        # if self.output:
        #     print("Total infected was " + str(total_infected))
        #     print("Total spreaders were " + str(total_spreaders))
        #     print("The R0 for this run was " + str(r0))
