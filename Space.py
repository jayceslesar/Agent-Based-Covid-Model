import Agent
from copy import deepcopy
import random as rd
import numpy as np
import Graph
import SocialNetwork as sn


class Space:
    def __init__(self, rows: int, cols: int, num_steps: int, output: bool, seed: int):
        rd.seed(seed)
        np.random.seed(seed)
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
        self.curr_number_of_infections = 0
        self.iterations = num_steps
        self.log = []  # log to keep track of each time the grid/space is changed
        self.swapped_agents = []  # keeps track of swapped agents so no agents are double swapped
        self.num_agents = [i for i in range(rows*cols)]  # total number of agents
        self.initial_infected = rd.choice(self.num_agents)  # sets one agent to start infected
        self.agents = []
        self.output = output
        self.data = Graph.DataSaver(0, 0, 0, 0, 0, 'output.csv', self.iterations)
        self.curr_number_of_infections = 0  # current number of infections at each step
        self.curr_iterations = 0

        # distributions to pick from when building each agent below
        # TODO:: needs a much heavier tail mathematically but it works (normal at mean 2 and sd of 1.5, but take the absolute value and it works out nicely)
        self.INCUBATION_PERIOD_DISTRIBUTION = list(np.absolute(np.around(np.random.normal(loc=3, scale=1.5, size=(rows*cols))).astype(int)))
        # infective length
        self.INFECTIVE_LENGTH_DISTRUBUTION = list(np.around(np.random.normal(loc=10.5, scale=3.5, size=(rows*cols))).astype(int))
        # choose agents to have a pre existing health condition float

        # initialized later in program
        self.social_network = None
        self.initial_agent = None

        # social network tracker
        self.social_network_log = []

        rows = []
        n = 0
        for i in range(self.rows):
            col = []
            for j in range(self.cols):
                agent = Agent.Agent(n, i, j)
                agent.INCUBATION_PERIOD = self.INCUBATION_PERIOD_DISTRIBUTION.pop(rd.randint(0, len(self.INCUBATION_PERIOD_DISTRIBUTION) - 1))
                agent.INFECTIVE_LENGTH = self.INFECTIVE_LENGTH_DISTRUBUTION.pop(rd.randint(0, len(self.INFECTIVE_LENGTH_DISTRUBUTION) - 1))
                if n == self.initial_infected:
                    agent.infected = True
                    col.append(agent)
                    self.agents.append(agent)
                    self.initial_agent = agent
                    agent.iteration_infected = 0
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
        init_swap_row = np.random.randint(0, self.rows - 1)
        init_swap_col = np.random.randint(0, self.cols - 1)
        # find the other agent to swap
        to_swap_row = np.random.randint(0, self.rows - 1)
        to_swap_col = np.random.randint(0, self.cols - 1)
        # make copies of objects
        init_agent = self.grid[init_swap_row][init_swap_col]
        to_agent = self.grid[to_swap_row][to_swap_col]
        # swap the object
        self.grid[init_swap_row][init_swap_col] = to_agent
        self.grid[to_swap_row][to_swap_col] = init_agent

        # update distances dict
        self.distance_dict = self.calc_distance_dict()
        if self.output:
            print("distances updated")

        # output
        if self.output:
            print("swapped agent number " + str(self.grid[init_swap_row][init_swap_col].number) +
                " with agent number " + str(self.grid[to_swap_row][to_swap_col].number))

    def _RL_agent_swap(self, init_swap_row, init_swap_col, to_swap_row, to_swap_col):
        init_agent = self.grid[init_swap_row][init_swap_col]
        to_agent = self.grid[to_swap_row][to_swap_col]
        # swap the object
        self.grid[init_swap_row][init_swap_col] = to_agent
        self.grid[to_swap_row][to_swap_col] = init_agent

        # update distances dict
        self.distance_dict = self.calc_distance_dict()

    def _specific_swap_(self):
        """
        Finds an agent who has yet to be infected and an agent who has recovered and swaps them
        in hopes to reduce or remove the chances that the yet to be infected agent gets infected

        O(n^2)
        """
        safe_spots = []
        untouched_agents = []
        # find all safe spots -> a point on the grid such that no neighborhoods of currently infected or exposed agents reach
        # must be a recovered spot so that the recovered induvidual can swap with them
        for i in range(self.rows):
            for j in range(self.cols):
                curr_agent = self.grid[i][j]
                if curr_agent.untouched:
                    untouched_agents.append(curr_agent)
                    continue
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
        if len(untouched_agents) > 0:
            for recovered_agent in safe_spots:
                # if we haven't swapped this agent yet
                if len(untouched_agents) > 0:
                    random_index = rd.randint(0, len(untouched_agents) - 1)
                    curr_untouched_agent = untouched_agents[random_index]
                    if recovered_agent not in self.swapped_agents and curr_untouched_agent not in self.swapped_agents:
                        # swap the recovered agent with a random untouched agent
                        self.grid[recovered_agent.row][recovered_agent.col] = curr_untouched_agent
                        self.grid[curr_untouched_agent.row][curr_untouched_agent.col] = recovered_agent
                        # keep track of what agents are swapped for consistency
                        self.swapped_agents.append(curr_untouched_agent)
                        self.swapped_agents.append(recovered_agent)
                        del untouched_agents[random_index]
        # update distances
        self.distance_dict = self.calc_distance_dict()

    def _smart_swap_(self):
        """
        use distance matrix to find "next infections" and move the recovered agents there
        do the same in the specific swap but keep a list of the n recovered_swapped_to_points
        and the initial_infection_origin
        find the m closest_exposed_or_infected person to the initial_infection_origin
        swap all the n available recovered_swapped_to_points to the m closest_exposed_or_infected in relation
        to the most recent swap -? put swaps together or try to bin them essentially
        keep track of all swaps so that no double swaps are made
        """
        safe_spots = []
        swappable_agents_distances = {}
        # find all safe spots -> a point on the grid such that no neighborhoods of currently infected or exposed agents reach
        # must be a recovered spot so that the recovered induvidual can swap with them
        for i in range(self.rows):
            for j in range(self.cols):
                curr_agent = self.grid[i][j]
                if curr_agent.untouched:
                    if curr_agent not in self.swapped_agents:
                        swappable_agents_distances[curr_agent.number] = self.distance_dict[self.initial_agent.number][curr_agent.number]
                    continue
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
        sorted_swappable_agents_distances = {k: v for k, v in sorted(swappable_agents_distances.items(), key=lambda item: item[1])}
        # make the swaps
        for recovered_agent in safe_spots:
            # if we haven't swapped this agent yet
            try:
                next_to_swap = next(iter(sorted_swappable_agents_distances))
            except StopIteration:
                continue
            curr_untouched_agent = self.agents[next_to_swap]
            curr_untouched_agent.need_to_see = True
            if recovered_agent not in self.swapped_agents and curr_untouched_agent not in self.swapped_agents:
                # swap the recovered agent with a random untouched agent
                self.grid[recovered_agent.row][recovered_agent.col] = curr_untouched_agent
                self.grid[curr_untouched_agent.row][curr_untouched_agent.col] = recovered_agent
                # keep track of what agents are swapped for consistency
                self.swapped_agents.append(curr_untouched_agent)
                self.swapped_agents.append(recovered_agent)
        # update distances
        self.distance_dict = self.calc_distance_dict()


    def __str__(self):
        out = ""
        for row in self.grid:
            for agent in row:
                out += agent.__str__() + " "
            out += "\n"
        return out

    def _neighborhood_infect_(self):
        """
        the neighborhood and radius defined spread run

        O(n^2)
        """
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
                                if curr_agent.number != agent.number and not curr_agent.exposed and not curr_agent.infected and not curr_agent.recovered and not curr_agent.pre_exposed:
                                    curr_agent.agent_who_exposed_me = agent
                                    # print(curr_agent.name + " got exposed by " + curr_agent.agent_who_exposed_me.name)
                                    curr_agent.days_pre_exposed += 1
                                    curr_agent.untouched = False
                                    agent.num_infected += 1
                                    self.curr_number_of_infections += 1
                            else:
                                curr_agent.days_pre_exposed -= 1
                        new_grid[k][l] = curr_agent
        self.grid = new_grid

    def _step_(self):
        """
        runs an iteration of the model

        O(n)
        """
        # reset current number of infections
        self.curr_number_of_infections = 0
        self.curr_iterations += 1
        self._neighborhood_infect_()
        # update based off of infections
        for i in range(self.rows):
            for j in range(self.cols):
                curr_agent = self.grid[i][j]
                # current agent
                if curr_agent.days_pre_exposed > 1:
                    curr_agent.exposed = True
                if curr_agent.infected:
                    curr_agent.days_infected += 1
                if curr_agent.exposed:
                    curr_agent.days_exposed += 1
                if curr_agent.days_exposed > curr_agent.INCUBATION_PERIOD and not curr_agent.infected and not curr_agent.recovered:
                    curr_agent.infected = True
                    curr_agent.iteration_infected = self.curr_iterations
                    curr_agent.agent_who_infected_me = curr_agent.agent_who_exposed_me
                    curr_agent.agent_who_infected_me.total_infected += 1
                    curr_agent.agent_who_infected_me.agents_infected.append(curr_agent)
                    curr_agent.agent_who_infected_me.agents_infected_iterations.append((curr_agent, self.curr_iterations))
                    curr_agent.exposed = False
                if curr_agent.days_infected > curr_agent.INFECTIVE_LENGTH and curr_agent.infected:
                    curr_agent.infected = False
                    curr_agent.recovered = True
                    curr_agent.iteration_recovered = self.curr_iterations
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

        # update social_network_log
        entry = sn.SocialNetwork(self.initial_agent, self.agents)
        self.social_network_log.append(entry)

        # swap
        # if self.recovered_count > 0:
        #     if self.swap_type == 'random':
        #         self._random_swap_()
        #     elif self.swap_type == 'specific':
        #         self._specific_swap_()
        #     elif self.swap_type == 'smart':
        #         self._smart_swap_()

        # step complete
        self.steps_taken += 1
        # print
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
        self.social_network = sn.SocialNetwork(self.initial_agent, self.agents)
        #Prints out Social network
        for key in self.social_network.network:
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
