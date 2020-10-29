import Agent


class SocialNetwork:
    def __init__(self, initial_agent: Agent, agents: list):
        # keeps track of initial agent
        # has a dictionary with agents as keys and values of a tuple of
        # the history of infections (tracing everyone who got infected back to the source)
        # and a list of everyone the agent has infected directly
        # the history is an array where index 0 is the source who infected index 1 who infected index 2 and so on
        self.initial_agent = initial_agent
        self.network = {}
        for agent in agents:
            curr_agent = agent
            history = []
            while curr_agent.agent_who_infected_me is not None:
                history.insert(0,curr_agent.agent_who_infected_me)
                curr_agent = curr_agent.agent_who_infected_me
            self.network[agent] = history

    def tracer(self, agent):
        # prints out string that traces how the agent was infected (infection history)
        # if len(self.network[agent][0]) == 0:
        #     string_out = "It was "
        #     for agent in self.network[agent][0]:
        #         string_out += agent.name
        #         string_out += " who infected "
        #     string_out += agent.name
        # elif agent is self.initial_agent:
        #     string_out = "I was patient zero"
        # else:
        #     string_out = "Not applicable, I wasn't infected"
        if len(self.network[agent]) != 0:
            string_out = "It was "
            for agent_c in self.network[agent]:
                string_out += agent_c.number
                string_out += " who infected "
            string_out += agent.number
        elif agent.number == self.initial_agent.number:
            string_out = "I, " + agent.number + ", was patient zero"
        else:
            string_out = "I, " + agent.number + ", was not infected"

        print(string_out)
