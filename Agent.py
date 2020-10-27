class Agent:
    def __init__(self, number: int, row: int, col: int):
        """
        Agent constructor

        Args:
            number (int): the ID of the agent
            row (int): row index
            col (int): col index
        """
        self.number = number
        self.row = row
        self.col = col
        self.untouched = True  # used to track exposure and subsequently infection
        self.infected = False  # used in step function
        self.recovered = False  # for stat tracking
        self.exposed = False  # for stat tracking
        self.num_infected = 0  # for stat tracking
        self.days_exposed = 0  # for stat tracking
        self.days_infected = 0
        self.INFECTED_LENGTH = 15
        self.infectiveness = None  # how likely to infect another
        self.neighborhood_size = None  # how big of a radius can they infect others in
        self.tested_since_last_step = None
        self.lag_from_contact_tracing = None
        self.currently_quarantined = False

        self.agent_who_exposed_me = None
        self.agent_who_infected_me = None
        self.agents_infected = []
        self.total_infected = 0
        self.name = ""


    def __str__(self):
        if self.infected:
            return "I"
        if self.recovered:
            return "R"
        if self.exposed:
            return "E"
        if not self.infected:
            return "O"
        if self.currently_quarantined:
            return "Q"


    def get_color(self):
        if self.infected:  # red
            return (255, 0, 0)
        if self.recovered:  # dark green
            return (61, 99, 17)
        if self.exposed:  # yellow
            return (250, 247, 36)
        if not self.infected:  # red
            return (0, 255, 0)
        if self.currently_quarantined:  # purple
            return (147, 112, 219)