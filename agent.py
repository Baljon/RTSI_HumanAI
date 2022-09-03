import numpy as np

# individual setting
# update_up_alfa_human_to_human = 0.03
# update_down_beta_human_to_human = 0.01
#
# update_up_alfa_human_to_AI = 0.007
# update_down_beta_human_to_AI = 0.2
#
# update_up_alfa_AI_to_human = 0.05
# update_down_beta_AI_to_human = 0.01
#
# update_up_alfa_AI_to_AI = 0.004
# update_down_beta_AI_to_AI = 0.064587
#
# initial_trust_human_to_self = 0.5523
# initial_trust_human_to_human = 0.3
# initial_trust_human_to_AI = 0.46
#
# initial_trust_AI_to_self = 0.57
# initial_trust_AI_to_AI = 0.25
# initial_trust_AI_to_human = 0.524


# group setting
initial_trust = 0.5
alfa = 0.01
beta = 0.01

initial_trust_human_to_self, initial_trust_human_to_human, initial_trust_human_to_AI, initial_trust_AI_to_self, initial_trust_AI_to_AI, initial_trust_AI_to_human = initial_trust, initial_trust, initial_trust, initial_trust, initial_trust, initial_trust
update_up_alfa_human_to_human, update_up_alfa_human_to_AI, update_up_alfa_AI_to_human, update_up_alfa_AI_to_AI = alfa, alfa, alfa, alfa
update_down_beta_human_to_human, update_down_beta_human_to_AI, update_down_beta_AI_to_AI, update_down_beta_AI_to_human = beta, beta, beta, beta


class Trust:
    def __init__(self, trust, alfa, beta, floor=0, ceiling=1):
        self.value = trust
        self.alfa = alfa
        self.beta = beta
        self.floor = floor
        self.ceiling = ceiling

    def __str__(self):
        return str(self.value)

    def increase_trust(self):
        self.value = self.value + (self.alfa * (1 - self.value))
        self.check_if_trust_within_range()

    def decrease_trust(self):
        self.value = self.value - (self.beta * self.value)
        self.check_if_trust_within_range()

    def check_if_trust_within_range(self):
        if self.value < self.floor:
            self.value = self.floor
        elif self.value > self.ceiling:
            self.value = self.ceiling


class Agent:
    def __init__(self, number, posx, posy, current_real_temperature_value, random_generator, number_of_agents,
                 agent_type="human"):
        # print(f"A new agent {number} is created.")
        self.number = number
        self.number_of_agents = number_of_agents
        self.random_generator = random_generator
        self.sensor_bias_std = 0.1
        self.sensor_bias_mean = 0.2
        self.posx = posx
        self.posy = posy
        self.selected_neighbours = None
        self.current_reading = None
        self.trust_to_neighbours = None
        self.type = agent_type
        self.current_real_value = current_real_temperature_value
        if self.type == "human":
            self.initial_trust_dict = {"self": initial_trust_human_to_self,
                                       "AI": initial_trust_human_to_AI,
                                       "human": initial_trust_human_to_human}
            self.trust = Trust(initial_trust_human_to_self, update_up_alfa_human_to_human,
                               update_down_beta_human_to_human)
            self.update_up_dict = {"AI": update_up_alfa_human_to_AI,
                                   "human": update_up_alfa_human_to_human}
            self.update_down_dict = {"AI": update_down_beta_human_to_AI,
                                     "human": update_down_beta_human_to_human}
        if self.type == "AI":
            self.initial_trust_dict = {"self": initial_trust_AI_to_self,
                                       "AI": initial_trust_AI_to_AI,
                                       "human": initial_trust_AI_to_human}
            self.trust = Trust(initial_trust_AI_to_self, update_up_alfa_AI_to_AI, update_down_beta_AI_to_AI)
            self.update_up_dict = {"AI": update_up_alfa_AI_to_AI,
                                   "human": update_up_alfa_AI_to_human}
            self.update_down_dict = {"AI": update_down_beta_AI_to_AI,
                                     "human": update_down_beta_AI_to_human}

    def __str__(self):
        text = "Object from the class Agent\n"
        text += "Agent number: " + str(self.number) + "\n"
        text += " Position x = " + str(self.posx) + "\n"
        text += " Position y = " + str(self.posy) + "\n"
        text += " Current real temp in this position = " + str(self.current_real_value) + "\n"
        text += " Current reading from the agent's sensor = " + str(self.current_reading) + "\n"
        return text

    def set_neighbours(self, selected_neighbours):
        trust_to_neighbours = []
        self.selected_neighbours = selected_neighbours
        for agent in self.selected_neighbours:
            trust_to_neighbours.append(Trust(self.initial_trust_dict[agent.type], self.update_up_dict[agent.type],
                                             self.update_down_dict[agent.type]))
        self.trust_to_neighbours = trust_to_neighbours

    def reset_trust(self):
        self.set_neighbours(self.selected_neighbours)

    def set_self_reading(self):
        self.current_reading = self.current_real_value * (
                1 + self.random_generator.normal(self.sensor_bias_mean, self.sensor_bias_std))

    def select_reading(self):
        maybe_expert = self.select_maybe_expert()
        if maybe_expert is None:
            return self.current_reading
        else:
            return maybe_expert.current_reading

    def select_maybe_expert(self):
        trust_values_to_neighbours = [trust.value for trust in self.trust_to_neighbours]
        if self.trust.value >= max(trust_values_to_neighbours):
            return None
        else:
            return self.selected_neighbours[np.argmax(trust_values_to_neighbours)]

    def update_trust(self, avg_grand_truth_signal, avg_reported_signal):
        my_difference = abs(self.current_reading - avg_grand_truth_signal)
        neighbours_differences = [abs(agent.current_reading - avg_grand_truth_signal) for agent in
                                  self.selected_neighbours]
        for i, neighbour_difference in enumerate(neighbours_differences):
            if my_difference < neighbour_difference:
                self.trust.increase_trust()
                self.trust_to_neighbours[i].decrease_trust()
            elif my_difference > neighbour_difference:
                self.trust.decrease_trust()
                self.trust_to_neighbours[i].increase_trust()
            else:
                pass

    def update_trust_2(self, avg_grand_truth_signal, avg_reported_signal_rtsi):
        avg_difference_with_me = abs(avg_grand_truth_signal - avg_reported_signal_rtsi)
        neighbours_differences = []
        for neighbour in self.selected_neighbours:
            avg_reported_signal_without_me = avg_reported_signal_rtsi - (self.current_reading / self.number_of_agents)
            avg_reported_signal_with_neighbour = avg_reported_signal_without_me + (neighbour.select_reading() / self.number_of_agents)
            neighbours_differences.append(abs(avg_grand_truth_signal - avg_reported_signal_with_neighbour))
        for i, avg_difference_with_neighbour in enumerate(neighbours_differences):
            if avg_difference_with_me < avg_difference_with_neighbour:
                self.trust.increase_trust()
                self.trust_to_neighbours[i].decrease_trust()
            elif avg_difference_with_me > avg_difference_with_neighbour:
                self.trust.decrease_trust()
                self.trust_to_neighbours[i].increase_trust()
            else:
                pass
