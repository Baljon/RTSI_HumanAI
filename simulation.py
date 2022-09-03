from collections import defaultdict
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
from agent import Agent


class Simulation:
    def __init__(self, the_same_initial_temp_everywhere=False, is_bidirectional=False, periods=40, agents=40, neighbours=5, grid_state=0, random_state=0):
        print("A new simulation is started.")
        # Parameters
        self.is_bidirectional = is_bidirectional
        self.grid_maxX = 100  # grid of cells of size [1..maxX, 1...maxY]
        self.grid_maxY = 100
        self.signal_x_range = (-2, 2)
        self.signal_y_range = (-2, 2)
        self.human_agents_on_meshgrid = None
        self.AI_agents_on_meshgrid = None
        self.number_of_periods = periods  # number of simulation steps
        self.current_period = 0
        self.number_of_agents = agents  # maximal number of agents
        self.number_of_human_agents = self.number_of_agents // 2
        self.number_of_AI_agents = self.number_of_agents // 2
        self.number_of_agents_to_select = neighbours  # the number of neighbours
        self.the_same_initial_temp_everywhere = the_same_initial_temp_everywhere

        self.signal_meshgrid = None

        self.sensor_signals_over_time_WoC = None  # signal readings by agents' sensors x periods
        self.sensor_signals_over_time_WoC_human = None
        self.sensor_signals_over_time_WoC_AI = None
        self.reported_signal_over_time_RTSI = None  # reported signal by agents x periods
        self.reported_signal_over_time_RTSI_human = None
        self.reported_signal_over_time_RTSI_AI = None

        self.avg_signal_reported_by_agents_over_time_RTSI = None
        self.avg_signal_reported_by_agents_over_time_RTSI_human = None
        self.avg_signal_reported_by_agents_over_time_RTSI_AI = None
        self.avg_signal_from_agents_sensors_over_time_WoC = None
        self.avg_signal_from_agents_sensors_over_time_WoC_human = None
        self.avg_signal_from_agents_sensors_over_time_WoC_AI = None
        self.avg_grand_truth_signal_over_time = None

        self.error_over_time_RTSI = None
        self.error_over_time_RTSI_human = None
        self.error_over_time_RTSI_AI = None
        self.error_over_time_WoC = None
        self.error_over_time_WoC_human = None
        self.error_over_time_WoC_AI = None

        self.experts_human = None
        self.experts_AI = None
        self.experts_grouped_by_type = None
        self.number_of_readings_taken_by_humans_from_humans = None
        self.number_of_readings_taken_by_humans_from_AI = None
        self.number_of_readings_taken_by_AI_from_humans = None
        self.number_of_readings_taken_by_AI_from_AI = None
        self.number_of_readings_taken_by_all_from_AI = None
        self.number_of_readings_taken_by_all_from_humans = None

        self.grid_rng = np.random.default_rng(grid_state)
        self.rng = np.random.default_rng(random_state)
        self.initialize_meshgrid()
        self.agents = []
        self.initialize_agents()

    def __str__(self):
        return 'sim'

    def initialize_agents(self):
        self.human_agents_on_meshgrid = np.zeros((self.grid_maxX, self.grid_maxY))
        self.AI_agents_on_meshgrid = np.zeros((self.grid_maxX, self.grid_maxY))
        self.place_agents(self.number_of_human_agents, "human", 0)
        self.place_agents(self.number_of_AI_agents, "AI", self.number_of_human_agents)
        # self.place_agents(self.number_of_AI_agents, "AI", 0)
        # self.place_agents(self.number_of_human_agents, "human", self.number_of_AI_agents)
        self.set_agents_neighbours()

    def place_agents(self, agents_number, agents_type, starting_number):
        if agents_type == "human":
            agents_meshgrid = self.human_agents_on_meshgrid
        else:
            agents_meshgrid = self.AI_agents_on_meshgrid
        agent_index = starting_number
        print(f'Creating {agents_number} { agents_type} agents')
        for i in range(agents_number):
            while True:
                x_grid = self.grid_rng.integers(0, self.grid_maxX)
                y_grid = self.grid_rng.integers(0, self.grid_maxY)
                if agents_meshgrid[y_grid][x_grid] == 1:
                    continue
                self.agents.append(
                    Agent(number=agent_index,
                          posx=x_grid,
                          posy=y_grid,
                          current_real_temperature_value=self.signal_meshgrid[x_grid][y_grid],
                          random_generator=self.rng,
                          number_of_agents=self.number_of_agents,
                          agent_type=agents_type)
                )
                break
            agents_meshgrid[y_grid][x_grid] += 1
            agent_index += 1

    def set_agents_neighbours(self):
        neighbours_dict = defaultdict(list)
        for agent in self.agents:
            while len(neighbours_dict[agent.number]) < self.number_of_agents_to_select:
                new_neighbour = self.grid_rng.choice(self.agents, 1)[0]
                neighbours_dict[agent.number].append(new_neighbour)
                if self.is_bidirectional:
                    neighbours_dict[new_neighbour.number].append(agent)
        for agent in self.agents:
            agent.set_neighbours(neighbours_dict[agent.number])

    def initialize_arrays(self):
        self.sensor_signals_over_time_WoC = np.zeros((self.number_of_periods, self.number_of_agents))
        self.reported_signal_over_time_RTSI = np.zeros((self.number_of_periods, self.number_of_agents))
        self.sensor_signals_over_time_WoC_human = np.zeros((self.number_of_periods, self.number_of_human_agents))
        self.reported_signal_over_time_RTSI_human = np.zeros((self.number_of_periods, self.number_of_human_agents))
        self.sensor_signals_over_time_WoC_AI = np.zeros((self.number_of_periods, self.number_of_AI_agents))
        self.reported_signal_over_time_RTSI_AI = np.zeros((self.number_of_periods, self.number_of_AI_agents))
        self.avg_grand_truth_signal_over_time = np.zeros(self.number_of_periods)
        self.error_over_time_RTSI = np.zeros(self.number_of_periods)
        self.error_over_time_WoC = np.zeros(self.number_of_periods)
        self.error_over_time_RTSI_human = np.zeros(self.number_of_periods)
        self.error_over_time_RTSI_AI = np.zeros(self.number_of_periods)
        self.error_over_time_WoC_human = np.zeros(self.number_of_periods)
        self.error_over_time_WoC_AI = np.zeros(self.number_of_periods)
        self.experts_human = np.zeros(self.number_of_periods)
        self.experts_AI = np.zeros(self.number_of_periods)
        self.number_of_readings_taken_by_humans_from_humans = np.zeros(self.number_of_periods)
        self.number_of_readings_taken_by_humans_from_AI = np.zeros(self.number_of_periods)
        self.number_of_readings_taken_by_AI_from_humans = np.zeros(self.number_of_periods)
        self.number_of_readings_taken_by_AI_from_AI = np.zeros(self.number_of_periods)
        self.number_of_readings_taken_by_all_from_AI = np.zeros(self.number_of_periods)
        self.number_of_readings_taken_by_all_from_humans = np.zeros(self.number_of_periods)
        self.experts_grouped_by_type = []

    def initialize_signals_sensed_by_agents_lists(self):
        #  Wisdom of Crowd – standard average
        self.avg_signal_from_agents_sensors_over_time_WoC = np.zeros(self.number_of_periods)
        self.avg_signal_from_agents_sensors_over_time_WoC_human = np.zeros(self.number_of_periods)
        self.avg_signal_from_agents_sensors_over_time_WoC_AI = np.zeros(self.number_of_periods)

        #  RTSI model – as perceived by agents
        self.avg_signal_reported_by_agents_over_time_RTSI = np.zeros(self.number_of_periods)
        self.avg_signal_reported_by_agents_over_time_RTSI_human = np.zeros(self.number_of_periods)
        self.avg_signal_reported_by_agents_over_time_RTSI_AI = np.zeros(self.number_of_periods)

    def initialize_meshgrid(self):
        x = np.linspace(self.signal_x_range[0] + 0.5, self.signal_x_range[1] - 0.5, self.grid_maxX)
        y = np.linspace(self.signal_y_range[0] + 0.5, self.signal_y_range[1] - 0.5, self.grid_maxY)
        xx, yy = np.meshgrid(x, y)
        self.signal_meshgrid = 5 * (0.25 + xx / 2. + xx ** 3 + yy ** 5) * np.exp(-xx ** 2 - yy ** 2)
        if self.the_same_initial_temp_everywhere:
            self.signal_meshgrid = np.ones(self.signal_meshgrid.shape)

    def draw_chart_agents(self, plot_name):
        # we want to have 0 in the middle of the chart
        range_min, range_max = -np.abs(self.signal_meshgrid).max(), np.abs(self.signal_meshgrid).max()
        fig, ax = plt.subplots()
        # draw temperatures on meshgrid
        x = np.linspace(self.signal_x_range[0], self.signal_x_range[1], self.grid_maxX)
        y = np.linspace(self.signal_y_range[0], self.signal_y_range[1], self.grid_maxY)
        xx, yy = np.meshgrid(x, y)
        c = ax.pcolormesh(xx, yy, self.signal_meshgrid, cmap='coolwarm', vmin=range_min, vmax=range_max)

        # ax.set_title('Agents')

        # setting x,y labels
        # locs_x, labels_x = plt.xticks()
        # print(locs_x)
        grid_labels = ["1", "25", "50", "75", "100"]
        plt.xticks([-2, -1, 0, 1, 2], grid_labels)
        plt.yticks([-2, -1, 0, 1, 2], grid_labels)

        # set the limits of the plot to the limits of the data
        ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])
        fig.colorbar(c, ax=ax)

        # draw connections based on trust
        # for agent in self.agents:
        #     for i, neighbour in enumerate(agent.selected_neighbours):
        #         width = agent.trust_to_neighbours[i].value
        #         plt.plot([self.grid_to_plane(agent.posx, self.signal_x_range[1]),
        #                   self.grid_to_plane(neighbour.posx, self.signal_x_range[1])],
        #                  [self.grid_to_plane(agent.posy, self.signal_y_range[1]),
        #                   self.grid_to_plane(neighbour.posy, self.signal_y_range[1])], linewidth=width * 0.6,
        #                  color='black', alpha=width, zorder=1)

        # draw neighbours
        # for agent in self.agents:
        #     for i, neighbour in enumerate(agent.selected_neighbours):
        #         plt.plot([self.grid_to_plane(agent.posx, self.signal_x_range[1]), self.grid_to_plane(neighbour.posx, self.signal_x_range[1])],
        #                  [self.grid_to_plane(agent.posy, self.signal_y_range[1]), self.grid_to_plane(neighbour.posy, self.signal_y_range[1])], linewidth=0.5,
        #                  color='black', alpha=0.5, zorder=1)

        # draw agents
        x = np.linspace(self.signal_x_range[0], self.signal_x_range[1], self.grid_maxX)
        y = np.linspace(self.signal_y_range[0], self.signal_y_range[1], self.grid_maxY)
        xx, yy = np.meshgrid(x, y)
        ax.scatter(xx, yy, self.human_agents_on_meshgrid * 20, c='black', marker="o", label="Human agents", zorder=2)
        ax.scatter(xx, yy, self.AI_agents_on_meshgrid * 20, c='white', edgecolors='black', marker="o",
                   label="AI agents", zorder=100)

        # plotting legend below the chart
        # plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), fancybox=True,
        #             borderaxespad=0, ncol=3)
        # plt.subplots_adjust(bottom=0.15)

        plt.savefig(f'images/{plot_name}.png')
        plt.show()

        print("\n")
        print("The figure is created.")
        print("\n")

        print("max temp in the entire plane = " + str(np.ndarray.max(self.signal_meshgrid)))
        print("min temp in the entire plane = " + str(np.ndarray.min(self.signal_meshgrid)))
        print("mean temp in the entire plane = " + str(np.ndarray.mean(self.signal_meshgrid)))

    # maxY and maxX should be the same, or it will not work
    def grid_to_plane(self, coord, plane_max):
        return (coord - (self.grid_maxY / 2)) * plane_max / (self.grid_maxY / 2)

    def draw_chart_readings_over_time(self, cond):
        plt.plot(self.avg_signal_from_agents_sensors_over_time_WoC, label="WoC – avg signal from agents sensors",
                 color="blue")
        plt.plot(self.avg_grand_truth_signal_over_time, label="GT – avg ground truth signal", color="black")
        plt.plot(self.avg_signal_reported_by_agents_over_time_RTSI, label="RTSI – avg reported signal by agents",
                 color="green")
        # plt.title(f'({cond}) Readings over time')
        plt.xlabel('Time')
        plt.ylabel('Signal value')
        plt.ylim(ymin=0, ymax=1)
        plt.legend()
        plt.savefig(f'images/Single_run_{cond}_2_Readings_over_time.png')
        plt.show()

    def draw_chart_readings_over_time_separate_types(self, cond):
        plt.plot(self.avg_grand_truth_signal_over_time, label="GT – avg ground truth signal", color="black")
        plt.plot(self.avg_signal_from_agents_sensors_over_time_WoC_human, label="WoC human", color="#8080ff")
        plt.plot(self.avg_signal_from_agents_sensors_over_time_WoC_AI, label="WoC AI", color="blue")
        plt.plot(self.avg_signal_reported_by_agents_over_time_RTSI_human, label="RTSI human", c="green")
        plt.plot(self.avg_signal_reported_by_agents_over_time_RTSI_AI, label="RTSI AI", c="#00e600")
        # plt.title(f'({cond}) Readings over time – separate agent types')
        plt.xlabel('Time')
        plt.ylabel('Signal value')
        plt.ylim(ymin=0, ymax=1)
        plt.legend()
        plt.savefig(f'images/Single_run_{cond}_3_Readings_over_time_separate_types.png')
        plt.show()

    def draw_chart_error_wrt_ground_truth(self, cond):
        plt.plot(self.error_over_time_WoC, label="WoC error", c="blue")
        plt.plot(self.error_over_time_RTSI, label="RTSI error", c="green")
        # plt.title(f'({cond}) Error over time')
        plt.xlabel('Time')
        plt.ylabel('Absolute error value')
        plt.ylim(ymin=0, ymax=1)
        plt.legend()
        plt.savefig(f'images/Single_run_{cond}_4_Error_over_time.png')
        plt.show()

    def draw_chart_error_wrt_ground_truth_separate_types(self, cond):
        plt.plot(self.error_over_time_WoC_human, label="WoC error human", c="blue")
        plt.plot(self.error_over_time_WoC_AI, label="WoC error AI", c="#8080ff")
        plt.plot(self.error_over_time_RTSI_human, label="RTSI error human", c="green")
        plt.plot(self.error_over_time_RTSI_AI, label="RTSI error AI", c="#00e600")
        # plt.title(f'({cond}) Error over time – separate agents types')
        plt.xlabel('Time')
        plt.ylabel('Absolute error value')
        plt.ylim(ymin=0, ymax=1)
        plt.legend()
        plt.savefig(f'images/Single_run_{cond}_5_Error_over_time_separate_types.png')
        plt.show()

    def draw_chart_experts(self, cond):
        n = self.number_of_periods
        x_axis = np.arange(n)
        experts_human = self.experts_human
        experts_AI = self.experts_AI
        width = 0.8
        plt.figure(dpi=700)
        plt.bar(x_axis, experts_human, width,align='edge', label="Human experts", color="g")
        plt.bar(x_axis, experts_AI, width, align='edge', bottom=experts_human, label="AI experts", color="#00e600")
        # plt.title(f'({cond}) Agents acting as expert sources')
        plt.xlabel('Time')
        plt.ylabel('Number')
        plt.legend()
        plt.savefig(f'images/Single_run_{cond}_6_agents_acting_as_expert_sources.png')
        plt.show()

    def draw_chart_readings_taken_from_agent_type(self, cond):
        n = self.number_of_periods
        x_axis = np.arange(n)
        readings_taken_from_AI = self.number_of_readings_taken_by_all_from_AI
        readings_taken_from_humans = self.number_of_readings_taken_by_all_from_humans
        width = 0.8
        plt.figure(dpi=700)
        plt.bar(x_axis, readings_taken_from_humans, width, align='edge', label="Readings taken from human", color="g")
        plt.bar(x_axis, readings_taken_from_AI, width, align='edge', bottom=readings_taken_from_humans, label="Readings taken from AI", color="#00e600")
        # plt.title(f'({cond}) Readings from agents types')
        plt.xlabel('Time')
        plt.ylabel('Number')
        plt.legend()
        plt.savefig(f'images/Single_run_{cond}_7_Readings_from_agent_type.png')
        plt.show()

    def draw_chart_readings_taken_from_agent_type_by_humans(self, cond):
        n = self.number_of_periods
        x_axis = np.arange(n)
        readings_taken_from_humans = self.number_of_readings_taken_by_humans_from_humans
        readings_taken_from_AI = self.number_of_readings_taken_by_humans_from_AI
        width = 0.8
        plt.figure(dpi=700)
        plt.bar(x_axis, readings_taken_from_humans, width, align='edge', label="Readings taken from human", color="g")
        plt.bar(x_axis, readings_taken_from_AI, width, align='edge', bottom=readings_taken_from_humans,
                label="Readings from AI", color="#00e600")
        # plt.title(f'({cond}) Readings by humans from agents types')
        plt.xlabel('Time')
        plt.ylabel('Number')
        plt.legend()
        plt.savefig(f'images/Single_run_{cond}_8_Readings_by_humans_from_agent_type.png')
        plt.show()

    def run(self):
        self.current_period = 0
        self.initialize_arrays()
        self.initialize_signals_sensed_by_agents_lists()
        for agent in self.agents:
            agent.reset_trust()
        self.avg_grand_truth_signal_over_time[0] = np.ndarray.mean(self.signal_meshgrid)
        for _ in range(self.number_of_periods):
            self.run_step()
        self.avg_signal_reported_by_agents_over_time_RTSI = np.mean(self.reported_signal_over_time_RTSI, axis=1)
        self.avg_signal_reported_by_agents_over_time_RTSI_human = np.mean(self.reported_signal_over_time_RTSI_human,
                                                                          axis=1)
        self.avg_signal_reported_by_agents_over_time_RTSI_AI = np.mean(self.reported_signal_over_time_RTSI_AI, axis=1)
        self.avg_signal_from_agents_sensors_over_time_WoC = np.mean(self.sensor_signals_over_time_WoC, axis=1)
        self.avg_signal_from_agents_sensors_over_time_WoC_human = np.mean(self.sensor_signals_over_time_WoC_human,
                                                                          axis=1)
        self.avg_signal_from_agents_sensors_over_time_WoC_AI = np.mean(self.sensor_signals_over_time_WoC_AI, axis=1)
        for step in range(len(self.experts_grouped_by_type)):
            human_from_human = len(self.experts_grouped_by_type[step]["human"]["human"])
            human_from_AI = len(self.experts_grouped_by_type[step]["human"]["AI"])
            AI_from_human = len(self.experts_grouped_by_type[step]["AI"]["human"])
            AI_from_AI = len(self.experts_grouped_by_type[step]["AI"]["AI"])

            self.number_of_readings_taken_by_humans_from_humans[step] = human_from_human
            self.number_of_readings_taken_by_humans_from_AI[step] = human_from_AI
            self.number_of_readings_taken_by_AI_from_humans[step] = AI_from_human
            self.number_of_readings_taken_by_AI_from_AI[step] = AI_from_AI
        self.number_of_readings_taken_by_all_from_AI = self.number_of_readings_taken_by_AI_from_AI + self.number_of_readings_taken_by_humans_from_AI
        self.number_of_readings_taken_by_all_from_humans = self.number_of_readings_taken_by_AI_from_humans + self.number_of_readings_taken_by_humans_from_humans

    def run_step(self):
        reading_results_RTSI = []
        reading_results_RTSI_human = []
        reading_results_RTSI_AI = []
        reading_results_WoC = []
        reading_results_WoC_human = []
        reading_results_WoC_AI = []
        grand_truth_signal = np.ndarray.mean(self.signal_meshgrid)
        for agent in self.agents:  # we have to set all agent values before further operations
            agent.set_self_reading()
        # experts
        experts = self.select_experts()
        experts_grouped_by_type = self.select_experts_group_by_type()
        number_of_experts_human = len(experts['human'])
        number_of_experts_AI = len(experts['AI'])
        self.experts_human[self.current_period] = number_of_experts_human
        self.experts_AI[self.current_period] = number_of_experts_AI
        self.experts_grouped_by_type.append(experts_grouped_by_type)

        for agent in self.agents:
            reported_reading = agent.select_reading()
            reading_results_RTSI.append(reported_reading)
            reading_results_WoC.append(agent.current_reading)
            if agent.type == "human":
                reading_results_RTSI_human.append(reported_reading)
                reading_results_WoC_human.append(agent.current_reading)
            else:
                reading_results_RTSI_AI.append(reported_reading)
                reading_results_WoC_AI.append(agent.current_reading)
        for agent in self.agents:
            agent.update_trust(grand_truth_signal, np.mean(reading_results_RTSI))

        self.reported_signal_over_time_RTSI[self.current_period] = reading_results_RTSI
        self.reported_signal_over_time_RTSI_human[self.current_period] = reading_results_RTSI_human
        self.reported_signal_over_time_RTSI_AI[self.current_period] = reading_results_RTSI_AI
        self.sensor_signals_over_time_WoC[self.current_period] = reading_results_WoC
        self.sensor_signals_over_time_WoC_human[self.current_period] = reading_results_WoC_human
        self.sensor_signals_over_time_WoC_AI[self.current_period] = reading_results_WoC_AI
        self.avg_grand_truth_signal_over_time[self.current_period] = grand_truth_signal
        # global error
        self.error_over_time_RTSI[self.current_period] = abs(
            np.mean(self.reported_signal_over_time_RTSI[self.current_period]) - grand_truth_signal)
        self.error_over_time_WoC[self.current_period] = abs(
            np.mean(self.sensor_signals_over_time_WoC[self.current_period]) - grand_truth_signal)
        # human only error
        self.error_over_time_RTSI_human[self.current_period] = abs(
            np.mean(self.reported_signal_over_time_RTSI_human[self.current_period]) - grand_truth_signal)
        self.error_over_time_WoC_human[self.current_period] = abs(
            np.mean(self.sensor_signals_over_time_WoC_human[self.current_period]) - grand_truth_signal)
        # AI only error
        self.error_over_time_RTSI_AI[self.current_period] = abs(
            np.mean(self.reported_signal_over_time_RTSI_AI[self.current_period]) - grand_truth_signal)
        self.error_over_time_WoC_AI[self.current_period] = abs(
            np.mean(self.sensor_signals_over_time_WoC_AI[self.current_period]) - grand_truth_signal)
        self.current_period += 1

    def select_experts_group_by_type(self) -> Dict[str, Dict[str, List]]:
        experts = {'human': defaultdict(list), "AI": defaultdict(list)}
        for agent in self.agents:
            maybe_expert = agent.select_maybe_expert()
            if maybe_expert is None:
                experts[agent.type][agent.type].append(agent)
            if maybe_expert is not None:
                experts[agent.type][maybe_expert.type].append(maybe_expert)
        return experts

    def select_experts(self) -> Dict[str, Set]:
        experts = {'human': set(), "AI": set()}
        for agent in self.agents:
            maybe_expert = agent.select_maybe_expert()
            if maybe_expert is not None:
                experts[maybe_expert.type].add(maybe_expert)
        return experts

