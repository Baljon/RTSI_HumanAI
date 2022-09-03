import matplotlib.pyplot as plt
import numpy as np


def plot_error(errors_over_time_rtsi_same_position, errors_over_time_woc_same_position, n_sim, cond):
    plt.plot(errors_over_time_rtsi_same_position, label=f"RTSI error", color="green")
    plt.plot(errors_over_time_woc_same_position, label=f"WoC error", color="blue")
    # plt.title(f"({cond}) Errors – avg. of {n_sim} runs")
    plt.xlabel('Time')
    plt.ylabel('Absolute error value')
    plt.legend()
    plt.savefig(f'images/Many_runs_{cond}_2_Errors_avg_{n_sim}_runs.png')
    plt.show()


def plot_error_separate_types(errors_over_time_rtsi_same_position_human,
                              errors_over_time_rtsi_same_position_AI,
                              errors_over_time_woc_same_position_human,
                              errors_over_time_woc_same_position_AI,
                              n_sim,
                              cond):
    plt.plot(errors_over_time_rtsi_same_position_human, label="RTSI error human", c="green")
    plt.plot(errors_over_time_rtsi_same_position_AI, label="RTSI error AI", c="#00e600")
    plt.plot(errors_over_time_woc_same_position_human, label="WoC error human", c="blue")
    plt.plot(errors_over_time_woc_same_position_AI, label="WoC error AI", c="#8080ff")
    # plt.title(f"({cond}) Errors for types – avg. of {n_sim} runs")
    plt.xlabel('Time')
    plt.ylabel('Absolute error value')
    plt.legend()
    plt.savefig(f'images/Many_runs_{cond}_3_Errors_for_types_avg_{n_sim}_runs.png')
    plt.show()


def plot_expert_source(no_periods, experts_over_time_human, experts_over_time_AI, n_sim, cond):
    n = no_periods
    x_axis = np.arange(n)
    width = 0.8
    plt.figure(dpi=700)
    plt.bar(x_axis, experts_over_time_human, width, align='edge', label="Human experts", color="g")
    plt.bar(x_axis, experts_over_time_AI, width, align='edge', bottom=experts_over_time_human, label="AI experts",
            color="#00e600")
    # plt.title(f'({cond}) Agents acting as expert sources – avg. of {n_sim} runs')
    plt.xlabel('Time')
    plt.ylabel('Number')
    plt.legend()
    plt.savefig(f'images/Many_runs_{cond}_4_Agents_as_expert_sources_avg_{n_sim}_runs.png')
    plt.show()


def plot_readings_taken_from_agent_type(no_periods, number_of_readings_taken_from_humans_over_time,
                                        number_of_readings_taken_from_AI_over_time, n_sim, cond):
    n = no_periods
    x_axis = np.arange(n)
    readings_taken_from_humans = number_of_readings_taken_from_humans_over_time
    readings_taken_from_AI = number_of_readings_taken_from_AI_over_time
    width = 0.8
    plt.figure(dpi=700)
    plt.bar(x_axis, readings_taken_from_humans, width, align='edge', label="Readings from human", color="g")
    plt.bar(x_axis, readings_taken_from_AI, width, align='edge', bottom=readings_taken_from_humans,
            label="Readings from AI", color="#00e600")
    # plt.title(f'({cond}) Readings from agents types – avg. of {n_sim} runs')
    plt.xlabel('Time')
    plt.ylabel('Number')
    plt.legend()
    plt.savefig(f'images/Many_runs_{cond}_5_Readings_from_agent_types_avg_{n_sim}_runs.png')
    plt.show()


def plot_readings_taken_from_agent_type_by_humans(no_periods, number_of_readings_taken_by_humans_from_humans_over_time,
                                                  number_of_readings_taken_by_humans_from_AI_over_time, n_sim, cond):
    n = no_periods
    x_axis = np.arange(n)
    readings_taken_from_humans = number_of_readings_taken_by_humans_from_humans_over_time
    readings_taken_from_AI = number_of_readings_taken_by_humans_from_AI_over_time
    width = 0.8
    plt.figure(dpi=700)
    plt.bar(x_axis, readings_taken_from_humans, width, align='edge', label="Readings from human", color="g")
    plt.bar(x_axis, readings_taken_from_AI, width, align='edge', bottom=readings_taken_from_humans,
            label="Readings from AI", color="#00e600")
    # plt.title(f'({cond}) Readings by humans from agents types – avg. of {n_sim} runs')
    plt.xlabel('Time')
    plt.ylabel('Number')
    plt.legend()
    plt.savefig(f'images/Many_runs_{cond}_6_Readings_by_humans_from_agent_types_avg_{n_sim}_runs.png')
    plt.show()
