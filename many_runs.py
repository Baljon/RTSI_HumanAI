from matplotlib import pyplot as plt
from charts import plot_error, plot_error_separate_types, plot_expert_source, plot_readings_taken_from_agent_type, \
    plot_readings_taken_from_agent_type_by_humans
from simulation import Simulation
import agent as ag
import numpy as np
from tqdm import tqdm


def run_n_sim_with_different_agent_positions(n_sim=30, is_bidirectional=False, no_periods=40, no_agents=40, condition="None",
                                             grid_state=None, random_state=None):
    if random_state is None:
        simulations = [Simulation(the_same_initial_temp_everywhere=False,
                                  is_bidirectional=is_bidirectional,
                                  periods=no_periods,
                                  agents=no_agents,
                                  grid_state=None,
                                  random_state=None)
                       for _ in range(n_sim)]
    else:
        simulations = [Simulation(the_same_initial_temp_everywhere=False,
                                  is_bidirectional=is_bidirectional,
                                  periods=no_periods,
                                  agents=no_agents,
                                  grid_state = grid_state + i,
                                  random_state=random_state + i)
                       for i in range(n_sim)]
    for s in tqdm(simulations):
        s.run()

    errors_over_time_rtsi_different_positions = np.mean([s.error_over_time_RTSI for s in simulations], axis=0)
    errors_over_time_rtsi_different_positions_human = np.mean([s.error_over_time_RTSI_human for s in simulations],
                                                              axis=0)
    errors_over_time_rtsi_different_positions_AI = np.mean([s.error_over_time_RTSI_AI for s in simulations], axis=0)

    errors_over_time_woc_different_positions = np.mean([s.error_over_time_WoC for s in simulations], axis=0)
    errors_over_time_woc_different_positions_human = np.mean([s.error_over_time_WoC_human for s in simulations], axis=0)
    errors_over_time_woc_different_positions_AI = np.mean([s.error_over_time_WoC_AI for s in simulations], axis=0)

    # plot error
    plt.plot(errors_over_time_rtsi_different_positions, label=f"RTSI error w.r.t. ground truth", color="green")
    plt.plot(errors_over_time_woc_different_positions, label=f"WoC error w.r.t. ground truth", color="blue")
    plt.legend()
    plt.title(f"Errors – avg. of {n_sim} runs, different positions")
    plt.xlabel('Time')
    plt.ylabel('Absolute error value')
    plt.savefig(f'images/Many_runs_{condition}_1_Errors_avg_{n_sim}_runs.png')
    plt.show()

    # plot error separate types
    # plt.plot(errors_over_time_woc_different_positions_human, label="WoC error human", c="blue")
    # plt.plot(errors_over_time_woc_different_positions_AI, label="WoC error AI", c="#8080ff")
    # plt.plot(errors_over_time_rtsi_different_positions_human, label="RTSI error human", c="green")
    # plt.plot(errors_over_time_rtsi_different_positions_AI, label="RTSI error AI", c="#00e600")
    # plt.title(f"Errors for types – avg. of {n_sim} runs, different positions")
    # plt.xlabel('Time')
    # plt.ylabel('Absolute error value')
    # plt.legend()
    # plt.show()


def run_n_runs_with_same_agent_positions(n_sim=30, is_bidirectional=False, no_periods=40, no_agents=40, state=0, grid_state=0,
                                         condition="default", plot_charts=True):
    n_runs = n_sim
    errors_rtsi = []
    errors_rtsi_human = []
    errors_rtsi_AI = []
    errors_woc = []
    errors_woc_human = []
    errors_woc_AI = []
    experts_human = []
    number_of_readings_taken_from_AI = []
    number_of_readings_taken_from_humans = []
    number_of_readings_taken_by_humans_from_AI = []
    number_of_readings_taken_by_humans_from_humans = []
    experts_AI = []
    simulations = []
    print(f'Running {n_runs} simulation with same agent position with condition {condition}...')
    for _ in tqdm(range(n_runs)):
        simulation = Simulation(the_same_initial_temp_everywhere=False, is_bidirectional=is_bidirectional,
                                periods=no_periods, agents=no_agents, grid_state=grid_state, random_state=state)
        simulation.run()
        errors_rtsi.append(simulation.error_over_time_RTSI)
        errors_rtsi_human.append(simulation.error_over_time_RTSI_human)
        errors_rtsi_AI.append(simulation.error_over_time_RTSI_AI)
        errors_woc.append(simulation.error_over_time_WoC)
        errors_woc_human.append(simulation.error_over_time_WoC_human)
        errors_woc_AI.append(simulation.error_over_time_WoC_AI)
        experts_human.append(simulation.experts_human)
        experts_AI.append(simulation.experts_AI)
        number_of_readings_taken_from_AI.append(simulation.number_of_readings_taken_by_all_from_AI)
        number_of_readings_taken_from_humans.append(simulation.number_of_readings_taken_by_all_from_humans)
        number_of_readings_taken_by_humans_from_AI.append(simulation.number_of_readings_taken_by_humans_from_AI)
        number_of_readings_taken_by_humans_from_humans.append(simulation.number_of_readings_taken_by_humans_from_humans)
        simulations.append(simulation)

        state += 1

    errors_over_time_rtsi_same_position = np.mean(errors_rtsi, axis=0)
    errors_over_time_rtsi_same_position_human = np.mean(errors_rtsi_human, axis=0)
    errors_over_time_rtsi_same_position_AI = np.mean(errors_rtsi_AI, axis=0)
    errors_over_time_woc_same_position = np.mean(errors_woc, axis=0)
    errors_over_time_woc_same_position_human = np.mean(errors_woc_human, axis=0)
    errors_over_time_woc_same_position_AI = np.mean(errors_woc_AI, axis=0)
    experts_over_time_human = np.mean(experts_human, axis=0)
    experts_over_time_AI = np.mean(experts_AI, axis=0)
    number_of_readings_taken_from_humans_over_time = np.mean(number_of_readings_taken_from_humans, axis=0)
    number_of_readings_taken_from_AI_over_time = np.mean(number_of_readings_taken_from_AI, axis=0)
    number_of_readings_taken_by_humans_from_humans_over_time = np.mean(number_of_readings_taken_by_humans_from_humans,
                                                                       axis=0)
    number_of_readings_taken_by_humans_from_AI_over_time = np.mean(number_of_readings_taken_by_humans_from_AI, axis=0)

    # plot agents
    simulation.draw_chart_agents(f'Many_runs_{condition}_1')
    if plot_charts:
        plot_error(errors_over_time_rtsi_same_position, errors_over_time_woc_same_position, n_sim,
                   cond=condition)
        plot_error_separate_types(errors_over_time_rtsi_same_position_human,
                                  errors_over_time_rtsi_same_position_AI,
                                  errors_over_time_woc_same_position_human,
                                  errors_over_time_woc_same_position_AI,
                                  n_sim,
                                  cond=condition)
        plot_expert_source(no_periods, experts_over_time_human, experts_over_time_AI, n_sim,
                           cond=condition)
        plot_readings_taken_from_agent_type(no_periods, number_of_readings_taken_from_humans_over_time,
                                            number_of_readings_taken_from_AI_over_time, n_sim,
                                            cond=condition)
        plot_readings_taken_from_agent_type_by_humans(no_periods,
                                                      number_of_readings_taken_by_humans_from_humans_over_time,
                                                      number_of_readings_taken_by_humans_from_AI_over_time, n_sim,
                                                      cond=condition)
    return {'n_runs': n_runs,
            'error_rtsi': errors_rtsi,
            'error_rtsi_over_time': errors_over_time_rtsi_same_position,
            'error_woc_over_time': errors_over_time_woc_same_position,
            'error_rtsi_human': errors_rtsi_human,
            'error_rtsi_AI': errors_rtsi_AI,
            'errors_woc': errors_woc,
            'errors_woc_human': errors_woc_human,
            'errors_woc_AI': errors_woc_AI,
            'experts_human': experts_human,
            'experts_AI': experts_AI,
            'simulations': simulations}
