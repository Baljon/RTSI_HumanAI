from many_runs import run_n_runs_with_same_agent_positions, run_n_sim_with_different_agent_positions
import agent as ag

######################
# Same positions
######################

# # Condition I
# ag.initial_trust_human_to_AI = 0.5
# ag.update_up_alfa_human_to_AI = 0.01
# ag.update_down_beta_human_to_AI = 0.01
#
# result_cond_A = run_n_runs_with_same_agent_positions(n_sim=30,
#                                                      grid_state=5,
#                                                      state=5,
#                                                      no_periods=300,
#                                                      no_agents=100,
#                                                      is_bidirectional=False,
#                                                      condition="I")
#
# # Setup II
# ag.initial_trust_human_to_AI = 0.2
# ag.update_up_alfa_human_to_AI = 0.01
# ag.update_down_beta_human_to_AI = 0.1
#
# result_cond_B = run_n_runs_with_same_agent_positions(n_sim=100,
#                                                      grid_state=5,
#                                                      state=5,
#                                                      no_periods=300,
#                                                      no_agents=100,
#                                                      is_bidirectional=False,
#                                                      condition="II")

# Setup III
ag.initial_trust_human_to_AI = 0.8
ag.update_up_alfa_human_to_AI = 0.005
ag.update_down_beta_human_to_AI = 0.1

result_cond_C = run_n_runs_with_same_agent_positions(n_sim=30,
                                                     grid_state=5,
                                                     state=5,
                                                     no_periods=300,
                                                     no_agents=100,
                                                     is_bidirectional=False,
                                                     condition="III")

######################
# Different positions
######################
# # Setup I
# ag.initial_trust_human_to_AI = 0.5
# ag.update_up_alfa_human_to_AI = 0.01
# ag.update_down_beta_human_to_AI = 0.01
#
# run_n_sim_with_different_agent_positions(n_sim=30,
#                                          no_periods=300,
#                                          no_agents=100,
#                                          is_bidirectional=False,
#                                          condition="I",
# grid_state=5,
#                                          random_state=5)
#
# # Setup II
# ag.initial_trust_human_to_AI = 0.2
# ag.update_up_alfa_human_to_AI = 0.01
# ag.update_down_beta_human_to_AI = 0.1
#
# run_n_sim_with_different_agent_positions(n_sim=30,
#                                          no_periods=300,
#                                          no_agents=100,
#                                          is_bidirectional=False,
#                                          condition="II",
# grid_state=5,
#                                          random_state=5)
#
# # Setup III
# ag.initial_trust_human_to_AI = 0.8
# ag.update_up_alfa_human_to_AI = 0.005
# ag.update_down_beta_human_to_AI = 0.1
#
# run_n_sim_with_different_agent_positions(n_sim=30,
#                                          no_periods=300,
#                                          no_agents=100,
#                                          is_bidirectional=False,
#                                          condition="III",
# grid_state = 5,
#                                          random_state=5)
