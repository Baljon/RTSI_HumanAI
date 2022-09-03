from simulation import Simulation
import agent as ag

condition = "I"
ag.initial_trust_human_to_AI = 0.5
ag.update_up_alfa_human_to_AI = 0.01
ag.update_down_beta_human_to_AI = 0.01

s = Simulation(the_same_initial_temp_everywhere=False,
               is_bidirectional=False,
               periods=300,
               agents=100,
               neighbours=5,
               grid_state=5,
               random_state=5)

s.run()
s.draw_chart_agents(f"Single_run_{condition}_1_agents")
s.draw_chart_readings_over_time(condition)
s.draw_chart_readings_over_time_separate_types(condition)
s.draw_chart_error_wrt_ground_truth(condition)
s.draw_chart_error_wrt_ground_truth_separate_types(condition)
s.draw_chart_experts(condition)
s.draw_chart_readings_taken_from_agent_type(condition)
s.draw_chart_readings_taken_from_agent_type_by_humans(condition)


condition = "II"
ag.initial_trust_human_to_AI = 0.2
ag.update_up_alfa_human_to_AI = 0.01
ag.update_down_beta_human_to_AI = 0.1

s = Simulation(the_same_initial_temp_everywhere=False,
               is_bidirectional=False,
               periods=300,
               agents=100,
               neighbours=5,
               grid_state=5,
               random_state=5)

s.run()
s.draw_chart_agents(f"Single_run_{condition}_1_agents")
s.draw_chart_readings_over_time(condition)
s.draw_chart_readings_over_time_separate_types(condition)
s.draw_chart_error_wrt_ground_truth(condition)
s.draw_chart_error_wrt_ground_truth_separate_types(condition)
s.draw_chart_experts(condition)
s.draw_chart_readings_taken_from_agent_type(condition)
s.draw_chart_readings_taken_from_agent_type_by_humans(condition)


condition = "III"
ag.initial_trust_human_to_AI = 0.8
ag.update_up_alfa_human_to_AI = 0.005
ag.update_down_beta_human_to_AI = 0.1

s = Simulation(the_same_initial_temp_everywhere=False,
               is_bidirectional=False,
               periods=300,
               agents=100,
               neighbours=5,
               grid_state=5,
               random_state=5)

s.run()
s.draw_chart_agents(f"Single_run_{condition}_1_agents")
s.draw_chart_readings_over_time(condition)
s.draw_chart_readings_over_time_separate_types(condition)
s.draw_chart_error_wrt_ground_truth(condition)
s.draw_chart_error_wrt_ground_truth_separate_types(condition)
s.draw_chart_experts(condition)
s.draw_chart_readings_taken_from_agent_type(condition)
s.draw_chart_readings_taken_from_agent_type_by_humans(condition)
