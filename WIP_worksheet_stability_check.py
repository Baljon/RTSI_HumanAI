import numpy as np
import agent as ag
from simulation import Simulation

# WORK IN PROGRESS

initial_trust_human_to_AI = 0.46
update_up_alfa_human_to_AI = 0.007
update_down_beta_human_to_AI = 0.2

ag.initial_trust_human_to_AI = initial_trust_human_to_AI
ag.update_up_alfa_human_to_AI = update_up_alfa_human_to_AI
ag.update_down_beta_human_to_AI = update_down_beta_human_to_AI

epsilon = 0.0001


def calculate_score(sim: Simulation, last_n=1):
    return np.mean((sim.error_over_time_RTSI[-last_n:]))


def check_stability(sim1: Simulation, sim2: Simulation):
    error1 = calculate_score(sim1)
    error2 = calculate_score(sim2)
    return {
        # 'sim1': copy.deepcopy(sim1),
        # 'sim2': copy.deepcopy(sim2),
        'error1': error1,
        'error2': error2,
        'result': abs(error1 - error2) < epsilon
    }


s_baseline = Simulation(the_same_initial_temp_everywhere=False,
                        is_bidirectional=False,
                        periods=300,
                        agents=100,
                        neighbours=5,
                        random_state=5)
s_baseline.run()
result = []

ag.initial_trust_human_to_AI = initial_trust_human_to_AI + epsilon
s_disturbed = Simulation(the_same_initial_temp_everywhere=False,
                         is_bidirectional=False,
                         periods=300,
                         agents=100,
                         neighbours=5,
                         random_state=5)
s_disturbed.run()
result.append(check_stability(s_baseline, s_disturbed))

ag.initial_trust_human_to_AI = initial_trust_human_to_AI - epsilon
s_disturbed = Simulation(the_same_initial_temp_everywhere=False,
                         is_bidirectional=False,
                         periods=300,
                         agents=100,
                         neighbours=5,
                         random_state=5)
s_disturbed.run()
result.append(check_stability(s_baseline, s_disturbed))

ag.initial_trust_human_to_AI = initial_trust_human_to_AI

#########

ag.update_up_alfa_human_to_AI = update_up_alfa_human_to_AI + epsilon
s_disturbed = Simulation(the_same_initial_temp_everywhere=False,
                         is_bidirectional=False,
                         periods=300,
                         agents=100,
                         neighbours=5,
                         random_state=5)
s_disturbed.run()
result.append(check_stability(s_baseline, s_disturbed))

ag.update_up_alfa_human_to_AI = update_up_alfa_human_to_AI - epsilon
s_disturbed = Simulation(the_same_initial_temp_everywhere=False,
                         is_bidirectional=False,
                         periods=300,
                         agents=100,
                         neighbours=5,
                         random_state=5)
s_disturbed.run()
result.append(check_stability(s_baseline, s_disturbed))

ag.update_up_alfa_human_to_AI = update_up_alfa_human_to_AI

#########
ag.update_down_beta_human_to_AI = update_down_beta_human_to_AI + epsilon
s_disturbed = Simulation(the_same_initial_temp_everywhere=False,
                         is_bidirectional=False,
                         periods=300,
                         agents=100,
                         neighbours=5,
                         random_state=5)
s_disturbed.run()
result.append(check_stability(s_baseline, s_disturbed))

ag.update_down_beta_human_to_AI = update_down_beta_human_to_AI - epsilon
s_disturbed = Simulation(the_same_initial_temp_everywhere=False,
                         is_bidirectional=False,
                         periods=300,
                         agents=100,
                         neighbours=5,
                         random_state=5)
s_disturbed.run()
result.append(check_stability(s_baseline, s_disturbed))

ag.update_down_beta_human_to_AI = update_down_beta_human_to_AI

s_disturbed.draw_chart_error_wrt_ground_truth("test")