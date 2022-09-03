from typing import Dict
import numpy as np
import agent as ag
from many_runs import run_n_runs_with_same_agent_positions

# WORK IN PROGRESS

initial_trust_human_to_AI = 0.46
update_up_alfa_human_to_AI = 0.007
update_down_beta_human_to_AI = 0.2

ag.initial_trust_human_to_AI = initial_trust_human_to_AI
ag.update_up_alfa_human_to_AI = update_up_alfa_human_to_AI
ag.update_down_beta_human_to_AI = update_down_beta_human_to_AI

epsilon = 0.001


def calculate_score(sim: Dict, last_n=1):
    return np.mean((sim['error_rtsi'][-last_n:]))


def check_stability(sim1: Dict, sim2: Dict):
    error1 = calculate_score(sim1)
    error2 = calculate_score(sim2)
    return {
        # 'sim1': copy.deepcopy(sim1),
        # 'sim2': copy.deepcopy(sim2),
        'error1': error1,
        'error2': error2,
        'result': abs(error1 - error2) < epsilon
    }


s_baseline = run_n_runs_with_same_agent_positions(n_sim=15,
                                                  grid_state=5,
                                                  state=5,
                                                  no_periods=300,
                                                  no_agents=100,
                                                  is_bidirectional=False,
                                                  condition="0",
                                                  plot_charts=False)
result = []

ag.initial_trust_human_to_AI = initial_trust_human_to_AI + epsilon
s_disturbed = run_n_runs_with_same_agent_positions(n_sim=15,
                                                   grid_state=5,
                                                   state=5,
                                                   no_periods=300,
                                                   no_agents=100,
                                                   is_bidirectional=False,
                                                   condition="0",
                                                   plot_charts=False)
result.append(check_stability(s_baseline, s_disturbed))

ag.initial_trust_human_to_AI = initial_trust_human_to_AI - epsilon
s_disturbed = run_n_runs_with_same_agent_positions(n_sim=15,
                                                   grid_state=5,
                                                   state=5,
                                                   no_periods=300,
                                                   no_agents=100,
                                                   is_bidirectional=False,
                                                   condition="0", plot_charts=False)
result.append(check_stability(s_baseline, s_disturbed))

ag.initial_trust_human_to_AI = initial_trust_human_to_AI

#########

ag.update_up_alfa_human_to_AI = update_up_alfa_human_to_AI + epsilon
s_disturbed = run_n_runs_with_same_agent_positions(n_sim=15,
                                                   grid_state=5,
                                                   state=5,
                                                   no_periods=300,
                                                   no_agents=100,
                                                   is_bidirectional=False,
                                                   condition="0", plot_charts=False)
result.append(check_stability(s_baseline, s_disturbed))

ag.update_up_alfa_human_to_AI = update_up_alfa_human_to_AI - epsilon
s_disturbed = run_n_runs_with_same_agent_positions(n_sim=15,
                                                   grid_state=5,
                                                   state=5,
                                                   no_periods=300,
                                                   no_agents=100,
                                                   is_bidirectional=False,
                                                   condition="0", plot_charts=False)
result.append(check_stability(s_baseline, s_disturbed))

ag.update_up_alfa_human_to_AI = update_up_alfa_human_to_AI

#########
ag.update_down_beta_human_to_AI = update_down_beta_human_to_AI + epsilon
s_disturbed = run_n_runs_with_same_agent_positions(n_sim=15,
                                                   grid_state=5,
                                                   state=5,
                                                   no_periods=300,
                                                   no_agents=100,
                                                   is_bidirectional=False,
                                                   condition="0", plot_charts=False)
result.append(check_stability(s_baseline, s_disturbed))

ag.update_down_beta_human_to_AI = update_down_beta_human_to_AI - epsilon
s_disturbed = run_n_runs_with_same_agent_positions(n_sim=15,
                                                   grid_state=5,
                                                   state=5,
                                                   no_periods=300,
                                                   no_agents=100,
                                                   is_bidirectional=False,
                                                   condition="0", plot_charts=False)
result.append(check_stability(s_baseline, s_disturbed))

ag.update_down_beta_human_to_AI = update_down_beta_human_to_AI
