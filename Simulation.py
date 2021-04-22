"""
Authors:
---
    Jayce Slesar
    Brandon Lee

Date:
---
    10/15/2020
"""
import pygame, sys
import matplotlib.pyplot as plt
from pygame.locals import *
import time
import Graph
import SocialNetwork as sn
from multiprocessing import Process
import SocialNetworkGrapher as sng
from finalproject import enumerate_states, get_next_states, reward_generator, policy_evaluation, value_iteration, RandomAgent, Deterministic_Agent, Soft_Deterministic_Agent, RL_Agent, expected_SARSA, q_learning
import copy
import pickle
import numpy as np
import json
import pandas as pd
import Space
from PIL import Image
import os

class Simulation:
    def __init__(self, rows: int, cols: int, num_steps: int, output: bool, seed: int, states):
        self.rows = rows
        self.cols = cols
        self.steps = num_steps
        self.WINDOW_HEIGHT = 800
        self.WINDOW_WIDTH = 800
        self.height_per_block = self.WINDOW_HEIGHT // rows
        self.width_per_block = self.WINDOW_WIDTH // cols
        self.seed = seed
        self.states = states

        self.output = output
        self.m = Space.Space(self.rows, self.cols, self.steps, self.output, self.seed)
        self.scores = []

    def play_sim(self, agent: RL_Agent, times):
        for i in range(times):
            print("game ", i, " running now")
            enum_m = copy.deepcopy(self.m)
            while enum_m.steps_taken < enum_m.iterations:
                action = agent.get_action(enum_m)
                print("got action")
                enum_m._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
                print("swap done")
                enum_m._step_()
                print("Game on step ", enum_m.steps_taken)

            self.scores.append(enum_m.infected_count + enum_m.recovered_count)


        # global SCREEN, CLOCK
        # BLACK = (0, 0, 0)
        # pygame.init()
        # SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        # CLOCK = pygame.time.Clock()
        # SCREEN.fill(BLACK)

        # m = copy.deepcopy(self.m)
        # while m.steps_taken < m.iterations:
        #     action = agent.get_action(m)
        #     m._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
        #     m._step_()
        #     self.draw(m.grid)
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             sys.exit()
        #     pygame.display.update()
        #     time.sleep(3)
        # pygame.quit()

    def play_sim_save_viz(self, agent: RL_Agent, sc_path):
        global SCREEN, CLOCK
        BLACK = (0, 0, 0)
        pygame.init()
        SCREEN = pygame.display.set_mode((self.WINDOW_HEIGHT, self.WINDOW_WIDTH))
        CLOCK = pygame.time.Clock()
        SCREEN.fill(BLACK)

        stills = []
        enum_m = copy.deepcopy(self.m)
        while enum_m.steps_taken < enum_m.iterations:
            action = agent.get_action(enum_m)
            enum_m._RL_agent_swap(action[0][0], action[0][1], action[1][0], action[1][1])
            enum_m._step_()
            self.draw(enum_m.grid)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()
            time.sleep(.3)
            self.screenshot(SCREEN, sc_path, enum_m.steps_taken)
            stills.append(os.path.join(sc_path, "step" + str(enum_m.steps_taken) + ".png"))

        self.scores.append(enum_m.infected_count + enum_m.recovered_count)
        pygame.quit()

        img, *imgs = [Image.open(f) for f in stills]
        img.save(fp=os.path.join(sc_path, f'{agent.type}.gif'), format='GIF', append_images=imgs, save_all=True, duration=20, loop=0)
        for im in stills:
            os.remove(im)

    def screenshot(self, screen, path, step):
        title = "step" + str(step)
        file_save_as = os.path.join(path, title + ".png")
        pygame.image.save(screen, file_save_as)
        print(f"step {step} has been screenshotted")


    def draw(self, grid):
        for x, i in enumerate(range(self.rows)):
            for y, j in enumerate(range(self.cols)):
                rect = pygame.Rect(x*self.height_per_block, y*self.height_per_block,
                                   self.height_per_block, self.height_per_block)
                pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)

'''
PART 1:
- Runs enumerate States for 4x4 for 5 steps and pickles it
- Runs Value iteration for the 4x4 for 5 steps adn pickles it
- Runs Expected SARSA and pickles output
- Runs Q-Learning and pickles output
- USE OUPUT OF SARSA AND Q-LEARNING TO COMPARE TO VALUE ITERATION POLICY
'''
def part1():
    # ___________Setting Parameters for Value iteration Board ___________________
    four_by_four_space = Space.Space(4, 4, 5, False, 42)
    copy_env = copy.deepcopy(four_by_four_space)
    print("_____________Running enumerate states for a 4 by 4 grid__________________")
    states = enumerate_states(copy_env)
    print("_____________Enumerating States Finished______________________")

    # pickling states
    with open(f'states_{4}_{4}_{5}.pickle', 'wb') as handle:
        pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("______________Running Value iteration for a 4 by 4 grid....________________________")
    val_iter_rand_agent = RandomAgent()
    rand_agent_value_iteration = value_iteration(states, val_iter_rand_agent)
    print("______________Finished Value iteration________________________")

    # pickling value iteration output
    with open(f'value_iteration_output.pickle', 'wb') as handle:
        pickle.dump(rand_agent_value_iteration, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("____________Expected SARSA for a 4 by 4 grid_______________")

    print("EXPECTED SARSA TRAINING RUNNING (4X4)...")
    copy_env = copy.deepcopy(four_by_four_space)
    SARSA4_Output = expected_SARSA(copy_env)
    print("EXPECTED SARSA TRAINING FINISHED!")

    # This saves the SARSA output to file
    # with open(f'SARSA4_Output', 'wb') as handle:
    #     pickle.dump(SARSA4_Output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("____________Q-Learning for a 4 by 4 grid_______________")

    print("Q-LEARNING TRAINING RUNNING (4X4)...")
    copy_env = copy.deepcopy(four_by_four_space)
    QL4_Output = q_learning(copy_env)
    print("Q-LEARNING TRAINING FINISHED!")

    # This saves the SARSA output to file
    # with open(f'QL4_Output', 'wb') as handle:
    #     pickle.dump(QL4_Output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return SARSA4_Output, QL4_Output

'''
PART 2:
- RUNS EXPECTED SARSA AND Q-LEARNING TO TRAIN 2 AGENTS
- OUTPUTS THE 2 AGENTS
'''

def part2():
    # -----------Setting Board Parameters for Training and Baseline-------------------
    rows = 10
    cols = 10
    num_steps = 20
    output = False
    seed = 42
    # -------------Creating Board-------------------------
    test_space = Space.Space(10, 10, 30, output, seed)

    # ------------------Training TD Agent with expected SARSA--------------------
    print("____________Expected SARSA_______________")

    print("EXPECTED SARSA TRAINING RUNNING...")
    copy_env = copy.deepcopy(test_space)
    SARSA_player, SARSA_diffs, SARSA_states, SARSA_num_episodes, SARSA_step, SARSA_TD_error = expected_SARSA(copy_env)
    SARSA_output = (SARSA_player, SARSA_diffs, SARSA_states, SARSA_num_episodes, SARSA_step, SARSA_TD_error)
    print("EXPECTED SARSA TRAINING FINISHED!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = SARSA_TD_error
    my_df.to_csv('SARSA_TD_ERROR.csv')

    print("Number of States Visited: ", len(SARSA_states))
    print("Number of episodes: ", SARSA_num_episodes)

    # This saves the SARSA output to file
    # with open(f'SARSA10_Output', 'wb') as handle:
    #     pickle.dump(SARSA_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------Training TD Agent with q-Learning--------------------
    print("____________Q_Learning_______________")

    print("Q-LEARNING TRAINING RUNNING...")
    copy_env = copy.deepcopy(test_space)
    Q_player, Q_diffs, Q_states, Q_num_episodes, Q_step, Q_TD_error = expected_SARSA(copy_env)
    QL_output = (Q_player, Q_diffs, Q_states, Q_num_episodes, Q_step, Q_TD_error)
    print("Q-LEARNING TRAINING FINISHED!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = Q_TD_error
    my_df.to_csv('Q_TD_ERROR.csv')

    print("Number of States Visited: ", len(Q_states))
    print("Number of episodes: ", Q_num_episodes)

    # This saves the SARSA output to file
    # with open(f'QL10_Output', 'wb') as handle:
    #     pickle.dump(QL_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return SARSA_player, Q_player

def part3(sarsa_agent, q_agent):

    # -----------Setting Board Parameters for Training and Baseline-------------------
    rows = 10
    cols = 10
    num_steps = 20
    output = False
    seed = 42
    times = 100
    agents_scores_dict = {}
    # ________________INITIALIZE THE BASE AGENTS________________
    rand = RandomAgent()
    det = Deterministic_Agent()
    soft = Soft_Deterministic_Agent()

    #________________RUN PLAY GAME FOR RANDOM________________
    rand_sim = Simulation(rows, cols, num_steps, output, seed, [])
    print('______________RANDOM BASE AGENT RUNNING FOR 100 TIMES...___________')
    rand_sim.play_sim(rand, times)
    scores = rand_sim.scores
    print(f'random mean: {np.mean(scores):.2f}, random stdv: {np.std(scores):.2f}')
    rand_mean = round(np.mean(scores), 3)
    rand_std = round(np.std(scores), 3)
    agents_scores_dict['rand_mean'] = rand_mean
    agents_scores_dict['rand_std'] = rand_std

    #______________RUN PLAY GAME FOR SOFT DETERMINISTIC_______________
    soft_sim = Simulation(rows, cols, num_steps, output, seed, [])
    print('______________SOFT DETERMINISTIC BASE AGENT RUNNING FOR 100 TIMES...___________')
    soft_sim.play_sim(soft, times)
    scores = soft_sim.scores
    print(f'soft mean: {np.mean(scores):.2f}, soft stdv: {np.std(scores):.2f}')
    soft_mean = round(np.mean(scores), 3)
    soft_std = round(np.std(scores), 3)
    agents_scores_dict['soft_mean'] = soft_mean
    agents_scores_dict['soft_std'] = soft_std

    # # ______________RUN PLAY GAME FOR DETERMINISTIC_______________
    # det_sim = Simulation(rows, cols, num_steps, output, seed, [])
    # print('______________DETERMINISTIC BASE AGENT RUNNING FOR 100 TIMES...___________')
    # det_sim.play_sim(det, times)
    # scores = det_sim.scores
    # print(f'det mean: {np.mean(scores):.2f}, det stdv: {np.std(scores):.2f}')
    # det_mean = round(np.mean(scores), 3)
    # det_std = round(np.std(scores), 3)
    # agents_scores_dict['det_mean'] = det_mean
    # agents_scores_dict['det_std'] = det_std

    #________________RUN PLAY GAME FOR SARSA AGENT_________________
    sarsa_sim = Simulation(rows, cols, num_steps, output, seed, [])
    print('______________SARSA BASE AGENT RUNNING FOR 100 TIMES...___________')
    sarsa_sim.play_sim(sarsa_agent, times)
    scores = sarsa_sim.scores
    print(f'sarsa mean: {np.mean(scores):.2f}, sarsa stdv: {np.std(scores):.2f}')
    sarsa_mean = round(np.mean(scores), 3)
    sarsa_std = round(np.std(scores), 3)
    agents_scores_dict['sarsa_mean'] = sarsa_mean
    agents_scores_dict['sarsa_std'] = sarsa_std

    # ________________RUN PLAY GAME FOR Q-Learning AGENT_________________
    q_sim = Simulation(rows, cols, num_steps, output, seed, [])
    print('______________Q-LEARNING BASE AGENT RUNNING FOR 100 TIMES...___________')
    q_sim.play_sim(q_agent, times)
    scores = q_sim.scores
    print(f'q mean: {np.mean(scores):.2f}, q stdv: {np.std(scores):.2f}')
    q_mean = round(np.mean(scores), 3)
    q_std = round(np.std(scores), 3)
    agents_scores_dict['q_mean'] = q_mean
    agents_scores_dict['q_std'] = q_std

    with open('agentscores.json', 'w') as json_file:
        json.dump(agents_scores_dict, json_file)

def part4():
    # -----------Setting Board Parameters for Training and Baseline-------------------
    rows = 10
    cols = 10
    num_steps = 20
    output = False
    seed = 42
    expected_sarsa_error = {}
    # -------------Creating Board-------------------------
    test_space = Space.Space(10, 10, 30, output, seed)
    # ------------------Training TD Agent with expected SARSA--------------------
    print("____________Expected SARSA Running Alpha Tests_______________")

    print("EXPECTED SARSA TRAINING RUNNING FOR ALPHA 0.99...")
    copy_env = copy.deepcopy(test_space)
    SARSA_player, SARSA_diffs, SARSA_states, SARSA_num_episodes, SARSA_step, SARSA_TD_error_99 = expected_SARSA(copy_env, alpha=0.99)
    print("EXPECTED SARSA TRAINING FINISHED FOR ALPHA 0.99...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = SARSA_TD_error_99
    my_df.to_csv('SARSA_TD_ERROR_99.csv')

    print("EXPECTED SARSA TRAINING RUNNING FOR ALPHA 0.75...")
    copy_env = copy.deepcopy(test_space)
    SARSA_player, SARSA_diffs, SARSA_states, SARSA_num_episodes, SARSA_step, SARSA_TD_error_75 = expected_SARSA(
        copy_env, alpha=0.75)
    print("EXPECTED SARSA TRAINING FINISHED FOR ALPHA 0.75...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = SARSA_TD_error_75
    my_df.to_csv('SARSA_TD_ERROR_75.csv')

    print("EXPECTED SARSA TRAINING RUNNING FOR ALPHA 0.5...")
    copy_env = copy.deepcopy(test_space)
    SARSA_player, SARSA_diffs, SARSA_states, SARSA_num_episodes, SARSA_step, SARSA_TD_error_50 = expected_SARSA(
        copy_env, alpha=0.5)
    print("EXPECTED SARSA TRAINING FINISHED FOR ALPHA 0.5...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = SARSA_TD_error_50
    my_df.to_csv('SARSA_TD_ERROR_50.csv')

    print("EXPECTED SARSA TRAINING RUNNING FOR ALPHA 0.25...")
    copy_env = copy.deepcopy(test_space)
    SARSA_player, SARSA_diffs, SARSA_states, SARSA_num_episodes, SARSA_step, SARSA_TD_error_25 = expected_SARSA(
        copy_env, alpha=0.25)
    print("EXPECTED SARSA TRAINING FINISHED FOR ALPHA 0.25...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = SARSA_TD_error_25
    my_df.to_csv('SARSA_TD_ERROR_25.csv')

    print("EXPECTED SARSA TRAINING RUNNING FOR ALPHA 0.1...")
    copy_env = copy.deepcopy(test_space)
    SARSA_player, SARSA_diffs, SARSA_states, SARSA_num_episodes, SARSA_step, SARSA_TD_error_10 = expected_SARSA(
        copy_env, alpha=0.1)
    print("EXPECTED SARSA TRAINING FINISHED FOR ALPHA 0.1...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = SARSA_TD_error_10
    my_df.to_csv('SARSA_TD_ERROR_10.csv')



    # ------------------Training TD Agent with q-Learning--------------------

    print("____________Q Learning Running Alpha Tests_______________")

    print("Q LEARNING TRAINING RUNNING FOR ALPHA 0.99...")
    copy_env = copy.deepcopy(test_space)
    QL_player, QL_diffs, QL_states, QL_num_episodes, QL_step, QL_TD_error_99 = q_learning(
        copy_env, alpha=0.99)
    print("Q LEARNING TRAINING FINISHED FOR ALPHA 0.99...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = QL_TD_error_99
    my_df.to_csv('QL_TD_ERROR_99.csv')

    print("Q LEARNING TRAINING RUNNING FOR ALPHA 0.75...")
    copy_env = copy.deepcopy(test_space)
    QL_player, QL_diffs, QL_states, QL_num_episodes, QL_step, QL_TD_error_75 = q_learning(
        copy_env, alpha=0.75)
    print("Q LEARNING TRAINING FINISHED FOR ALPHA 0.75...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = QL_TD_error_75
    my_df.to_csv('QL_TD_ERROR_75.csv')

    print("Q LEARNING TRAINING RUNNING FOR ALPHA 0.5...")
    copy_env = copy.deepcopy(test_space)
    QL_player, QL_diffs, QL_states, QL_num_episodes, QL_step, QL_TD_error_50 = q_learning(
        copy_env, alpha=0.5)
    print("Q LEARNING TRAINING FINISHED FOR ALPHA 0.5...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = QL_TD_error_50
    my_df.to_csv('QL_TD_ERROR_50.csv')

    print("Q LEARNING TRAINING RUNNING FOR ALPHA 0.25...")
    copy_env = copy.deepcopy(test_space)
    QL_player, QL_diffs, QL_states, QL_num_episodes, QL_step, QL_TD_error_25 = q_learning(
        copy_env, alpha=0.25)
    print("Q LEARNING TRAINING FINISHED FOR ALPHA 0.25...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = QL_TD_error_25
    my_df.to_csv('QL_TD_ERROR_25.csv')

    print("Q LEARNING TRAINING RUNNING FOR ALPHA 0.1...")
    copy_env = copy.deepcopy(test_space)
    QL_player, QL_diffs, QL_states, QL_num_episodes, QL_step, QL_TD_error_10 = q_learning(
        copy_env, alpha=0.1)
    print("Q LEARNING TRAINING FINISHED FOR ALPHA 0.1...!")

    # This saves the TD error to a csv so we can look at Convergence
    print("Writing TD Error out to File")
    my_df = pd.DataFrame()
    my_df['TD_error'] = QL_TD_error_10
    my_df.to_csv('QL_TD_ERROR_10.csv')

def part3_test():

    # -----------Setting Board Parameters for Training and Baseline-------------------
    rows = 10
    cols = 10
    num_steps = 20
    output = False
    seed = 42
    times = 100
    agents_scores_dict = {}
    # ________________INITIALIZE THE BASE AGENTS________________
    rand = RandomAgent()
    det = Deterministic_Agent()
    soft = Soft_Deterministic_Agent()

    #________________RUN PLAY GAME FOR RANDOM________________
    rand_sim = Simulation(rows, cols, num_steps, output, seed, [])
    print('______________RANDOM BASE AGENT RUNNING FOR 100 TIMES...___________')
    rand_sim.play_sim(rand, times)
    scores = rand_sim.scores
    print(f'random mean: {np.mean(scores):.2f}, random stdv: {np.std(scores):.2f}')
    rand_mean = round(np.mean(scores), 3)
    rand_std = round(np.std(scores), 3)
    agents_scores_dict['rand_mean'] = rand_mean
    agents_scores_dict['rand_std'] = rand_std

    #______________RUN PLAY GAME FOR SOFT DETERMINISTIC_______________
    soft_sim = Simulation(rows, cols, num_steps, output, seed, [])
    print('______________SOFT DETERMINISTIC BASE AGENT RUNNING FOR 100 TIMES...___________')
    soft_sim.play_sim(soft, times)
    scores = soft_sim.scores
    print(f'soft mean: {np.mean(scores):.2f}, soft stdv: {np.std(scores):.2f}')
    soft_mean = round(np.mean(scores), 3)
    soft_std = round(np.std(scores), 3)
    agents_scores_dict['soft_mean'] = soft_mean
    agents_scores_dict['soft_std'] = soft_std

    # ______________RUN PLAY GAME FOR DETERMINISTIC_______________
    det_sim = Simulation(rows, cols, num_steps, output, seed, [])
    print('______________DETERMINISTIC BASE AGENT RUNNING FOR 100 TIMES...___________')
    det_sim.play_sim(det, times)
    scores = det_sim.scores
    print(f'det mean: {np.mean(scores):.2f}, det stdv: {np.std(scores):.2f}')
    det_mean = round(np.mean(scores), 3)
    det_std = round(np.std(scores), 3)
    agents_scores_dict['det_mean'] = det_mean
    agents_scores_dict['det_std'] = det_std


    with open('agentscores.json', 'w') as json_file:
        json.dump(agents_scores_dict, json_file)

if __name__ == '__main__':
    global SARSA_output
    global QL4_output

    #______________PART 1__________________
    #
    # SARSA4_output, QL4_output = part1()
    #
    # SARSA4_player, SARSA4_diffs, SARSA4_states, SARSA4_num_episodes, SARSA4_step, SARSA4_TD_error = SARSA4_output
    # filtered_SARSA4 = SARSA4_player.qtable
    #
    # # with open('SARSA4_Output.pickle', 'wb') as handle:
    # #     pickle.dump(filtered_SARSA4, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('SARSA4_Output.json', 'w') as json_file:
    #     json.dump(filtered_SARSA4, json_file)
    #
    # Q_player, Q_diffs, Q_states, Q_num_episodes, Q_step, Q_TD_error = QL4_output
    # filtered_QL4 = Q_player.qtable
    #
    # # with open('QL4_Output.pickle', 'wb') as handle:
    # #     pickle.dump(filtered_QL4, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('QL4_Output.json', 'w') as json_file:
    #     json.dump(filtered_QL4, json_file)
    #
    # # _______________PART 2_____________________
    #
    # sarsa_agent, q_agent = part2()
    #
    # part3(sarsa_agent, q_agent)

    # part4()

    # part3_test()





    rand = RandomAgent()
    rand_sim = Simulation(10, 10, 30, False, 42, [])
    rand_sim.play_sim_save_viz(rand, os.getcwd())
