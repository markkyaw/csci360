# autograder.py
#=============================================================
# Autograder for project 3 - MDP and Reinforcement Learning

# Created by Shihan Lu, Feb 21, 2021
# If you find any mistakes about this template, please contact
# Shihan Lu (shihanlu@usc.edu) or post on Piazza.
#=============================================================


import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from code_template import value_iteration, policy_evaluation, policy_iteration, q_learning_1, q_learning_2, q_learning_3


if __name__ == '__main__':
    np.random.seed(42)
    np.set_printoptions(suppress=True)

    # load maps and parameters
    maps_params = np.load('maps_params.npz', allow_pickle=True)

    num_cases = maps_params['num_cases']
    map = maps_params['maps']
    # print(map.item().get(0))
    gamma = maps_params['gamma']
    alpha = maps_params['alpha']
    epsilon = maps_params['epsilon']

    fileNames = ['autograder_reference'+str(i)+'.npz' for i in range(num_cases)]

    Q1_score, Q2_score, Q3_score, Q4_1_score, Q4_2_score, Q4_3_score, Q5_score = 0, 0, 0, 0, 0, 0, 0

    for i in range(num_cases):
        referece = np.load(fileNames[i]) # load references

        ##### Q1
        env = gym.make('FrozenLake-v0', desc=map.item().get(i))  # load map
        env.seed(42)
        policy1_ref = referece['policy1']
        policy1 = value_iteration(env, gamma=gamma[i], theta=0.0001)
        if np.shape(np.array(policy1)) == np.shape(np.array(policy1_ref)):
            if np.allclose(np.array(policy1), np.array(policy1_ref), rtol=0.01, atol=0.01):
                Q1_score +=1
            else:
                print('\nTask 1: failed test case %d' % (i+1))
                print('Your policy:\n', policy1)
                print('Expected policy:\n', policy1_ref)
        else:
            print('\nTask 1: failed test case %d' % (i+1))
            print('Your policy:\n', policy1)
            print('Expected policy:\n', policy1_ref)

        ##### Q2
        env = gym.make('FrozenLake-v0', desc=map.item().get(i))
        env.seed(42)
        nS = env.nS
        nA = env.nA
        value2_ref = referece['value2']
        policy2 = np.ones([nS, nA]) / nA
        value2 = policy_evaluation(policy2, env, gamma=gamma[i], theta=0.0001)
        if np.shape(np.array(value2)) == np.shape(np.array(value2_ref)):
            if np.allclose(np.array(value2), np.array(value2_ref), rtol=0.01, atol=0.01):
                Q2_score +=1
            else:
                print('\nTask 2: failed test case %d' % (i + 1))
                print('Your value:\n', value2)
                print('Expected value:\n', value2_ref)
        else:
            print('\nTask 2: failed test case %d' % (i+1))
            print('Your value:\n', value2)
            print('Expected value:\n', value2_ref)

        ##### Q3
        env = gym.make('FrozenLake-v0', desc=map.item().get(i))
        env.seed(42)
        policy3_ref = referece['policy3']
        policy3 = policy_iteration(env, gamma=gamma[i], theta=0.0001)
        if np.shape(np.array(policy3)) == np.shape(np.array(policy3_ref)):
            if np.allclose(np.array(policy3), np.array(policy3_ref), rtol=0.01, atol=0.01):
                Q3_score +=1
            else:
                print('\nTask 3: failed test case %d' % (i + 1))
                print('Your policy:\n', policy3)
                print('Expected policy:\n', policy3_ref)
        else:
            print('\nTask 3: failed test case %d' % (i+1))
            print('Your policy:\n', policy3)
            print('Expected policy:\n', policy3_ref)

        ##### Q4-1
        env = gym.make('FrozenLake-v0', desc=map.item().get(i))
        env.seed(42)
        Q4_1_ref = referece['Q4_1']
        policy4_1_ref = referece['policy4_1']
        Q4_1, policy4_1 = q_learning_1(env, alpha=alpha[i], gamma=gamma[i], epsilon=epsilon[i], num_episodes=500)
        if np.shape(np.array(Q4_1)) == np.shape(np.array(Q4_1_ref)) and \
                np.shape(np.array(policy4_1)) == np.shape(np.array(policy4_1_ref)):
            if np.allclose(np.array(Q4_1), np.array(Q4_1_ref), rtol=0.01, atol=0.01) and \
                    np.allclose(np.array(policy4_1), np.array(policy4_1_ref), rtol=0.01, atol=0.01):
                Q4_1_score +=1
            else:
                print('\nTask 4-1: failed test case %d' % (i + 1))
                print('Your Q table:\n', Q4_1)
                print('Expected Q table:\n', Q4_1_ref)
                print('Your policy:\n', policy4_1)
                print('Expected policy:\n', policy4_1_ref)
        else:
            print('\nTask 4-1: failed test case %d' % (i+1))
            print('Your Q table:\n', Q4_1)
            print('Expected Q table:\n', Q4_1_ref)
            print('Your policy:\n', policy4_1)
            print('Expected policy:\n', policy4_1_ref)

        ##### Q4-2
        env = gym.make('FrozenLake-v0', desc=map.item().get(i))
        env.seed(42)
        Q4_2_ref = referece['Q4_2']
        policy4_2_ref = referece['policy4_2']
        Q4_2, policy4_2 = q_learning_2(env, alpha=alpha[i], gamma=gamma[i], epsilon_0=epsilon[i], epsilon_min=0.01,
                                       num_episodes=500)
        if np.shape(np.array(Q4_2)) == np.shape(np.array(Q4_2_ref)) and \
                np.shape(np.array(policy4_2)) == np.shape(np.array(policy4_2_ref)):
            if np.allclose(np.array(Q4_2), np.array(Q4_2_ref), rtol=0.01, atol=0.01) and \
                    np.allclose(np.array(policy4_2), np.array(policy4_2_ref), rtol=0.01, atol=0.01):
                Q4_2_score +=1
            else:
                print('\nTask 4-2: failed test case %d' % (i + 1))
                print('Your Q table:\n', Q4_2)
                print('Expected Q table:\n', Q4_2_ref)
                print('Your policy:\n', policy4_2)
                print('Expected policy:\n', policy4_2_ref)
        else:
            print('\nTask 4-2: failed test case %d' % (i+1))
            print('Your Q table:\n', Q4_2)
            print('Expected Q table:\n', Q4_2_ref)
            print('Your policy:\n', policy4_2)
            print('Expected policy:\n', policy4_2_ref)

        ##### Q4-3
        env = gym.make('FrozenLake-v0', desc=map.item().get(i))
        env.seed(42)
        Q4_3_ref = referece['Q4_3']
        policy4_3_ref = referece['policy4_3']
        Q4_3, policy4_3 = q_learning_3(env, alpha=alpha[i], gamma=gamma[i], epsilon=epsilon[i], num_episodes=500)
        if np.shape(np.array(Q4_3)) == np.shape(np.array(Q4_3_ref)) and \
                np.shape(np.array(policy4_3)) == np.shape(np.array(policy4_3_ref)):
            if np.allclose(np.array(Q4_3), np.array(Q4_3_ref), rtol=0.01, atol=0.01) and \
                    np.allclose(np.array(policy4_3), np.array(policy4_3_ref), rtol=0.01, atol=0.01):
                Q4_3_score +=1
            else:
                print('\nTask 4-3: failed test case %d' % (i + 1))
                print('Your Q table:\n', Q4_3)
                print('Expected Q table:\n', Q4_3_ref)
                print('Your policy:\n', policy4_3)
                print('Expected policy:\n', policy4_3_ref)
        else:
            print('\nTask 4-3: failed test case %d' % (i+1))
            print('Your Q table:\n', Q4_3)
            print('Expected Q table:\n', Q4_3_ref)
            print('Your policy:\n', policy4_3)
            print('Expected policy:\n', policy4_3_ref)


    print('\nTest Result Report:')
    print('Q1: %d/%d'% (Q1_score, num_cases))
    print('Q2: %d/%d'% (Q2_score, num_cases))
    print('Q3: %d/%d'% (Q3_score, num_cases))
    print('Q4-1: %d/%d'% (Q4_1_score, num_cases))
    print('Q4-2: %d/%d' % (Q4_2_score, num_cases))
    print('Q4-3: %d/%d' % (Q4_3_score, num_cases))
    print('***Note: This is not your final score for Task 1-4. Your implementation will be tested and graded by different but '
          'similar maps and parameters after the submission.')

