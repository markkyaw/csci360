# validate.py
#=============================================================
# validate.py for project 3 - MDP and Reinforcement Learning

# Created by Shihan Lu, Feb 21, 2021
# If you find any mistakes about this template, please contact
# Shihan Lu (shihanlu@usc.edu) or post on Piazza.
#=============================================================


import argparse
import gym
import numpy as np
import time
from code_template import q_learning_cart, get_discrete_state

parser = argparse.ArgumentParser(description='Select a particular question')
parser.add_argument("-train", "--train",
                    help="visualize the lake and execution",
                    action="store_true")
parser.add_argument("-test", "--test",
                    help="visualize the lake and execution",
                    action="store_true")
args = parser.parse_args()

np.random.seed(42) # seed numpy - do not change this line
env = gym.make('CartPole-v0').unwrapped
env.seed(42)  # seed env - do not change this line
env.reset()

if args.train:
    print('Run q-learning...')
    Q = q_learning_cart(env)

    # save the q-table
    print('Save the q-table...')
    np.save('q_table_cartpole.npy', Q)

    print('Test your policy from q-learning for CartPole...')
    env = gym.make('CartPole-v0').unwrapped  # reload the environment for reproducibility
    env.seed(42)  # seed env - do not change this line
    env.reset()
    num_test = 10
    t_total = 0
    step_total = 0
    if np.shape(np.array(Q)) != (30, 30, 50, 50, env.action_space.n):
        print('Q-table is in the wrong shape.')
    else:
        for i in range(num_test):
            env.reset()
            current_state = get_discrete_state(env.reset())
            t0 = time.time() # initial time
            step = 0
            while True:
                step += 1
                env.render()
                action = np.argmax(Q[current_state])
                next_state, reward, done, _ = env.step(action)
                current_state = get_discrete_state(next_state)

                if done or step >= 2000:
                    t1 = time.time()
                    t = t1 - t0
                    t_total += t
                    step_total += step
                    print('time: %.2f' %t, end=' ')
                    print('step:', step)
                    break
        env.close()

    print('Average time:', t_total/num_test)
    print('Average step:', step_total/num_test)
    print('\nTask 5 Result Report:')
    avg_step = step_total / num_test
    if avg_step < 50:
        print('Score: 0/3')
    elif avg_step >= 50 and avg_step < 200:
        print('Score: 1/3')
    elif avg_step >= 200 and avg_step < 500:
        print('Score: 2/3')
    elif avg_step >= 500:
        print('Score: 3/3')

if args.test:
    # load the q-table
    print('Load the q-table...')
    Q = np.load('q_table_cartpole.npy')

    print('Test your policy from q-learning for CartPole...')
    num_test = 10
    t_total = 0
    step_total = 0
    if np.shape(np.array(Q)) != (30, 30, 50, 50, env.action_space.n):
        print('Q-table is in the wrong shape.')
    else:
        for i in range(num_test):
            env.reset()
            current_state = get_discrete_state(env.reset())
            t0 = time.time()  # initial time
            step = 0
            while True:
                step += 1
                env.render()
                action = np.argmax(Q[current_state])
                next_state, reward, done, _ = env.step(action)
                current_state = get_discrete_state(next_state)

                if done or step >= 2000:
                    t1 = time.time()
                    t = t1 - t0
                    t_total += t
                    step_total += step
                    print('time: %.2f' % t, end=' ')
                    print('step:', step)
                    break
        env.close()

    print('Average time:', t_total / num_test)
    print('Average step:', step_total / num_test)

    print('\nTask 5 Result Report:')
    avg_step = step_total / num_test
    if avg_step < 50:
        print('Score: 0/3')
    elif avg_step >= 50 and avg_step < 200:
        print('Score: 1/3')
    elif avg_step >=200 and avg_step < 500:
        print('Score: 2/3')
    elif avg_step >= 500:
        print('Score: 3/3')