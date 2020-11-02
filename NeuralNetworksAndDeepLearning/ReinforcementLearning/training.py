import dill
import numpy as np
import agent
import environment
import matplotlib.pyplot as plt
import random
import show
import os


###  Process
show_results = False
path = 'result/'

### Training parameters
# number of training episodes
episodes = 2000         
# maximum episode length
episode_length = 50     

### Environment
# dimension of the grid
x = 10                  
y = 10                  
# goal, wall, portal and sand
goal = [0,3]    
wall = [[2,1],[2,2],[2,3],[3,3],[3,4],[3,9],[3,8],[9,6],[7,6],[6,6],[6,6],[6,7],[6,8],[8,3],[7,0],[0,7]]
portal = [[0,6],[9,8]]
sand = [[2,0],[8,6],[4,3],[5,3],[7,1],[3,7],[9,3],[3,5],[1,7]]



# Training function

def training(run, episodes, wall, portal, sand, alpha, epsilon, discount, softmax=False, sarsa=False, filepath=None):

    #reward
    reward_final = np.zeros((100,run))
    reward_all = np.zeros((episodes,run))

    for time in range(run):
        # initialize the agent
        learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

        index_reward = 0
        # perform the training
        for index in range(0, episodes):
            # start from a random state
            initial = [np.random.randint(0, x), np.random.randint(0, y)]
            while(initial in wall) or (initial in sand) or (initial in portal) or (np.array(initial) == np.array(goal)).all():
                initial = [np.random.randint(0, x), np.random.randint(0, y)]

            # initialize environment
            state = initial
            env = environment.Environment(x, y, state, goal, wall, sand, portal)
            reward = 0
            # run episode
            for step in range(0, episode_length):
                # find state index
                state_index = state[0]*y+state[1]
                # choose an action
                action = learner.select_action(state_index, epsilon[index])
                # the agent moves in the environment
                result = env.move(action)
                # Q-learning update
                next_index = result[0][0] * y + result[0][1]
                learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
                # update state and reward
                reward += result[1]
                state = result[0]
            reward /= episode_length

            #record rewaed
            reward_all[index,time] = reward

            if index >= (episodes-100):
                reward_final[index_reward,time] = reward
                index_reward += 1

            print('Episode ', index + 1, ': the agent has obtained an average reward of ', reward, ' starting from position ', initial) 
        #print('Run ', time+1, ': the agent has obtained an average reward of ', np.mean(reward_final[:,time])) 
            ##periodically save the agent
            #if ((index + 1) % 10 == 0):
            #    with open('agent.obj', 'wb') as agent_file:
            #        dill.dump(agent, agent_file)

    if filepath is not None:
        np.save(path+filepath,np.mean(reward_all, axis=1))

    return learner, np.mean(reward_final), np.mean(reward_all, axis=1)


# Test function

def test(learner,initial,filepath=None):
    # Save results
    state_list = []
    gridworld_list = []

    # Initial state
    state = initial
    state_list.append(list(state))

    # Environment
    env = environment.Environment(x, y, state, goal, wall, sand, portal)
    gridworld_list.append(env.grid(0))

    # Run 
    for step in range(1, episode_length):

        # if the goal is reached stop the simulation
        if (np.array(state) == np.array(goal)).all():   
            # save results
            gridworld_list.append(env.grid(step))
            state_list.append(list(state))
            break 
    
        state_index = state[0] * y + state[1]
        # choose an action
        action = learner.select_action(state_index, 0)
        # the agent moves in the environment
        result = env.move(action)

        # update state
        state = result[0]
        # save results
        gridworld_list.append(env.grid(step))
        state_list.append(list(state))


    # show results
    if show_results:
        show.show_path(state_list,gridworld_list[0],portal)
        show.plot_path(state_list,gridworld_list[0],portal,path+filepath)
    else:
        show.show_gif(gridworld_list)


# Exponential decay for epsilon
     
def exp_decay(episodes):
    e=0.8
    factor = (0.001/e)**(1/float(episodes))
    epsilon = []
    epsilon.append(e)
    for i in range(episodes-1):
        e *= factor
        epsilon.append(e)
    return epsilon


# Run

if __name__ == "__main__":

    if show_results:

        # exponential discount factor
        discount = 0.9        

        # alpha and epsilon
        alpha = np.ones(episodes)*0.1
        epsilon_lin = np.linspace(0.8, 0.001, episodes)
        epsilon_exp = exp_decay(episodes)
        epsilon_list = [epsilon_lin,epsilon_exp]
        filename = ['LIN','EXP']

        # number of runs
        run = 30
        # training
        for i,epsilon in enumerate(epsilon_list):
            print('Decay:',filename[i])
            print('\n')
           
            # Q learning and epsilon greedy
            print('Q learning and epsilon greedy')
            alpha = [0.3,0.5]
            learner, reward_final, reward = training(run,episodes,wall, portal, sand, np.ones(episodes)*alpha[i], epsilon_exp, discount,False,False,filename[i]+'09FF')
            print('----------------------------------------')

            # Sarsa and epsilon greedy
            print('Sarsa and epsilon greedy')
            alpha = [0.5,0.5]
            learner, reward_final, reward = training(run,episodes,wall, portal, sand, np.ones(episodes)*alpha[i], epsilon_exp, discount,False,True,filename[i]+'09FT')
            print('----------------------------------------')  

            # Q learning and softmax
            print('Q learning and softmax')
            alpha = [0.3,0.3]
            learner, reward_final, reward = training(run,episodes,wall, portal, sand, np.ones(episodes)*alpha[i], epsilon_exp, discount,True,False,filename[i]+'09TF')
            print('----------------------------------------')   

            # Sarsa and softmax
            print('Sarsa and softmax')
            alpha = [0.1,0.3]
            learner, reward_final, reward = training(run,episodes,wall, portal, sand, np.ones(episodes)*alpha[i], epsilon_exp, discount,True,True,filename[i]+'09TT')
            print('----------------------------------------')  

            print('\n')

            # plot rewards
            FF = np.load(path+filename[i]+'09FF.npy')
            FT = np.load(path+filename[i]+'09FT.npy')
            TF = np.load(path+filename[i]+'09TF.npy')
            TT = np.load(path+filename[i]+'09TT.npy')
            plt.plot(FF,label='Q-learning, '+r'$\epsilon$'+'-Greedy')
            plt.plot(FT,label='SARSA, '+r'$\epsilon$'+'-Greedy')
            plt.plot(TF,label='Q-learning, Softmax')
            plt.plot(TT,label='SARSA, Softmax')
            plt.xlabel('Episode',fontsize=15)
            plt.ylabel('Average reward',fontsize=15)
            plt.legend(fontsize=15) 
        
            plt.savefig(path+filename[i]+'.png')       
            plt.close('all')


    else:
        #alpha and epsilon profile
        alpha = np.ones(episodes)*0.3
        epsilon= exp_decay(episodes)
        #discount
        discount=0.9
        learner, _, _ = training(1,episodes,wall, portal, sand, alpha, epsilon, discount)
        initial_list = [[5,1],[9,5],[5,9]]
        for initial in initial_list:
            test(learner,initial=initial)
            

