import dill
import numpy as np
import agent
import environment
import matplotlib.pyplot as plt
import random
import training

path = 'grid_search/'

gd = False

def grid_search(bool,discount,softmax,sarsa,filepath):

    episodes = 2000         # number of training episodes

    alpha_list = [0.1,0.3,0.5]
    epsilon_list = ['Lin', 'Exp']
    epsilon_list_values = [np.linspace(0.8, 0.001, episodes),training.exp_decay(episodes)]

    if bool:
        
        print('Discount:',discount, 'Softmax:',softmax,'Sarsa:',sarsa)

        # wall
        wall = [[2,1],[2,2],[2,3],[3,3],[3,4],[3,9],[3,8],[9,6],[7,6],[6,6],[6,6],[6,7],[6,8],[8,3],[7,0]]
        portal = [[0,6],[9,8]]
        sand = [[2,0],[8,6],[4,3],[5,3],[7,1],[3,7],[9,3]]

        reward_val = np.zeros((2,3))

        for a,alpha in enumerate(alpha_list):
            alpha_values = np.ones(episodes)*alpha
            for e,epsilon_values in enumerate(epsilon_list_values):
                _, reward, _ = training.training(10,episodes,wall, portal, sand, alpha_values, epsilon_values, discount,softmax, sarsa)
                reward_val[e,a] = reward
                print('Alpha:',alpha,'Epsilon:',epsilon_list[e],'Reward:',reward)

        print('--------------------------------------------------')
        print('\n')

        np.save(path+filepath,reward_val)
    reward_val = np.load(path+filepath+'.npy')
    plot_map(reward_val,alpha_list,epsilon_list,filepath)


def plot_map(matrix, x, y, filepath):
    plt.close('all')
    cmap = 'YlGnBu'
    fig, ax = plt.subplots()
    mat = ax.matshow(matrix,cmap=cmap)
    cax = plt.colorbar(mat)
    cax.ax.tick_params(labelsize=20)
    ax.set_xlabel('Alpha',fontsize=22)
    ax.set_ylabel('Epsilon decay',fontsize=22) 
    ax.set_xticklabels(['']+x,fontsize=20)
    ax.set_yticklabels(['']+y,fontsize=20,rotation=90)

    # put text on matrix elements
    for i, x_coor in enumerate(np.arange(len(x))):
        for j, y_coor in enumerate(np.arange(len(y))):
            reward = "${0:.3f}\\%$".format(matrix[j,i])
            ax.text(x_coor, y_coor, reward, va='center', ha='center',color='red',fontsize=20)
    plt.savefig(path+filepath+'.png')

if __name__ == "__main__":

    grid_search(gd,0.9,softmax=False,sarsa=False,filepath='09FF')

    grid_search(gd,0.9,softmax=True,sarsa=False,filepath='09TF')

    grid_search(gd,0.9,softmax=False,sarsa=True,filepath='09FT')

    grid_search(gd,0.9,softmax=True,sarsa=True,filepath='09TT')
  