import dill
import numpy as np
import agent
import environment
import matplotlib.pyplot as plt
import random

def show_gif(gridworld_list):
    plt.close('all')
    # cmap plot
    cmap = plt.get_cmap('YlGnBu', 6.)
        
    for grid in gridworld_list:
        fig, ax = plt.subplots()
        mat = ax.matshow(grid,cmap=cmap, vmin = -0.5, vmax = 5.5)
        #tell the colorbar to tick at integers
        cax = plt.colorbar(mat, ticks=np.arange(0, 6))
        cax.ax.set_yticklabels(['Path', 'Goal', 'Wall', 'Sand', 'Portal','State'])
        plt.pause(0.05)
        plt.close()

def plot_path(state_list, environment, portal,filepath):

    plt.close('all')
    # cmap plot
    cmap = plt.get_cmap('YlGnBu', 5.)

    # path
    fig, ax = plt.subplots()
    mat = ax.matshow(environment,cmap=cmap, vmin = -0.5, vmax = 4.5)
    cax = plt.colorbar(mat, ticks=np.arange(0, 5))
    cax.ax.set_yticklabels(['Path', 'Goal', 'Portal', 'Sand', 'Wall'])
    plt.savefig('environment.png')

    plt.close('all')
    
    # path
    fig, ax = plt.subplots()
    mat = ax.matshow(environment,cmap=cmap, vmin = -0.5, vmax = 4.5)
    cax = plt.colorbar(mat, ticks=np.arange(0, 5))
    cax.ax.set_yticklabels(['Path', 'Goal', 'Portal', 'Sand', 'Wall'])
  
    if (portal[0] in state_list):
        portal_index = 0
        while (state_list[portal_index] != portal[0]):
            portal_index +=1

        state_list = np.array(state_list).reshape(len(state_list),2)
        # start
        plt.plot(state_list[0,1], state_list[0,0], 'o', markersize=17, color='black')
        # finish
        plt.plot(state_list[-1,1], state_list[-1,0], "*", markersize=20, color='black')
        
        #path
        from_start_to_portal = list(state_list[:portal_index,:])
        from_start_to_portal.append(portal[1])
        from_start_to_portal = np.array(from_start_to_portal).reshape(len(from_start_to_portal),2)

        plt.plot(from_start_to_portal[:,1], from_start_to_portal[:,0], "-", linewidth=2.5, color='black')
        plt.plot(state_list[portal_index:,1], state_list[portal_index:,0], "-", linewidth=2.5, color='black')
        # portal
        plt.plot(portal[0][1], portal[0][0], "P", markersize=17, color='black')
        plt.plot(portal[1][1], portal[1][0], "P", markersize=17, color='black')
        
    else:
        state_list = np.array(state_list).reshape(len(state_list),2)
        # start
        plt.plot(state_list[0,1], state_list[0,0], 'o', markersize=17, color='black')
        # finish
        plt.plot(state_list[-1,1], state_list[-1,0], "*", markersize=20, color='black')
        #path
        plt.plot(state_list[:,1], state_list[:,0], "-", linewidth=2.5, color='black')
    
    plt.savefig(filepath+'.png')
    plt.close('all')

def show_path(state_list, environment,portal):

    plt.close('all')
    # cmap plot
    cmap = plt.get_cmap('YlGnBu', 5.)
    
    # path
    fig, ax = plt.subplots()
    mat = ax.matshow(environment,cmap=cmap, vmin = -0.5, vmax = 4.5)
    cax = plt.colorbar(mat, ticks=np.arange(0, 5))
    cax.ax.set_yticklabels(['Path', 'Goal', 'Portal', 'Sand', 'Wall'])
  
    if (portal[0] in state_list):
        portal_index = 0
        while (state_list[portal_index] != portal[0]):
            portal_index +=1

        state_list = np.array(state_list).reshape(len(state_list),2)

        # start
        plt.plot(state_list[0,1], state_list[0,0], 'o', markersize=17, color='black')
        # finish
        plt.plot(state_list[-1,1], state_list[-1,0], "*", markersize=20, color='black')
        
        #path
        from_start_to_portal = list(state_list[:portal_index,:])
        from_start_to_portal.append(portal[1])
        from_start_to_portal = np.array(from_start_to_portal).reshape(len(from_start_to_portal),2)
        plt.plot(from_start_to_portal[:,1], from_start_to_portal[:,0], "-", linewidth=2.5, color='black')
        plt.plot(state_list[portal_index:,1], state_list[portal_index:,0], "-", linewidth=2.5, color='black')

        # portal
        plt.plot(portal[0][1], portal[0][0], "P", markersize=17, color='black')
        plt.plot(portal[1][1], portal[1][0], "P", markersize=17, color='black')
        
    else:
        state_list = np.array(state_list).reshape(len(state_list),2)
        # start
        plt.plot(state_list[0,1], state_list[0,0], 'o', markersize=17, color='black')
        # finish
        plt.plot(state_list[-1,1], state_list[-1,0], "*", markersize=20, color='black')
        #path
        plt.plot(state_list[:,1], state_list[:,0], "-", linewidth=2.5, color='black')
    
    plt.show()
    plt.close('all')

