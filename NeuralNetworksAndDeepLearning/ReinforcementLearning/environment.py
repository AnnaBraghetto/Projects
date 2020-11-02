import numpy as np

class Environment:

    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }

    wall = []
    sand = []
    portal = [[1,1],[5,5]]
    
    def __init__(self, x, y, initial, goal, wall=[], sand=[], portal=[]):
        self.x = x
        self.y = y
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
        self.wall = wall
        self.sand = sand
        self.portal = portal
    
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        reward = 0
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if(self.check_boundaries(next_state)):
            reward = -1
        elif(self.check_wall(next_state)):
            reward = -1
        elif(self.check_sand(next_state)):
            reward = -0.5
            self.state = next_state
        elif(self.check_portal(next_state)):
            reward = -0.25
            self.state = self.out_portal(next_state)
        else:
            self.state= next_state
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0

    # check if the coordinates coincide with wall
    def check_wall(self, state):
        state_check = list(state)
        if state_check in self.wall: return True
        else: return False

    # check if the coordinates coincide with sand
    def check_sand(self, state):
        state_check = list(state)
        if state_check in self.sand: return True
        else: return False

    # check if the coordinates coincide with portal
    def check_portal(self, state):
        state_check = list(state)
        if state_check in self.portal: return True
        else: return False

    def out_portal(self, state):
        if (state == self.portal[0]).all(): return np.asarray(self.portal[1])
        elif (state == self.portal[1]).all(): return np.asarray(self.portal[0])


    # grid at each time step
    def grid(self, step):

        gridworld = np.zeros((self.x,self.y))
        for portal_coor in self.portal:
            gridworld[portal_coor[0],portal_coor[1]] = 2
        for sand_coor in self.sand:
            gridworld[sand_coor[0],sand_coor[1]] = 3
        for wall_coor in self.wall:    
            gridworld[wall_coor[0],wall_coor[1]] = 4
        gridworld[self.goal[0],self.goal[1]] = 1
        if step == 0:
            gridworld[self.state[0],self.state[1]] = 0
        else:
            gridworld[self.state[0],self.state[1]] = 5

        return gridworld




