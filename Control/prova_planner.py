import numpy as np
import sympy
import math
import sys
import os
import time

import logging
logger = logging.getLogger('logger1')
logger.setLevel(logging.INFO)  # Imposta il livello di log a ERROR
handler1 = logging.StreamHandler()  # Invia i log a stdout
formatter1 = logging.Formatter('%(levelname)s: %(message)s')
handler1.setFormatter(formatter1)
logger.addHandler(handler1)
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# logger2 = logging.getLogger('logger2')
# logger2.setLevel(logging.ERROR)  # Imposta il livello di log a ERROR
# handler2 = logging.FileHandler('logger2.log')  # Salva i log in un file
# formatter2 = logging.Formatter('%(levelname)s: %(message)s')
# handler2.setFormatter(formatter2)
# logger2.addHandler(handler2)

# Get the directory containing the current file
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Current folder:", project_folder)

sys.path.append(project_folder)
from decimal import Decimal

# sys.path.append(r'C:\Users\Luca\AMR23-FP7-MPSWMR\AMR23-FP7-MPSWMR')#ognuno mette il suo
# sys.path.append(r'C:\Users\ciarp\OneDrive\Desktop\progetto-amr\AMR23-FP7-MPSWMR')#ognuno mette il suo

from Model.model_creator import * 
from Utils.utils import * 
from World.occupancyGridMap import OccupancyGridMap

MIN_V1 = 1e-6
R_DIMENSION = 3
RESOLUTION = 1
N_ITERS = 10
TAU_STEP = 0.1
FREQUENCY = 10

class Node():
    def __init__(self, state):
        self.state = state
        self.x = state[0]
        self.y = state[1]
        self.position = np.array([self.x, self.y])

        self.parent = None
        self.children = []
        self.cost = None
        
        self.trajectory = [self.state]
        self.inputs = np.array([0,0,0])

        self.name = ""
        return
    
    def __str__(self):
        return f"state: {self.state}, cost: {self.cost}, name: {self.name}"

class RRT_STAR():
    def __init__(self, 
                 root_state, 
                 goal_state,
                 map,
                 step_meter):

        self.root_node = Node(root_state)
        self.goal_node = Node(goal_state)

        self.random_tree = [self.root_node]

        self.map = map

        self.step_m = step_meter # distance performed by the robot in 1 time_step
        self.distance_tol = 0.001

        expanded_q={
        'x':1,
        'y':1,
        'θ':1,
        'ϕ_1':1,
        'v_1':1,
        'ω':1
        }
        my_u="prova"
        self.model = Dynamic_Model(expanded_q,my_u)

        return
    
    def rrt_plan(self):

        history = []
        total_cost = 0

        for iter in range(N_ITERS):

            logger.info(f"ITERATION N°: {iter}")

            n_rand = self.generate_random_node()
            logger.debug(f'RANDOM: {n_rand.state}')
        
            n_nearest, n_new = self.get_NN(n_rand)
            logger.debug(f'nearest: {n_nearest}, new: {n_new.state}')            
        
            cost, path, inputs, tau_star = self.optimal_trajectory(n_nearest.state, n_new.state, TAU_STEP, FREQUENCY) # From nearest to new
            
            if cost <= 0 : 
                logger.info("COSTI NEGATIVI")
                continue
            
            collision_or_ranges = False

            for i, state_i in enumerate(path):
                # print(f'state_i: {state_i}, inputs_i: {inputs[i]}')
                check_ranges = self.check_admissible_ranges(state_i, inputs[i])
                check_collision = self.check_free_collision(state_i)

                if not check_ranges:
                    logger.warning(f"Range found, state_i: {state_i}, inputs_i: {inputs[i]}, from: {n_nearest}, to: {n_new}")
                    collision_or_ranges = True
                    break

                if not check_collision:
                    logger.warning("Collision found, state_i: {state_i}, inputs_i: {inputs[i]}")
                    collision_or_ranges = True
                    break

                if state_i[4] < MIN_V1:
                    logger.warning(f"Velocity near to zero, state_i: {state_i}, inputs_i: {inputs[i]}")
                    collision_or_ranges = True
                    break
                
            if collision_or_ranges:
                continue
            
            history.append({
                'cost': cost,
                'tau': tau_star,
                'path': path,
                'inputs': inputs,
                'iter': iter,
                'index_node': len(self.random_tree)
            })

            nearest_nodes = self.get_multiple_NN(n_new)

            parent, cost_from_parent, traj_from_parent, inputs_from_parent = self.choose_parent(n_new, nearest_nodes)
            
            if cost_from_parent <= 0: 
                logger.info("COSTI NEGATIVI choose_parent")
                continue

            logger.debug(f'parent: {parent}, n_new: {n_new}')

            if parent == None:
                logger.info(f"Parent None, parent: {parent}, n_new: {n_new}")
                continue

            n_new.cost = cost_from_parent + parent.cost
            n_new.parent = parent
            n_new.name = "Node " + str(len(self.random_tree))
            n_new.trajectory = traj_from_parent
            n_new.inputs = inputs_from_parent

            self.random_tree.append(n_new)
            
            if not self.update_costs(n_new, nearest_nodes):
                logger.info("COSTI NEGATIVI update_costs")
                continue
            
            logger.info(f'node added: {n_new}')
            # if n_new.x - self.distance_tol <= self.goal_node.x <= n_new.x + self.distance_tol and n_new.y - self.distance_tol <= self.goal_node.y <= n_new.y + self.distance_tol:
            #     #reached a node near the goal node
            #     break
        
        nearest_goal_node = self.get_nearest_goal()
        goal_node = self.check_goal_path(nearest_goal_node)
        complete_path = self.get_complete_path(goal_node)

        print("End planning")
            
        return complete_path
    
    def get_nearest_goal(self):

        dist_list = np.array([self.euclidean_distance(node, self.goal_node)
                             for node in self.random_tree])
        
        min_idx = np.argmin(dist_list)
        min_cost = self.random_tree[min_idx].cost

        for n in range(len(dist_list)):
            if dist_list[n] == min(dist_list) and self.random_tree[n].cost < min_cost:
                
                min_cost = self.random_tree[n].cost
                min_idx = n
    
        nearest_goal = self.random_tree[min_idx]

        return nearest_goal
    
    def check_goal_path(self, nearest_goal):
        cost, path, inputs, tau_star = self.optimal_trajectory(nearest_goal.state, self.goal_node.state, TAU_STEP, FREQUENCY) # From nearest to new
        if cost <=0:
            return nearest_goal
        
        collision_or_ranges = False

        for i, state_i in enumerate(path):
            logger.debug(f'state_i: {state_i}, inputs_i: {inputs[i]}')
            check_ranges = self.check_admissible_ranges(state_i, inputs[i])
            check_collision = self.check_free_collision(state_i)

            if not check_ranges:
                logger.warn("Range found")
                collision_or_ranges = True
                break

            if not check_collision:
                logger.warn("Collision found")
                collision_or_ranges = True
                break

            
        if collision_or_ranges:
            return nearest_goal
        return nearest_goal
        

        nearest_nodes = self.get_multiple_NN(self.goal_node)

        parent, cost_from_parent, traj_from_parent, inputs_from_parent = self.choose_parent(self.goal_node, nearest_nodes)

        logger.debug(f'parent: {parent}, self.goal_node: {self.goal_node}')

        if parent == None:
            logger.debug("Parent None")
            return nearest_goal
        
        self.goal_node.cost = cost_from_parent + parent.cost
        self.goal_node.parent = parent

        self.random_tree.append(self.goal_node)
        
        self.update_costs(self.goal_node, nearest_nodes)

        return self.goal_node
    
    def get_complete_path(self, goal_node):
        logger.info("START get_complete_path")

        path = []
        n_curr = goal_node

        logger.debug(f"goal_node: {goal_node}")

        while n_curr.parent != None:
            path.insert(0, n_curr)
            n_curr = n_curr.parent
            logger.debug(f"n_curr: {n_curr}")

        path.insert(0, n_curr)

        return path

    def optimal_trajectory(self,q0,q1,tau_step,frequency):
        logger.debug("START optimal_trajectory")

        self.model.set_state(q0)

        tau = 0.0
        iter_star = 0
        tau_star=0
        iter_max = 100000

        c = np.zeros((iter_max))
        q_bar = np.zeros((iter_max, *self.model.get_vector_state().shape))
        G = np.zeros((iter_max, *self.model.get_linearized_A().shape))

        c[iter_star] = np.inf
        G[0] = np.zeros((self.model.get_linearized_A().shape))
        iter = 0
        q_bar[0] = q0
        while tau < c[iter_star] and iter<iter_max:
            
            tau+= tau_step
            iter += 1
            q_bar[iter] = self.RK4(self.q_bar_d, q_bar[iter-1], tau_step,frequency)
            G[iter] = self.RK4(self.G_d, G[iter-1], tau_step,frequency)
            c[iter] = self.cost_fn(G[iter], q1 ,q_bar[iter], tau)

            if c[iter] < c[iter_star]:
                if c[iter]>0:
                    iter_star = iter
                    tau_star = tau
        d_star = np.linalg.pinv(G[iter_star]) @ (q1 - q_bar[iter_star])
        state_f = np.concatenate([q1, d_star])

        temp_path, temp_y = self.back_RK4(self.ODE_state, state_f, tau_star,frequency) #valutare se fare anche qua come sopra

        # computed_tau=np.arange(0,len(temp_path),frequency*tau_step, dtype=int)
        # if len(temp_path)-1 not in computed_tau:
        #     computed_tau=np.concatenate((computed_tau,np.array([len(temp_path)-1])))
        # path=[temp_path[i] for i in computed_tau]
        # y=[temp_y[i] for i in computed_tau]

        RB = np.linalg.inv(self.R) @ self.model.get_linearized_B().T
        temp_inputs=[RB @temp_y[i] for i in range(len(temp_y))]
        #inputs = [RB @ y[i] for i in range(len(y))]
        
        logger.debug("END optimal_trajectory")
        return c[iter_star],temp_path,temp_inputs,tau_star

    def RK4(self,f,y0,tf,freq): #tf mi dice quali sono i secondi finali
        t = np.arange(0, tf, 1/(freq))
        if tf not in t:
            t = np.concatenate((t, np.array([tf])))
        n = len(t)
        y = np.zeros((n, *y0.shape))
        y[0] = y0
        for i in range(n - 1):
            h = t[i+1] - t[i] #h sarebbe l'intervallo do tempo. se h=2, vuol dire che passano due secondi
            k1=f(y[i])
            k2=f(y[i] +k1*h/2.)
            k3=f(y[i] +k2*h/2.)
            k4=f(y[i] +k3*h)
            y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)

        return y[-1]

    def back_RK4(self,f,final_state,tf,freq):
        t = np.arange(0, tf, 1/(freq))
        if tf not in t:
            t = np.concatenate((t, np.array([tf])))
        n=len(t)
        state=np.zeros((n,*final_state.shape))
        state[0]=final_state
        for i in range(n-1):
            h = t[i+1] - t[i]
            k1 = f(state[i])
            k2 = f(state[i] - k1 * h / 2.)
            k3 = f(state[i] - k2 * h / 2.)
            k4 = f(state[i] - k3 * h)
            state[i+1] = state[i] - (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
        #poi qui dobbiamo flippare il valore state e suddividerlo in q e y
        flipped_state=state[::-1]
        q=flipped_state[:,:self.model.get_vector_state().shape[0]]
        y=flipped_state[:,self.model.get_vector_state().shape[0]:]
        return q,y
    
    
    def q_bar_d(self,q_):
        f=self.model.get_linearized_A()@q_ + self.model.get_linearized_c()
        return np.array(f)
    
    def G_d(self,G_):
        G_d = self.model.get_linearized_A() @ G_ + G_ @ self.model.get_linearized_A().T + self.model.get_linearized_B() @ np.linalg.inv(self.R) @ self.model.get_linearized_B().T #transition function
        return np.array(G_d)
    
    def ODE_state(self, state):
    
        q_d = self.model.get_linearized_A() @ state[:self.model.get_vector_state().shape[0]] + self.model.get_linearized_B() @ np.linalg.inv(self.R) @ self.model.get_linearized_B().T @ state[self.model.get_vector_state().shape[0]:] + self.model.get_linearized_c()
        y_d = - self.model.get_linearized_A().T @ state[self.model.get_vector_state().shape[0]:]

        return np.concatenate([q_d, y_d])
    

    def cost_fn(self, G, q1, q_bar, t):
        '''      
            Cost function   
        '''
        
        c = t + (q1 - q_bar).T @ np.linalg.pinv(G) @ (q1 - q_bar)
        
        return c
    
    def d_cost_fn(self, A, B, R, c, G, x1, xb):
        '''
            Derivative of the cost function  
        '''

        d_t = np.linalg.inv(G) @ (x1 - xb)
        c_d = 1 - 2*(A @ x1 + c).T @ d_t - d_t.T @ B @ np.linalg.inv(R) @ B.T @ d_t

        return c_d #[0][0]
    
    def get_states(self):
        basic_states = self.model.get_state()

        R = np.eye(R_DIMENSION)
        return basic_states, R
    
    def get_NN(self, current_node):
        ''' 
            Compute the nearest neighbor
        '''
        logger.debug("Start get_NN")

        dist_list = np.array([self.euclidean_distance(current_node, tree_node)
                             for tree_node in self.random_tree])
        
        idx_min = np.argmin(dist_list)
        dist_min = np.min(dist_list)

        nearest = self.random_tree[idx_min]
        dist = min(dist_min, self.step_m)

        theta = math.atan2(current_node.y - nearest.y, current_node.x - nearest.x)

        #TODO: fix state based on dict or array, now is assumed to be an array: [x, y, θ, ϕ_1, v_1, ω]
        new_state = np.array([(nearest.x + dist*math.cos(theta)), 
                              (nearest.y + dist*math.sin(theta)),
                              current_node.state[2], 
                              current_node.state[3], 
                              current_node.state[4], 
                              current_node.state[5]])
        
        new_node = Node(new_state)

        logger.debug("End get_NN")

        return nearest, new_node
    
    def get_multiple_NN(self, current_node):
        ''' 
            Compute the list of the nearest neightbors
        '''
        logger.debug("Start get_multiple_NN")

        distances = np.array([self.euclidean_distance(current_node, tree_node)
                             for tree_node in self.random_tree])
        
        number_neighbours = 10 if len(distances) > 10 else len(distances)
        distances_argsorted = distances.argsort()[:number_neighbours]
        
        nearest_nodes = [self.random_tree[i] for i in distances_argsorted]
        
        logger.debug("End get_multiple_NN")
        return nearest_nodes
    
    def generate_random_node(self):
        ''' Generate a random node '''
        
        x_new = np.random.uniform(low = 0, high = self.map.width * self.map.resolution)
        y_new = np.random.uniform(low = 0, high = self.map.height * self.map.resolution)
        theta_new = np.random.uniform(low = 0, high = 2*math.pi)
        # phi_new = np.random.uniform(low = 0.1, high = 0.3)
        # v_new = np.random.uniform(low = 0.1, high = 0.3)
        # omega_new = np.random.uniform(low = 0.1, high = 0.3)
        phi_new = np.random.uniform(low = 0.5, high = 3.)
        v_new = np.random.uniform(low = 0.5, high = 3.)
        omega_new = np.random.uniform(low = 0.5, high = 3.)
        
            
        state = np.array([x_new, y_new, theta_new, phi_new, v_new, omega_new])

        
        return Node(state)



    def euclidean_distance(self, n1, n2):
        dist = np.linalg.norm(n1.position - n2.position)
        return dist


    # assumo che state sia = [x, y, θ, ϕ_1, v_1, ω]
    def check_in_map(self, state):
        '''
        Return True if the position (x,y) is within the map limits
        '''

        resolution = self.map.resolution

        x = state[0] / resolution
        y = state[1] / resolution

        if 0 <= x < self.map.width and 0 <= y <= self.map.height:
            return True

        return False

    # assumo che state sia = [x, y, θ, ϕ_1, v_1, ω]
    # assumo che inputs sia = [v_ϕ_1 a_v_1 a_ω]
    def check_admissible_ranges(self, state, inputs):
        '''
        Check if the velocity values are whithin their admissible ranges
        '''
        logger.debug("START check_admissible_ranges")
        #?
        if not self.check_in_map(state) or not (self.min_v1 < state[4] < self.max_v1 and self.min_om <= state[5] <= self.max_om):
            
            logger.debug("END check_admissible_ranges")
            return False
        
        if not ( self.min_vf1 <= inputs[0] <= self.max_vf1 and
            self.min_av1 <= inputs[1] <= self.max_av1 and 
            self.min_aom <= inputs[2] <= self.max_aom) :

                logger.debug("END check_admissible_ranges")
                return False
        
        logger.debug("END check_admissible_ranges")
        return True
                
    #
    # inserire length e threshold come parametri della classe
    def check_free_collision(self, state,):
        '''
        Check if the robot in this configuration is free of collisions with any obstacle
        
        self.length is the length of the side a max square that can contain the robot
        self.threshold is a small margin to add to the surface occupied by the robot
        
        return true if free
        '''
        logger.debug("START check_free_collision")


        x = state[0]
        y = state[1]
        length = self.length
        threshold = self.threshold

        resolution = self.map.resolution
        occupancy_grid = self.map.occupancy_grid

        min_x = int((x - length/2 - threshold) / resolution)
        # min_x = min_x if min_x >= 0 else 0

        max_x = int((x + length/2 + threshold) / resolution) + 1
        # max_x = max_x if max_x < self.map.width else self.map.width - 1

        min_y = int((y - length/2 - threshold) / resolution)
        # min_y = min_y if min_y >= 0 else 0

        max_y = int((y + length/2 + threshold) / resolution) +1
        # max_y = max_y if max_y < self.map.height else self.map.height - 1
        
        if not( self.check_in_map((min_x ,min_y)) and self.check_in_map((min_x, max_y)) and self.check_in_map((max_x, min_y)) and self.check_in_map((max_x, max_y)) ):
            logger.debug("END check_free_collision")
            return False

        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                if occupancy_grid[i, j] == 1:

                    logger.debug("END check_free_collision")
                    return False

        logger.debug("END check_free_collision")
        return True
    
    def update_costs(self, new_node, near_neighbours):
        """
        Update the cost of all the nearest neighbours of the new node, 
        if the cost of the trajectory that passes through it is lower than the corrunt cost of the neighbour.
        """
        # #
        # 
        # - Compute trajectory and cost from new_node to near_node
        # - Check if each configuration in the trajectory is collision free
        # - if the cost of new_node + the cost from new_node to near_node (computed in the first step) is lower than the cost of near_node t
        # - then: 
        #       - set the parent of near_node as new_node
        #       - set the cost of near_node as the cost of new_node + the cost from new_node to near_node (computed in the first step)
        #       - may need to set other parameters for near node
        #
        # - nella reference near node non viene modificato in place, invece viene creata una copia che viene agigunta al Tree senza rimuovere la versione precedente (?)
        # 
        # #
        logger.debug("START update_costs")
        
        parent_node = new_node.parent
        cost_new_node = new_node.cost
        
        for near_node in near_neighbours:
            
            if (near_node.x,near_node.y) == (parent_node.x, parent_node.y) or (near_node.x,near_node.y) == (new_node.x,new_node.y):
                continue 
            
            cost_nn, trajectory_nn, commands_nn, _ = self.optimal_trajectory(new_node.state, near_node.state, TAU_STEP, FREQUENCY)      #####
            if cost_nn <= 0:
                logger.warning(f'[update_costs], negative costs: {cost_nn}')
                return False
            
            skip = False
            
            for i in range(len(trajectory_nn)):
                if not (self.check_admissible_ranges(trajectory_nn[i], commands_nn[i]) and self.check_free_collision(trajectory_nn[i])) :
                    skip = True
                    break

                if trajectory_nn[i][4] < MIN_V1:
                    logger.warning(f"Velocity near to zero")
                    skip = True
                    break
            
            # if skip:
            #     continue
            
            
            if cost_new_node + cost_nn < near_node.cost:
                near_node.parent = new_node
                near_node.cost = cost_new_node + cost_nn
                near_node.trajectory = trajectory_nn
                near_node.inputs = commands_nn
                ### maybe save the trajectory in the node ?
                
                
                ##### self.Tree.append(near_node)
                ##### IN TEORIA near_node è la reference all'ogetto già presente nella lista, quindi non dovrebbe esserci bisogno di riaggiungerlo
        logger.debug("END update_costs")
        
        return True
    
    def choose_parent(self, new_node, near_neighbours):
        
        """
        Select which of the nearest neighobours is the parent, based on the cost of the trajectory between each of them and the new node.
        If no trajectory is feasible between any of the nearest neighbours and the new node return None as parent_node.
        """
        logger.debug("START choose_parent")
        
        parent_node = None
        cost_from_parent = 1e10 # high initial value 
        trajectory_from_parent = None
        inputs_from_parent = None
        
        for near_node in near_neighbours:
            
            if near_node.x == new_node.x and near_node.y == new_node.y:
                continue
        
            cost_nn, trajectory_nn, commands_nn, _ = self.optimal_trajectory(near_node.state, new_node.state, TAU_STEP, FREQUENCY)      #####
            
            skip = False
            
            #commands_nn = np.array(commands_nn)
            
            for i in range(len(trajectory_nn)):
                if not (self.check_admissible_ranges(trajectory_nn[i], commands_nn[i]) and self.check_free_collision(trajectory_nn[i])) :
                    # logger.warning(f"[choose_parent] check not valid, node: {trajectory_nn[i]}, inputs: {commands_nn[i]}")
                    skip = True
                    break

                if trajectory_nn[i][4] < MIN_V1:
                    # logger.warning(f"[choose_parent],Velocity near to zero")
                    skip = True
                    break

            # if skip:
            #     continue
            
            
            cost_from_near = cost_nn + near_node.cost
            
            if cost_from_near < cost_from_parent:
                
                cost_from_parent = cost_from_near
                parent_node = near_node
                trajectory_from_parent = trajectory_nn
                inputs_from_parent = commands_nn
                
        logger.debug("END choose_parent")
        return parent_node, cost_from_parent, trajectory_from_parent, inputs_from_parent


def initMap():
    # Example usage with visualization:
    heigth = 100
    width = 100
    inflate = True  # Set to True to inflate obstacles
    inflation_radius = 4
    obstacle_probability = (
        0.000 # Probability of each cell being an obstacle (before inflating)
    )

    grid_map = OccupancyGridMap(
        width, heigth, None, inflate, inflation_radius, obstacle_probability, resolution=RESOLUTION, seed = 1,
    )

    return grid_map
#To test

def main_test_general():
    grid_map = initMap()
    # grid_map.plot_map()


    initial_state = np.array([1, 1, 0, 0, 0, 0])
    goal_state = np.array([10, 10, 0, 0, 0, 0])

    rrt = RRT_STAR(
        initial_state, goal_state, grid_map, 1 #1 is step_meter
    ) 

    random_node1 = rrt.generate_random_node()
    random_node2 = rrt.generate_random_node()
    random_node3 = rrt.generate_random_node()
    random_node4 = rrt.generate_random_node()
    random_node5 = rrt.generate_random_node()
    random_node6 = rrt.generate_random_node()
    random_node7 = rrt.generate_random_node()
    random_node8 = rrt.generate_random_node()
    random_node9 = rrt.generate_random_node()
    random_node10 = rrt.generate_random_node()
    random_node11 = rrt.generate_random_node()
    random_node12 = rrt.generate_random_node()
    random_node13 = rrt.generate_random_node()
    random_node14 = rrt.generate_random_node()
    random_node15 = rrt.generate_random_node()
    
    rrt.random_tree.extend([random_node1,
                            random_node2,
                            random_node3,
                            # random_node4,
                            # random_node5,
                            # random_node7,
                            # random_node8,
                            # random_node9,
                            # random_node10,
                            # random_node11,
                            # random_node12,
                            # random_node13,
                            # random_node14,
                            random_node15])
    
    rrt.root_node.cost = 100
    rrt.root_node.parent = None
    rrt.goal_node.cost = 0
    rrt.goal_node.parent = random_node15
    
    
    random_node1.cost = 50
    random_node1.parent = rrt.root_node
    random_node2.cost = 25
    random_node2.parent = random_node1
    random_node3.cost = 15
    random_node3.parent = random_node2
    random_node15.cost = 5
    random_node15.parent = random_node3

    print(f'current_tree: {[str(node) for node in rrt.random_tree]}')

    random_node = rrt.generate_random_node()
    print(f'random_node: {random_node}')

    nearest_node, current_node = rrt.get_NN(random_node)
    print(f'nearest_node: {nearest_node}, current_node: {current_node}')

    nearest_nodes = rrt.get_multiple_NN(random_node)
    print(f'nearest_nodes: {[str(node) for node in nearest_nodes]}, nearest_len: {len(nearest_nodes)}')
    
    rrt.R = np.eye(3)
    rrt.min_v1 =   -100000
    rrt.max_v1 =    100000
    rrt.min_om =   -100000
    rrt.max_om =    100000
    rrt.min_vf1 =  -100000
    rrt.max_vf1 =   100000
    rrt.min_av1 =  -100000
    rrt.max_av1 =   100000
    rrt.min_aom =  -100000
    rrt.max_aom =   100000
    rrt.length = 1
    rrt.threshold = 0.1
        
    
    parent, cost_from_parent = rrt.choose_parent(current_node, nearest_nodes)
    current_node.parent = parent
    current_node.cost = cost_from_parent + parent.cost
    
    rrt.update_costs(current_node, nearest_nodes)
    
    for el in rrt.random_tree:
        print(el.cost)

def main_planning():
    grid_map = initMap()

    initial_state = np.array([40, 40, 1, 1, 1, 1])
    goal_state = np.array([50, 50, 1, 1, 1, 1])

    rrt = RRT_STAR(
        initial_state, goal_state, grid_map, 5 #1 is step_meter
    ) 

    rrt.root_node.cost = 0
    rrt.root_node.parent = None
    
    rrt.R = np.eye(3)
    rrt.min_v1 =    0.005
    rrt.max_v1 =    100000
    rrt.min_om =   -100000
    rrt.max_om =    100000
    rrt.min_vf1 =  -100000
    rrt.max_vf1 =   100000
    rrt.min_av1 =  -100000
    rrt.max_av1 =   100000
    rrt.min_aom =  -100000
    rrt.max_aom =   100000
    rrt.length = 1
    rrt.threshold = 0.1
    
    optimal_path = rrt.rrt_plan()

    print(f'optimal_path: {[str(node) for node in optimal_path]}')
    print("optimal path view: ",[o.state.tolist() for o in optimal_path])
    optimal_trajectory = np.array([conf.tolist() for n in optimal_path for conf in n.trajectory])
    optimal_inputs = [np.array(n.inputs) for n in optimal_path]
    optimal_inputs = np.vstack(optimal_inputs)
    print("full trajectory optimal path: ",optimal_trajectory)
    print()
    print("full inputs optimal path: ",optimal_inputs)

    plot_in_time(optimal_trajectory[:,0], TAU_STEP, label='x', filename='x')
    plot_in_time(optimal_trajectory[:,1], TAU_STEP, label='y', filename='y')
    plot_in_time(optimal_trajectory[:,2], TAU_STEP, label='θ', filename='theta')
    plot_in_time(optimal_trajectory[:,3], TAU_STEP, label='ϕ_1', filename='phi_1')
    plot_in_time(optimal_trajectory[:,4], TAU_STEP, label='v_1', filename='v_1')
    plot_in_time(optimal_trajectory[:,5], TAU_STEP, label='ω', filename='omega')
  
    plot_in_time(optimal_inputs[:,0], TAU_STEP, label='v_ϕ_1', filename='v_phi_1')
    plot_in_time(optimal_inputs[:,1], TAU_STEP, label='a_v_1', filename='a_v_1')
    plot_in_time(optimal_inputs[:,2], TAU_STEP, label='a_ω', filename='v_omega')
    # print("reversed full trajectory optimal path: ",[conf.tolist() for n in optimal_path for conf in reversed(n.trajectory)])

    return rrt

def main_planning2():
    grid_map = initMap()

    initial_state = np.array([40, 40, 0, 0, 1, 1])
    goal_state = np.array([50, 50, 1, 1, 1, 1])

    rrt = RRT_STAR(
        initial_state, goal_state, grid_map, 5 #1 is step_meter
    ) 

    rrt.root_node.cost = 0
    rrt.root_node.parent = None
    temp=Node(state=np.array([45,45,0,0,1,1]))
    temp.cost=50
    rrt.random_tree.append(temp)
    temp.parent=rrt.root_node
    for nod in rrt.random_tree:
        print(nod.state)
        print(nod.cost)
        if nod.parent is not None:
            print("genitore",nod.parent.state)
        else:
            print("None")
    rrt.R = np.eye(3)
    rrt.min_v1 =    0.005
    rrt.max_v1 =    100000
    rrt.min_om =   -100000
    rrt.max_om =    100000
    rrt.min_vf1 =  -100000
    rrt.max_vf1 =   100000
    rrt.min_av1 =  -100000
    rrt.max_av1 =   100000
    rrt.min_aom =  -100000
    rrt.max_aom =   100000
    rrt.length = 1
    rrt.threshold = 0.1
    
   


    n_rand = Node(state=np.array([43,43,0,0,1,1]))
    logger.debug(f'RANDOM: {n_rand.state}')

    n_nearest, n_new = rrt.get_NN(n_rand)
    logger.debug(f'nearest: {n_nearest.state}, new: {n_new.state}')            

    cost, path, inputs, tau_star = rrt.optimal_trajectory(n_nearest.state, n_new.state, 0.1, 10) # From nearest to new
    

    for i, state_i in enumerate(path):
        # print(f'state_i: {state_i}, inputs_i: {inputs[i]}')
        check_ranges = rrt.check_admissible_ranges(state_i, inputs[i])
        check_collision = rrt.check_free_collision(state_i)

        if not check_ranges:
            logger.warning(f"Range found, state_i: {state_i}, inputs_i: {inputs[i]}, from: {n_nearest}, to: {n_new}")
            collision_or_ranges = True
            break

        if not check_collision:
            logger.warning("Collision found, state_i: {state_i}, inputs_i: {inputs[i]}")
            collision_or_ranges = True
            break

        if state_i[4] < MIN_V1:
            logger.warning(f"Velocity near to zero, state_i: {state_i}, inputs_i: {inputs[i]}")
            collision_or_ranges = True
            break
    

    nearest_nodes = rrt.get_multiple_NN(n_new)


    parent, cost_from_parent, traj_from_parent, inputs_from_parent = rrt.choose_parent(n_new, nearest_nodes)
    
    print("Genitore scelto per 43", parent.state)
    if cost_from_parent <= 0: 
        logger.info("COSTI NEGATIVI choose_parent")

    logger.debug(f'parent: {parent.state}, n_new: {n_new.state}')

    if parent == None:
        logger.info(f"Parent None, parent: {parent}, n_new: {n_new}")

    n_new.cost = cost_from_parent + parent.cost
    n_new.parent = parent
    n_new.name = "Node " + str(len(rrt.random_tree))
    n_new.trajectory = traj_from_parent
    n_new.inputs = inputs_from_parent

    rrt.random_tree.append(n_new)
    
    if not rrt.update_costs(n_new, nearest_nodes):
        logger.info("COSTI NEGATIVI update_costs")
    
    logger.info(f'node added: {n_new}')
    # if n_new.x - self.distance_tol <= self.goal_node.x <= n_new.x + self.distance_tol and n_new.y - self.distance_tol <= self.goal_node.y <= n_new.y + self.distance_tol:
    #     #reached a node near the goal node
    #     break
    
    for nod in rrt.random_tree:
        print(nod.state)
        print(nod.cost)
        if nod.parent is not None:
            print("genitore",nod.parent.state)
        else:
            print("None")
        
    return rrt

if __name__ == "__main__":

    rrt = main_planning2()
    print(f'tree: {[str(node) for node in rrt.random_tree]}')
    
    # robot_trajectory = [[1, 1, 1, 1, 1 ,1],  [3.9577794,  4.72793913, 2.39589418, 2.08235857, 0.85152718, 0.62543511],  [7.67569874, 9.37871788, 2.1794398 , 2.0717892,  2.45001947, 0.94138447],  [13.47713923, 15.0223176 ,  0.01976392,  2.72604305,  2.57613641,  2.46105665],  [18.25219143, 20.92763219 , 1.17325213,  2.85730783 , 1.35129603 , 0.71615759],  [19.50437986, 24.71032277,  0.79837991 , 1.14581284 , 1.17405516 , 1.2986318 ],  [21.89997192 ,30.17015509 , 1.57504492 , 1.38215166 , 2.76778006 , 1.30327787],  [19.1268989,  36.77366338,  2.33808823,  0.5914706  , 1.74115345,  1.02255005],  [23.06313442, 44.45574102,  1.41336657 , 2.88030073 , 2.58299568 , 2.9442662 ], [22.10655356, 43.06417286,  2.2067761 ,  1.7898561 ,  0.63730004 , 1.77538698],  [27.37852421, 46.15611744 , 0.89404962,  2.20649931,  2.24853743,  1.75598689],  [35.58640175, 45.76953008 , 1.61227853 , 1.89267094,  2.4951089,   2.25287184],  [39.73653238, 46.38466935,  2.63642555, 1.40475791 , 1.37133822 , 0.51117331],  [45.94028748, 43.99102342,  2.46537105,  2.4750513 ,  0.61071359 , 1.27734601]]
    # grid_map = initMap()

    # initial_state = np.array([1, 1, 1, 1, 1, 1])
    # goal_state = np.array([75, 75, 1, 1, 1, 1])

    # rrt = RRT_STAR(
    #     initial_state, goal_state, grid_map, 1 #1 is step_meter
    # ) 

    # rrt.root_node.cost = 0
    # rrt.root_node.parent = None
    
    # rrt.length = 1
    # rrt.threshold = 0.1
    
    # rrt.R = np.eye(3)
    
    # tr = []
    
    # for i in range(len(robot_trajectory)-1):
    #     cost_nn, trajectory_nn, commands_nn, _ = rrt.optimal_trajectory(robot_trajectory[i], robot_trajectory[i+1],1,10) 
    #     for el in trajectory_nn:
    #         print("is collision free: ",rrt.check_free_collision(el))
    #         tr.append(el[:3].tolist())
            
    # print(tr)
        