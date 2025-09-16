import numpy as np
import sympy
import math
import sys
import os
import time
import pickle
import matplotlib.pyplot as plt
#DEBUG, INFO, WARNING, ERROR, CRITICAL
import logging
logger = logging.getLogger('logger1')
logger.setLevel(logging.INFO)  # Imposta il livello di log a ERROR
handler1 = logging.StreamHandler()  # Invia i log a stdout
formatter1 = logging.Formatter('%(levelname)s: %(message)s')
handler1.setFormatter(formatter1)
logger.addHandler(handler1)
handler1_2 = logging.FileHandler('logger1.log')  # Salva i log in un file

logger.addHandler(handler1_2)
# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

logger2 = logging.getLogger('logger2')
logger2.setLevel(logging.DEBUG)  # Imposta il livello di log a ERROR
handler2 = logging.FileHandler('phi_s.log')  # Salva i log in un file
formatter2 = logging.Formatter('%(levelname)s: %(message)s')
handler2.setFormatter(formatter2)
logger2.addHandler(handler2)

# Get the directory containing the current file
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Current folder:", project_folder)

sys.path.append(project_folder)
from decimal import Decimal

# sys.path.append(r'C:\Users\Luca\AMR23-FP7-MPSWMR\AMR23-FP7-MPSWMR')#ognuno mette il suo
# sys.path.append(r'C:\Users\ciarp\OneDrive\Desktop\progetto-amr\AMR23-FP7-MPSWMR')#ognuno mette il suo

from Model.model_creator import * 
from Utils.utils import * 
from Utils.constants import *
from World.occupancyGridMap import OccupancyGridMap


d = 1 


class Node():
    def __init__(self, state):
        self.state = state
        self.x = state[0]
        self.y = state[1]
        self.position = np.array([self.x, self.y])
        
        self.phis = []
        self.vs = []

        self.parent = None
        self.children = []
        self.cost = None
        
        self.trajectory = [self.state]
        self.inputs = np.array([0,0,0])
        self.time=0
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

        self.generated_nodes = []

        self.random_tree = [self.root_node]

        self.map = map

        self.step_m = step_meter # distance performed by the robot in 1 time_step
        self.distance_tol = 0.001

        expanded_q=np.array([0,0,0,0,1,0])
        my_u=np.array([0,0,0])
        self.model = Dynamic_Model(expanded_q,my_u, P, d)
        self.start_v_joint_1=self.model.get_v_joint(root_state,P1,root_state[3])
        self.start_v_joint_2=self.model.get_v_joint(root_state,P2,get_phi_n(root_state,P1,P2))
        self.start_v_joint_3=self.model.get_v_joint(root_state,P3,get_phi_n(root_state,P1,P3))
        self.start_v_joint_4=self.model.get_v_joint(root_state,P4,get_phi_n(root_state,P1,P4))

        logger2.warning(f'starting: v1: {self.start_v_joint_1}, v2: {self.start_v_joint_2}, v3: {self.start_v_joint_3}, v4: {self.start_v_joint_4}')
        return
    
    def plot_random_tree_nodes(self):
        x = [node.x for node in self.random_tree]
        y = [node.y for node in self.random_tree]

        plt.scatter(x, y, color='orange')

        # Aggiunge titolo e etichette agli assi
        plt.title('Random Tree')
        plt.xlabel('x')
        plt.ylabel('y')

        # plt.gca().invert_yaxis()
        
        plt.savefig('random_tree.png')
        # Mostra il grafico
        # plt.show()
        return

    def rrt_plan(self):

        history = []
        total_cost = 0

        for iter in range(N_ITERS):

            logger.info(f"ITERATION N°: {iter}")

            if len(self.generated_nodes) > 0:
                n_rand = self.generated_nodes.pop(0)
            
            else:
                n_rand = self.generate_random_node()
            logger.debug(f'RANDOM: {n_rand.state}')
        
            ####################################n_nearest, n_new = self.get_NN(n_rand)
            ####################################logger.debug(f'nearest: {n_nearest}, new: {n_new.state}')            
        ####################################
            ####################################cost, path, inputs, tau_star = self.optimal_trajectory(n_nearest.state, n_new.state, TAU_STEP, FREQUENCY) # From nearest to new
            ####################################
            ####################################if cost <= 0 : 
            ####################################    logger.info("COSTI NEGATIVI")
            ####################################    continue
            ####################################
            ####################################collision_or_ranges = False
####################################
            ####################################prev_phis = [phi for phi in n_nearest.phis]
####################################
            ####################################path = path.tolist()
####################################
            ####################################for i, state_i in enumerate(path):
####################################
            ####################################    curr_phis = compute_all_phis(state_i, prev_phis)
            ####################################    curr_vs = compute_all_vs(state_i, curr_phis, self.model)
####################################
            ####################################    # print(f'state_i: {state_i}, inputs_i: {inputs[i]}')
            ####################################    check_ranges = self.check_admissible_ranges(state_i, inputs[i], curr_vs)
            ####################################    check_collision = self.check_free_collision(state_i)
####################################
            ####################################    if not check_ranges:
            ####################################        logger.warning(f"Range found, state_i: {state_i}, inputs_i: {inputs[i]}, from: {n_nearest}, to: {n_new}")
            ####################################        collision_or_ranges = True
            ####################################        break
####################################
            ####################################    if not check_collision:
            ####################################        logger.warning(f"Collision found, state_i: {state_i}, inputs_i: {inputs[i]}")
            ####################################        collision_or_ranges = True
            ####################################        break
####################################
            ####################################    if state_i[4] < MIN_V1:
            ####################################        logger.warning(f"Velocity near to zero, state_i: {state_i}, inputs_i: {inputs[i]}")
            ####################################        collision_or_ranges = True
            ####################################        break
            ####################################    
            ####################################    #Non aggiungere 2 volte phis e v_S nello stato
            ####################################    # if i != 0:
            ####################################    #     path[i].append(curr_phis)
            ####################################    #     path[i].append(curr_vs)
            ####################################    prev_phis = curr_phis
####################################
            ####################################    
            ####################################if collision_or_ranges:
            ####################################    continue
            ####################################
            ####################################history.append({
            ####################################    'cost': cost,
            ####################################    'tau': tau_star,
            ####################################    'path': path,
            ####################################    'inputs': inputs,
            ####################################    'iter': iter,
            ####################################    'index_node': len(self.random_tree)
            ####################################})

            n_new = n_rand ########################################################################################################################################à
            ####################################nearest_nodes = self.get_multiple_NN(n_new)
            nearest_nodes = self.random_tree 

            parent, cost_from_parent, traj_from_parent, inputs_from_parent,time_from_parent, phis_from_parent, vs_from_parent = self.choose_parent(n_new, nearest_nodes)
            
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
            n_new.time = time_from_parent
            n_new.phis_from_parent = phis_from_parent
            n_new.vs_from_parent = vs_from_parent
            n_new.phis = phis_from_parent[-1]

            self.random_tree.append(n_new)
            
            nearest_nodes = nearest_nodes + [self.goal_node]
            
            if not self.update_costs(n_new, nearest_nodes):
                logger.info("COSTI NEGATIVI update_costs")
                continue
            
            logger.info(f'node added: {n_new}')
            # if n_new.x - self.distance_tol <= self.goal_node.x <= n_new.x + self.distance_tol and n_new.y - self.distance_tol <= self.goal_node.y <= n_new.y + self.distance_tol:
            #     #reached a node near the goal node
            #     break
            
        # if self.goal_node.parent != None:
        #     complete_path = self.get_complete_path(self.goal_node)
            
        # else:
        nearest_goal_node = self.get_nearest_goal()
        goal_node = self.check_goal_path(nearest_goal_node)
        complete_path = self.get_complete_path(goal_node)

        print("End planning")
            
        return complete_path
    
    def get_nearest_goal(self):

        dist_list = np.array([self.distance_cost(node, self.goal_node)
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
        # cost, path, inputs, tau_star = self.optimal_trajectory(nearest_goal.state, self.goal_node.state, TAU_STEP, FREQUENCY) # From nearest to new
        # if cost <=0:
        #     return nearest_goal
        
        # collision_or_ranges = False

        # for i, state_i in enumerate(path):
        #     logger.debug(f'state_i: {state_i}, inputs_i: {inputs[i]}')
        #     check_ranges = self.check_admissible_ranges(state_i, inputs[i])
        #     check_collision = self.check_free_collision(state_i)

        #     if not check_ranges:
        #         logger.warn("Range found")
        #         collision_or_ranges = True
        #         break

        #     if not check_collision:
        #         logger.warn("Collision found")
        #         collision_or_ranges = True
        #         break

            
        # if collision_or_ranges:
        #     return nearest_goal
        # # return nearest_goal
        

        nearest_nodes = self.get_multiple_NN(self.goal_node)

        parent, cost_from_parent, traj_from_parent, inputs_from_parent,time_from_parent, phis_from_parent, vs_from_parent = self.choose_parent(self.goal_node, nearest_nodes)

        logger.debug(f'parent: {parent}, self.goal_node: {self.goal_node}')

        if parent == None:
            logger.error("Parent None")
            return nearest_goal
        
        self.goal_node.cost = cost_from_parent + parent.cost
        self.goal_node.parent = parent
        self.goal_node.trajectory = traj_from_parent
        self.goal_node.inputs = inputs_from_parent
        self.goal_node.phis_from_parent = phis_from_parent
        self.goal_node.vs_from_parent = vs_from_parent
        self.goal_node.time=time_from_parent
        self.goal_node.name = "Node Goal"

        self.random_tree.append(self.goal_node)
        
        # self.update_costs(self.goal_node, nearest_nodes)

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

        tau = Decimal('0.0')
        tau_step_s=str(tau_step)
        camp=1/frequency
        camp_s=str(camp)
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
            
            tau+= Decimal(tau_step_s)
            iter += 1
            q_bar[iter] = self.RK4(self.q_bar_d, q_bar[iter-1], Decimal(tau_step_s),camp_s)
            G[iter] = self.RK4(self.G_d, G[iter-1], Decimal(tau_step_s),camp_s)
            c[iter] = self.cost_fn(G[iter], q1 ,q_bar[iter], tau)

            if c[iter] < c[iter_star]:
                if c[iter]>0:
                    iter_star = iter
                    tau_star = tau
        d_star = np.linalg.pinv(G[iter_star]) @ (q1 - q_bar[iter_star])
        state_f = np.concatenate([q1, d_star])

        temp_path, temp_y = self.back_RK4(self.ODE_state, state_f, tau_star,camp_s) #valutare se fare anche qua come sopra

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

    def RK4(self,f,y0,tf,camp_s): #tf mi dice quali sono i secondi finali
        t = np.arange(Decimal('0.0'), tf, Decimal(camp_s))
        if tf not in t:
            t = np.concatenate((t, np.array([tf])))
        n = len(t)
        y = np.zeros((n, *y0.shape))
        y[0] = y0
        for i in range(n - 1):
            h = t[i+1] - t[i] #h sarebbe l'intervallo do tempo. se h=2, vuol dire che passano due secondi
            k1=f(y[i])
            k2=f(y[i] +k1*float(h)/2.)
            k3=f(y[i] +k2*float(h)/2.)
            k4=f(y[i] +k3*float(h))
            y[i+1] = y[i] + (float(h) / 6.) * (k1 + 2*k2 + 2*k3 + k4)

        return y[-1]

    def back_RK4(self,f,final_state,tf,camp_s):
        t = np.arange(Decimal('0.0'), tf, Decimal(camp_s))
        if tf not in t:
            t = np.concatenate((t, np.array([tf])))
        n=len(t)
        state=np.zeros((n,*final_state.shape))
        state[0]=final_state
        for i in range(n-1):
            h = t[i+1] - t[i]
            k1 = f(state[i])
            k2 = f(state[i] - k1 * float(h) / 2.)
            k3 = f(state[i] - k2 * float(h) / 2.)
            k4 = f(state[i] - k3 * float(h))
            state[i+1] = state[i] - (float(h) / 6.) * (k1 + 2*k2 + 2*k3 + k4)
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
        
        c = float(t) + (q1 - q_bar).T @ np.linalg.pinv(G) @ (q1 - q_bar)
        
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

        dist_list = np.array([self.distance_cost(current_node, tree_node)
                             for tree_node in self.random_tree])
        
        idx_min = np.argmin(dist_list)
        # dist_min = np.min(dist_list)

        nearest = self.random_tree[idx_min]
        
        #dist = dist_list[idx_min]  
        #dist = min(dist, self.step_m) ############
        # dist = self.euclidean_distance(current_node, self.random_tree[idx_min])

        #theta = math.atan2(current_node.y - nearest.y, current_node.x - nearest.x)          ##########à
        # print(f'THETA: {theta}, ')
        #TODO: fix state based on dict or array, now is assumed to be an array: [x, y, θ, ϕ_1, v_1, ω]
        # new_state = np.array([(nearest.x + dist*math.cos(theta)), ##########################
        #                       (nearest.y + dist*math.sin(theta)),##########################
        #                       current_node.state[2], ##########################
        #                       current_node.state[3], ##########################
        #                       current_node.state[4], ##########################
        #                       current_node.state[5]])##########################
        
        new_state = np.array([current_node.state[0], ####################################
                              current_node.state[1],####################################
                              current_node.state[2], ####################################
                              current_node.state[3], ####################################
                              current_node.state[4], ####################################
                              current_node.state[5]])####################################
        
        new_node = Node(new_state)

        logger.debug("End get_NN")

        return nearest, new_node
    
    def get_multiple_NN(self, current_node):
        ''' 
            Compute the list of the nearest neightbors
        '''
        logger.debug("Start get_multiple_NN")

        distances = np.array([self.distance_cost(current_node, tree_node)
                             for tree_node in self.random_tree])
        
        # number_neighbours = 100 if len(distances) > 100 else len(distances)
        number_neighbours = len(distances)
        distances_argsorted = distances.argsort()[:number_neighbours]
        
        nearest_nodes = [self.random_tree[i] for i in distances_argsorted]
        
        logger.debug("End get_multiple_NN")
        return nearest_nodes
    
    def generate_random_node(self):
        ''' Generate a random node '''
        
        x_new = np.random.uniform(low = OFFSET_ROBOT_MAP, high = (self.map.width-OFFSET_ROBOT_MAP) * self.map.resolution)
        y_new = np.random.uniform(low = OFFSET_ROBOT_MAP, high = (self.map.height-OFFSET_ROBOT_MAP) * self.map.resolution)
        # x_new = np.random.uniform(low = 15, high = (70) * self.map.resolution)
        # y_new = np.random.uniform(low = 15, high = (70) * self.map.resolution)
        theta_new = np.random.uniform(low = THETA_RANDOM[0], high = THETA_RANDOM[1])
        phi_new = np.random.uniform(low = PHI_RANDOM[0], high = PHI_RANDOM[1])
        v_new = np.random.uniform(low = V_1_RANDOM[0], high = V_1_RANDOM[1])
        omega_new = np.random.uniform(low = OMEGA_RANDOM[0], high = OMEGA_RANDOM[1])
        
            
        state = np.array([x_new, y_new, theta_new, phi_new, v_new, omega_new])

        
        return Node(state)



    def euclidean_distance(self, n1, n2):
        dist = np.linalg.norm(n1.position - n2.position)
        return dist
    
    def distance_cost(self, n1, n2):
        dist = np.linalg.norm(n1.position - n2.position)
        theta_distance = np.linalg.norm(wrap_angle(n1.state[2] - n2.state[2]))
        phi_1_distance = np.linalg.norm(wrap_angle(n1.state[3] - n2.state[3]))
        v_1_distance = np.linalg.norm(n1.state[4] - n2.state[4])
        om_distance = np.linalg.norm(n1.state[5] - n2.state[5])

        result = dist * 0.4 + theta_distance * 0.3 + phi_1_distance  * 0.1 + v_1_distance * 0.1 + om_distance * 0.1
        return result


    # assumo che state sia = [x, y, θ, ϕ_1, v_1, ω]
    def check_in_map(self, state):
        '''
        Return True if the position (x,y) is within the map limits
        '''

        # resolution = self.map.resolution
        
        # print(state)

        x = state[0] 
        y = state[1] 
        
        # x = state[0] / resolution
        # y = state[1] / resolution
        
        # print(x,y)

        if 0 <= x < self.map.width and 0 <= y <= self.map.height:
            return True

        return False

    # assumo che state sia = [x, y, θ, ϕ_1, v_1, ω]
    # assumo che inputs sia = [v_ϕ_1 a_v_1 a_ω]
    def check_admissible_ranges(self, state, inputs, curr_vs):
        '''
        Check if the velocity values are whithin their admissible ranges
        '''
        logger.debug("START check_admissible_ranges")
        
        resolution = self.map.resolution
        
        if not self.check_in_map((state[0]/ resolution,state[1]/resolution)) or not (self.min_v1 < state[4] < self.max_v1 and self.min_om <= state[5] <= self.max_om):
            
            logger.debug("END check_admissible_ranges")
            return False
        
        if not ( self.min_vf1 <= inputs[0] <= self.max_vf1 and
            self.min_av1 <= inputs[1] <= self.max_av1 and 
            self.min_aom <= inputs[2] <= self.max_aom) :

                logger.debug("END check_admissible_ranges")
                return False
        
        vs_1 = curr_vs[0]
        vs_2 = curr_vs[1]
        vs_3 = curr_vs[2]
        vs_4 = curr_vs[3]
        
        logger2.info(f'vj_1: {vs_1}, vj_2: {vs_2}, vj_2: {vs_3}, vj_4: {vs_4}')
        if np.sign(vs_1)!=np.sign(self.start_v_joint_1):
            logger2.warning(f"joint 1 has changed sign")
            return False
        if np.sign(vs_2)!=np.sign(self.start_v_joint_2):
            logger2.warning(f"joint 2 has changed sign")
            return False
        if np.sign(vs_3)!=np.sign(self.start_v_joint_3):
            logger2.warning(f"joint 3 has changed sign")
            return False
        if np.sign(vs_4)!=np.sign(self.start_v_joint_4):
            logger2.warning(f"joint 4 has changed sign")
            return False
        
        if vs_1<self.min_vel_joint:
            logger2.warning(f"joint 1 has zero velocity")#################################################################################################
            return False#################################################################################################
        if vs_2<self.min_vel_joint:#################################################################################################
            logger2.warning(f"joint 2 has zero velocity")#################################################################################################
            return False#################################################################################################
        if vs_3<self.min_vel_joint:#################################################################################################
            logger2.warning(f"joint 3 has zero velocity")#################################################################################################
            return False#################################################################################################
        if vs_4<self.min_vel_joint:#################################################################################################
            logger2.warning(f"joint 4 has zero velocity")#################################################################################################
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
        width = self.width
        height = self.height
        length = self.length
        threshold = self.threshold

        resolution = self.map.resolution
        occupancy_grid = self.map.occupancy_grid

        min_x = math.floor((x - length/2 - threshold) / resolution)
        # min_x = min_x if min_x >= 0 else 0

        max_x = math.ceil((x + length/2 + threshold) / resolution) 
        # max_x = max_x if max_x < self.map.width else self.map.width - 1

        min_y = math.floor((y - length/2 - threshold) / resolution)
        # min_y = min_y if min_y >= 0 else 0

        max_y = math.ceil((y + length/2 + threshold) / resolution) 
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
      
        logger.debug("START update_costs")
        
        parent_node = new_node.parent
        cost_new_node = new_node.cost
        
        for near_node in near_neighbours:
            
            if (near_node.x,near_node.y) == (parent_node.x, parent_node.y) or (near_node.x,near_node.y) == (new_node.x,new_node.y):
                continue 
            
            cost_nn, trajectory_nn, commands_nn, time_nn = self.optimal_trajectory(new_node.state, near_node.state, TAU_STEP, FREQUENCY)      #####
            if cost_nn <= 0:
                logger.warning(f'[update_costs], negative costs: {cost_nn}')
                return False
            
            skip = False

            prev_phis = [phi for phi in new_node.phis]

            trajectory_nn = trajectory_nn.tolist()
            
            for i in range(len(trajectory_nn)):

                state_i  = trajectory_nn[i]

                curr_phis = compute_all_phis(state_i, prev_phis)
                curr_vs = compute_all_vs(state_i, curr_phis, self.model)

                if not (self.check_admissible_ranges(trajectory_nn[i], commands_nn[i], curr_vs) and self.check_free_collision(trajectory_nn[i])) :
                    skip = True
                    break

                if trajectory_nn[i][4] < MIN_V1:
                    logger.warning(f"Velocity near to zero")
                    skip = True
                    break

                #Non aggiungere 2 volte phis e v_S nello stato
                # if i != 0:
                #     trajectory_nn[i].append(curr_phis)
                #     trajectory_nn[i].append(curr_vs)
                prev_phis = curr_phis
            
            if skip:
                continue
            
            if cost_new_node + cost_nn < near_node.cost:
                near_node.parent = new_node
                near_node.cost = cost_new_node + cost_nn
                near_node.trajectory = trajectory_nn
                near_node.inputs = commands_nn
                near_node.time= time_nn
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
        time_from_parent=0
        trajectory_from_parent = None
        inputs_from_parent = None
        phis_from_parent = []
        vs_from_parent = []
        
        for near_node in near_neighbours:
            
            if near_node.x == new_node.x and near_node.y == new_node.y:
                continue
        
            cost_nn, trajectory_nn, commands_nn, time_nn = self.optimal_trajectory(near_node.state, new_node.state, TAU_STEP, FREQUENCY)      #####
            
            skip = False
            
            #commands_nn = np.array(commands_nn)

            prev_phis = [phi for phi in near_node.phis]

            trajectory_nn = trajectory_nn.tolist()
            
            for i in range(len(trajectory_nn)):

                state_i  = trajectory_nn[i]

                curr_phis = compute_all_phis(state_i, prev_phis)
                curr_vs = compute_all_vs(state_i, curr_phis, self.model)

                if not (self.check_admissible_ranges(trajectory_nn[i], commands_nn[i], curr_vs) and self.check_free_collision(trajectory_nn[i])) :
                    skip = True
                    break

                if trajectory_nn[i][4] < MIN_V1:
                    # logger.warning(f"[choose_parent],Velocity near to zero")
                    skip = True
                    break
                
                #Non aggiungere 2 volte phis e v_S nello stato
                # if i != 0:
                #     trajectory_nn[i].append(curr_phis)
                #     trajectory_nn[i].append(curr_vs)
                phis_from_parent.append(curr_phis)
                vs_from_parent.append(curr_vs)
                prev_phis = curr_phis
            
            if skip:
                continue
            
            cost_from_near = cost_nn + near_node.cost
            
            if cost_from_near < cost_from_parent:
                
                cost_from_parent = cost_from_near
                parent_node = near_node
                trajectory_from_parent = trajectory_nn
                inputs_from_parent = commands_nn
                time_from_parent= time_nn
                
        logger.debug("END choose_parent")
        return parent_node, cost_from_parent, trajectory_from_parent, inputs_from_parent,time_from_parent, phis_from_parent, vs_from_parent


def initMap():
    # Example usage with visualization:
    heigth = MAP_WIDTH
    width = MAP_HEIGHT
    inflate = INFLATE  # Set to True to inflate obstacles
    inflation_radius = INFLATION_RADIUS
    obstacle_probability = (
        OBSTACLE_PROBABILITY # Probability of each cell being an obstacle (before inflating)
    )

    grid_map = OccupancyGridMap(
        width, heigth, OBSTACLE_POSITION, inflate, inflation_radius, obstacle_probability, resolution=RESOLUTION, seed = SEED,
    )

    return grid_map
#To test


def main_planning():
    grid_map = initMap()

    # [x, y, θ, ϕ_1, v_1, ω]
    initial_state = np.array(STARTING_NODE) 
    goal_state = np.array(GOAL_NODE)

    rrt = RRT_STAR(
        initial_state, goal_state, grid_map, 100 #1 is step_meter
    ) 

    rrt.root_node.cost = 0
    rrt.root_node.parent = None

    initial_phis = []
    initial_phis.append(rrt.root_node.state[3])
    initial_phis.append(get_phi_n(rrt.root_node.state, P1, P2))
    initial_phis.append(get_phi_n(rrt.root_node.state, P1, P3))
    initial_phis.append(get_phi_n(rrt.root_node.state, P1, P4))

    initial_vs = []
    initial_vs.append(rrt.model.get_v_joint(rrt.root_node.state, P1, initial_phis[0]))
    initial_vs.append(rrt.model.get_v_joint(rrt.root_node.state, P2, initial_phis[1]))
    initial_vs.append(rrt.model.get_v_joint(rrt.root_node.state, P3, initial_phis[2]))
    initial_vs.append(rrt.model.get_v_joint(rrt.root_node.state, P4, initial_phis[3]))

    rrt.root_node.phis = initial_phis
    rrt.root_node.vs = initial_vs
    
    # rrt.goal_node.phis = initial_phis
    # rrt.goal_node.vs = initial_vs
    rrt.goal_node.cost = 1e7 #########################################################################################
    
    rrt.R = np.eye(3)
    rrt.R[2, 2] = 100
    rrt.min_v1 =    MIN_V1
    rrt.max_v1 =    MAX_V1
    rrt.min_om =    MIN_OM
    rrt.max_om =    MAX_OM
    rrt.min_vf1 =   MIN_VF1
    rrt.max_vf1 =   MAX_VF1
    rrt.min_av1 =   MIN_AV1
    rrt.max_av1 =   MAX_AV1
    rrt.min_aom =   MIN_AOM
    rrt.max_aom =   MAX_AOM
    # rrt.length = 1
    rrt.width = ROBOT_WIDTH
    rrt.height = ROBOT_HEIGHT
    rrt.length = max(ROBOT_HEIGHT, ROBOT_WIDTH)
    rrt.threshold = THRESHOLD
    rrt.min_vel_joint = MIN_VEL_JOINT
    
    # rrt.generated_nodes.append(Node(np.array([ 20, 40, np.pi/2, 0, 2, 0])))
    # rrt.generated_nodes.append(Node(np.array([ 20, 60, np.pi/3, 0, 2, -0.2])))
    start_time = time.time()
    optimal_path = rrt.rrt_plan()
    print("planning time: ", time.time() - start_time)

    # print(f'optimal_path: {[str(node) for node in optimal_path]}')
    total_time=np.sum(list(node.time for node in optimal_path))
    print("total cost: ",optimal_path[-1].cost)
    print("number_nodes: ",len(rrt.random_tree))
    # print("optimal path view: ",[o.state.tolist() for o in optimal_path])
    # optimal_trajectory = [conf for n in optimal_path for conf in n.trajectory]
    optimal_trajectory = [conf for i, n in enumerate(optimal_path) for conf in n.trajectory if i != 0]

    optimal_inputs = [np.array(n.inputs) for n in optimal_path]
    optimal_inputs = np.vstack(optimal_inputs)
    
    #optimal_inputs = [np.array(n.inputs) for n in optimal_path]
    optimal_inputs_ev=[]
    for n in optimal_path:
        if np.all(n.inputs == 0) and n.inputs.ndim==1:
            continue
        temp=np.array(n.inputs[:-1])
        optimal_inputs_ev.append(temp)
    #optimal_inputs_ev = [np.array(np.delete(n.inputs, -1, axis=0)) for n in optimal_path]
    #[print("adesso2",np.array(np.delete(n.inputs, -1, axis=0))) for n in optimal_path]
    optimal_inputs_ev = np.vstack(optimal_inputs_ev)
    # print()
    # print("full inputs optimal path: ",optimal_inputs)
    #rrt.model.evolution(rrt.root_node.state,optimal_inputs,total_time,FREQUENCY)
    camp=1/FREQUENCY
    camp_s=str(camp)
    traj_not_linearized=rrt.model.evolution(rrt.root_node.state,optimal_inputs_ev,total_time,camp_s)

    #PLOT ALL STATES FOR LINEAR MODEL
    plt.figure()

    all_x = [node[0] for node in optimal_trajectory]
    all_y = [node[1] for node in optimal_trajectory]
    all_theta = [node[2] for node in optimal_trajectory]
    all_phi = [node[3] for node in optimal_trajectory]
    all_v_1 = [node[4] for node in optimal_trajectory]
    all_omega = [node[5] for node in optimal_trajectory]


    plt.figure()
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    subplot_in_time(all_x, FREQUENCY, axs[0, 0], label='x')
    subplot_in_time(all_y, FREQUENCY, axs[1, 0], label='y')
    subplot_in_time(all_theta, FREQUENCY, axs[2, 0], label='θ')
    subplot_in_time(all_phi, FREQUENCY, axs[0,1], label='ϕ_1')
    subplot_in_time(all_v_1, FREQUENCY, axs[1,1], label='v_1')
    subplot_in_time(all_omega, FREQUENCY, axs[2,1], label='ω')
    fig.tight_layout()

    plt.savefig(os.path.join(project_folder ,f'./imgs/linear/states.png'))
    plt.close()

    #PLOT RANDOM TREE
    grid_map.create_base_plot_map()
    rrt.plot_random_tree_nodes()
    # grid_map.create_base_plot_map()
    plot_optimal_path(optimal_trajectory)
    plt.close()


    plt.figure()
    grid_map.create_base_plot_map()
    single_plot_path(rrt.random_tree, color='green')
    plt.savefig(os.path.join(project_folder ,f'./imgs/linear/all_traj.png'))
    plt.close()

    # #PLOT ALL POSSIBILE TRAJECTORIES
    # plt.figure()
    # fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    # plot_path(rrt.random_tree, axs[0])
    # plot_path(rrt.random_tree, axs[1], filename = 'all_traj_multi_color.png')
    # plot_path(rrt.random_tree, axs[2], filename = 'all_traj_single_color.png', color='green')

    # fig.tight_layout()

    # plt.savefig(os.path.join(project_folder ,f'./imgs/linear/all_traj.png'))
    # plt.close()


    #PLOT ALL INPUTS
    plt.figure()
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    subplot_in_time(optimal_inputs[1:,0], FREQUENCY, axs[0], label='v_ϕ_1')
    subplot_in_time(optimal_inputs[1:,1], FREQUENCY, axs[1], label='a_v_1')
    subplot_in_time(optimal_inputs[1:,2], FREQUENCY, axs[2], label='a_ω')
    fig.tight_layout()

    plt.savefig(os.path.join(project_folder ,f'./imgs/inputs.png'))
    plt.close()


    #COMPUTE ALL EXTRA PLOTS FOR LINEAR MODEL
    # print(f'traj_LINEAR: {np.array(optimal_trajectory)}')
    # print(f'len_traj_LINEAR: {np.array(optimal_trajectory).shape}')
    optimal_trajectory = compute_extra_sub_plot(optimal_trajectory, rrt, optimal_inputs, initial_state, subfolder='linear/')


    #PLOT STATES FOR NON LINEAR MODEL
    # print(f'traj_not_linearized: {traj_not_linearized}')
    all_x = [node[0] for node in traj_not_linearized]
    all_y = [node[1] for node in traj_not_linearized]
    all_theta = [node[2] for node in traj_not_linearized]
    all_phi = [node[3] for node in traj_not_linearized]
    all_v_1 = [node[4] for node in traj_not_linearized]
    all_omega = [node[5] for node in traj_not_linearized]


    plt.figure()
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    subplot_in_time(all_x, FREQUENCY, axs[0, 0], label='x')
    subplot_in_time(all_y, FREQUENCY, axs[1, 0], label='y')
    subplot_in_time(all_theta, FREQUENCY, axs[2, 0], label='θ')
    subplot_in_time(all_phi, FREQUENCY, axs[0,1], label='ϕ_1')
    subplot_in_time(all_v_1, FREQUENCY, axs[1,1], label='v_1')
    subplot_in_time(all_omega, FREQUENCY, axs[2,1], label='ω')
    fig.tight_layout()

    plt.savefig(os.path.join(project_folder ,f'./imgs/nonlinear/states.png'))
    plt.close()

    #PLOT TRAJECTORY FOR NON LINEAR MODEL
    plt.figure()
    plot_trajectory(traj_not_linearized)

    plt.savefig(os.path.join(project_folder ,f'./imgs/nonlinear/trajectory.png'))
    plt.close()


    #COMPUTE ALL EXTRA PLOTS FOR NON LINEAR MODEL
    compute_extra_sub_plot(traj_not_linearized, rrt, optimal_inputs, initial_state, subfolder='nonlinear/', linear = False)

    
    
    file_path = 'optimal_trajectory.pkl'
    file_path_nonlinear = 'optimal_trajectory_nonlinear.pkl'

    # Open the file in binary write mode
    with open(file_path, 'wb') as file:
        # Dump the list into the file using pickle
        pickle.dump(optimal_trajectory, file)
    
    with open(file_path_nonlinear, 'wb') as file:
        # Dump the list into the file using pickle
        pickle.dump(traj_not_linearized, file)

    return rrt

if __name__ == "__main__":

    start_time = time.time()
    rrt = main_planning()
    # print(f'total_time: {time.time() - start_time}')
    # print(f'tree: {[str(node) for node in rrt.random_tree]}')
    # print()
    
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
        