import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, diff, Matrix, cos, sin
from itertools import combinations
import os
import sys

# Get the directory containing the current file
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Current folder:", project_folder)

sys.path.append(project_folder)

from Utils.constants import *
from Utils.utils import *
x, y, θ, ϕ_1, v_1, ω, v_ϕ_1, a_v_1, a_ω = symbols('x y θ ϕ_1 v_1 ω v_ϕ_1 a_v_1 a_ω')

# P1_x=0.24 #x ruota 1
# P1_y=0.19 #y ruota 1
P1_x=P1[0]
P1_y=P1[1]

# Definizione del vettore di funzioni
f1 = cos(θ + ϕ_1)*v_1 + ω*(P1_x*sin(θ)+P1_y*cos(θ))
f2 = sin(θ + ϕ_1)*v_1 + ω*(P1_y*sin(θ)-P1_x*cos(θ))
f3 = ω
f4 = v_ϕ_1
f5 = a_v_1
f6 = a_ω


class Base_Model:
    def __init__(self,q,u,P,d):
        # Instance attributes
        self.f=Matrix([f1, f2,f3,f4])
        self.set_state(q)
        self.u=u
        self.P = P #wheel axis position (in robot frame)
        self.d = d #Wheel axis offset
    
    def set_state(self,new_q):
        if self.q is None or not np.array_equal(self.q,new_q):
            self.q=new_q
            self.update_linearization()


    def update_linearization(self):
        self.compute_linearized_A()
        self.compute_linearized_B()
        self.compute_linearized_c()

    
    def get_vector_state(self):
        return self.q
    
    def get_state(self):
        return self.q
    
    def get_input(self):
        return self.u
    
    def get_linearized_A(self):
        return self.A

    def get_linearized_B(self):
        return self.B

    def get_linearized_c(self):
        return self.c
    
    def get_symbolic_Matrix(self):
        return self.symbolic_A, self.symbolic_B
    
    def compute_linearized_A(self):
        jac_A = self.f.jacobian([x, y,θ, ϕ_1])
        A=jac_A.subs({x: self.q['x'],y: self.q['y'],θ: self.q['θ'], ϕ_1: self.q['ϕ_1'], v_1: 0, v_ϕ_1: 0,ω:0}).evalf() ##qua sostituiamo i valori che ci servono
        self.A=np.array(A)
        return self.A
    
    def compute_linearized_B(self):
        jac_B = self.f.jacobian([v_1, ω, v_ϕ_1])
        B = jac_B.subs({x: self.q['x'],y: self.q['y'],θ: self.q['θ'], ϕ_1: self.q['ϕ_1'], v_1: 0, v_ϕ_1: 0,ω:0}).evalf()
        self.B=np.array(B)
        return self.B

    def compute_linearized_c(self):
        computed_f= self.f.subs({x: self.q['x'],y: self.q['y'],θ: self.q['θ'], ϕ_1: self.q['ϕ_1'], v_1: 0, v_ϕ_1: 0,ω:0})
        #self.c=np.array(computed_f).ravel() - np.dot(np.array(self.A),self.get_vector_state())
        self.c=np.array(computed_f).ravel() - self.A @ (self.get_vector_state()) #scegliere o quello sopra o quello sotto
        # print(np.array(computed_f).ravel())
        # print(np.dot(np.array(self.A),self.get_vector_state()))
        # print(self.A @ (self.get_vector_state()))
        return self.c


class Dynamic_Model(Base_Model):
    def __init__(self,q,u,P, d):
        # Instance attributes
        #super().__init__(q,u)
        self.f=Matrix([f1, f2,f3,f4,f5,f6])
        self.symbolic_A=self.f.jacobian([x, y,θ, ϕ_1,v_1,ω])
        self.symbolic_B=self.f.jacobian([v_ϕ_1,a_v_1, a_ω])
        self.u=u
        self.q=None
        self.set_state(q)
        self.P = P #wheel axis position (in robot frame)
        self.d = d #Wheel axis offset
    
    def compute_linearized_A(self):
        #jac_A = self.f.jacobian([x, y,θ, ϕ_1,v_1,ω])
        theta=self.q[2]
        phi_1=self.q[3]
        vel_1=self.q[4]
        om=self.q[5]
        A=np.array([[0, 0, -vel_1*np.sin(theta+phi_1) + om*(P1_x*np.cos(theta)-P1_y*np.sin(theta)), -vel_1*np.sin(theta+phi_1), np.cos(theta+phi_1), P1_y*np.cos(theta)+P1_x*np.sin(theta)],
                    [0, 0, vel_1*np.cos(theta+phi_1) + om*(P1_y*np.cos(theta)+P1_x*np.sin(theta)), vel_1*np.cos(theta+phi_1), np.sin(theta+phi_1), -P1_x*np.cos(theta)+P1_y*np.sin(theta)],
           [0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])
        #A=self.symbolic_A.subs({x: self.q[0],y: self.q[1],θ: self.q[2], ϕ_1: self.q[3], v_1: self.q[4],ω: self.q[5], a_v_1: 0, v_ϕ_1: 0,a_ω:0}).evalf() ##qua sostituiamo i valori che ci servono
        self.A=A
        return self.A
    
    
    def compute_linearized_B(self):
        #self.symbolic_B = self.f.jacobian([v_ϕ_1,a_v_1, a_ω])
        #B = self.symbolic_B.subs({x: self.q[0],y: self.q[1],θ: self.q[2], ϕ_1: self.q[3], v_1: self.q[4],ω: self.q[5], a_v_1: 0, v_ϕ_1: 0,a_ω:0}).evalf()
        B=np.array([[0,0,0],
           [0,0,0],
           [0,0,0],
           [1,0,0],
           [0,1,0],
           [0,0,1]])
        self.B=B
        return self.B

    def compute_linearized_c(self):
        #computed_f= self.f.subs({x: self.q[0],y: self.q[1],θ: self.q[2], ϕ_1: self.q[3], v_1: self.q[4],ω: self.q[5], a_v_1: 0, v_ϕ_1: 0,a_ω:0}).evalf()
        #self.c=np.array(computed_f).ravel() - self.A @ (self.get_vector_state())
        theta=self.q[2]
        phi_1=self.q[3]
        vel_1=self.q[4]
        om=self.q[5]

        temp=np.array([vel_1*np.cos(theta + phi_1) + om*(P1_x*np.sin(theta)+P1_y*np.cos(theta)),
                       vel_1*np.sin(theta + phi_1) + om*(P1_y*np.sin(theta)-P1_x*np.cos(theta)),
                       om,
              0,
              0,
              0])
        self.c=temp - self.A @ self.q
        return self.c
    
    def get_v_joint(self, state, joint_position,beta_i):
        theta=state[2]
        phi_1=state[3]
        vel_1=state[4]
        om=state[5]
        d_x=np.cos(theta + phi_1)*vel_1 + om*(P1_x*np.sin(theta)+P1_y*np.cos(theta))
        d_y=np.sin(theta + phi_1)*vel_1 + om*(P1_y*np.sin(theta)-P1_x*np.cos(theta))
        d_theta=om
        # R = np.array([[np.cos(theta), -np.sin(theta)],
        #             [np.sin(theta), np.cos(theta)]])
        d_R=np.array([[-np.sin(theta), -np.cos(theta)],
                    [np.cos(theta), -np.sin(theta)]])*d_theta
        d_Osi=np.array([d_x,d_y])+d_R@np.array([joint_position[0],joint_position[1]])
        v_si=np.array([np.cos(theta+beta_i),np.sin(theta+beta_i)])@d_Osi
        return v_si


    
    def get_symbolic_Matrix(self):
        jac_A = self.f.jacobian([x, y,θ, ϕ_1,v_1,ω])
        jac_B = self.f.jacobian([v_ϕ_1,a_v_1, a_ω])
        return jac_A, jac_B
    
    
    def controllability(self):
        #A=self.get_linearized_A().astype(float)
        #B=self.get_linearized_B().astype(float)
        #A_B=A@B
        A,B=self.get_symbolic_Matrix()
        A_B=A@B
        A_A_B=A@A@B
        raggiung = Matrix(B.row_join(A_B))
        raggiung=Matrix(raggiung.row_join(A_A_B))
        print("A:",A)
        print("B:",B)
        print("A_B:",A_B)
        print ("raggiung:",raggiung)
        determinante=raggiung.rank()
        print(determinante)
        for col_indexes in combinations(range(9), 6):
            submatrix = raggiung[:, col_indexes]  # Estraiamo la sottomatrice 6x6
            det = submatrix.det()  # Calcoliamo il determinante della sottomatrice
            print(f"Determinante della sottomatrice con colonne {col_indexes}: {det}")
        #raggiung = (np.concatenate((B, A_B), axis=1))
        #determinante = np.linalg.det(raggiung)
        #print(determinante)
        
    def coordinating_function(self, previous_phi):
        
        P1 = self.P[0]
        v_1 = self.q[4]
        om = self.q[5]
        phi_1 = self.q[3]
        
        phi = [phi_1]
        
        for i in range(1,4):
            
            Pi = self.P[i]  
            phi_i_1 = np.arctan2((v_1 * np.sin(phi_1) + om * (Pi[0] - P1[0] )),( v_1 * np.cos(phi_1) + om * (P1[1] - Pi[1] )))  
            phi_i_2 = phi_i_1 + np.pi
            
            if abs(wrap_angle(phi_i_1 - previous_phi[i])) < abs(wrap_angle(abs(phi_i_2 - previous_phi[i]))) : 
                phi.append(phi_i_1)
                
            else:
                phi.append(phi_i_2)
            
        self.phi = phi

        return phi


    def wheels_steer_vel(self):
        
        # phi = self.coordinating_function() if self.phi==None else self.phi  # Not actually necessary to compute (since phi_i are functions of phi_1, v_1 and omega which are in the state) 
        
        P1 = self.P[0]
        
        phi_1 = self.q[3]
        v_1 = self.q[4]
        om = self.q[5]
        v_phi_1 = self.u[0]
        a_v_1 = self.u[1]
        a_om = self.u[2]
        
        v_phi = [v_phi_1]
        
        for i in range(1,4):
            
            Pi = P[i]
            v_phi_i = ((om*(P1[1] - Pi[1]) + v_1*np.cos(phi_1))*(a_v_1*np.sin(phi_1) - a_om*(P1[0] - Pi[0]) + v_1*v_phi_1*np.cos(phi_1)) + (om*(P1[0] - Pi[0]) - v_1*np.sin(phi_1))*(a_om*(P1[1] - Pi[1]) + a_v_1*np.cos(phi_1) - v_1*v_phi_1*np.sin(phi_1)))/(np.power(om*(P1[1] - Pi[1]) + v_1*np.cos(phi_1),2) + np.power(om*(P1[0] - Pi[0]) - v_1*np.sin(phi_1),2))
            v_phi.append(v_phi_i)
        
        self.v_phi = v_phi

        return v_phi

    def wheels_vel(self):
        
        phi = self.phi if hasattr(self,"phi") else self.coordinating_function() 
        v_phi = self.v_phi if hasattr(self,"v_phi") else self.wheel_steer_vel()
            
        P1=P[0]
        th = self.q[2]
        phi_1 =self.q[3]
        v_1 = self.q[4]
        om = self.q[5]
        d = self.d
        
        dx = np.cos(th+phi_1)*v_1 + om*(P1[0]*np.sin(th) + P1[1]*np.cos(th)) # cos(θ + ϕ_1)*v_1 + ω*(P1_x*sin(θ)+P1_y*cos(θ))
        dy =  np.sin(th+phi_1)*v_1 + om*(-P1[0]*np.cos(th) + P1[1]*np.sin(th))  # sin(θ + ϕ_1)*v_1 + ω*(P1_y*sin(θ)-P1_x*cos(θ))
        
        v_d = []
        
        for i in range(0,4):
            Pi = P[i]
            
            phi_i = phi[i]
            v_phi_i = v_phi[i]
            
            Px = Pi[0]
            Py = Pi[1]
            
            v_i = [(om*np.cos(phi_i + th) + v_phi_i*np.cos(phi_i + th))*d + dx - om*(Py*np.cos(th) + Px*np.sin(th)),
            (om*np.sin(phi_i + th) + v_phi_i*np.sin(phi_i + th))*d + dy + om*(Px*np.cos(th) - Py*np.sin(th))]
            
            v_d.append(v_i)
        
        self.v_d = v_d
        
        return v_d
    
    def q_d(self,state,input):
        theta=state[2]
        phi_1=state[3]
        vel_1=state[4]
        om=state[5]
        f=np.array([0,0,0,0,0,input[2]]) + np.array([0,0,0,0,input[1],0]) + np.array([np.cos(theta + phi_1)*vel_1 + om*(P1_x*np.sin(theta)+P1_y*np.cos(theta)),
        np.sin(theta + phi_1)*vel_1 + om*(P1_y*np.sin(theta)-P1_x*np.cos(theta)),om,input[0],0,0])
        return np.array(f)
    
    def evolution(self,x_0,inputs,tf,camp_s):
        t = np.arange(Decimal('0.0'), tf, Decimal(camp_s))
        if tf not in t:
            t = np.concatenate((t, np.array([tf])))
        n=len(t)
        x=np.zeros((n,*self.get_vector_state().shape))
        x[0]=x_0
        for i in range(n-1):
            h = t[i+1] - t[i]
            k1=self.q_d(x[i],inputs[i])
            k2=self.q_d(x[i] + k1*float(h)/2.,inputs[i])
            k3=self.q_d(x[i] + k2*float(h)/2.,inputs[i])
            k4=self.q_d(x[i] + k3*float(h),inputs[i])
            x[i+1]=x[i] + (float(h)/6.)*(k1 + 2*k2 + 2*k3 + k4)
            #x[i+1]= x[i] + (self.model.get_linearized_A()@ x[i] + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c())*h
        #print(len(x))
        #print(x[:10])
        return x
        
if __name__ == "__main__":
    new_q=np.array([0, 0, 0, 0, 1, 1])
    u=np.array([1, 2, 3])
    d = 1
    model=Dynamic_Model(new_q,u,P, d)
    print(model.q_d(new_q,u))
    #print(model.coordinating_function())
    #model.get_process(new_q,np.array([1,-1]),np.pi)