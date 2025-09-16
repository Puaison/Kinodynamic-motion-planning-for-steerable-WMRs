import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, diff, Matrix, cos, sin
from itertools import combinations

x, y, θ, ϕ_1, v_1, ω, v_ϕ_1, a_v_1, a_ω = symbols('x y θ ϕ_1 v_1 ω v_ϕ_1 a_v_1 a_ω')

# Definizione del vettore di funzioni
f1 = cos(θ + ϕ_1)*v_1 + ω*(sin(θ)+cos(θ))
f2 = sin(θ + ϕ_1)*v_1 + ω*(sin(θ)-cos(θ))
f3 = ω
f4 = v_ϕ_1
f5 = a_v_1
f6 = a_ω


class Base_Model:
    def __init__(self,q,u):
        # Instance attributes
        self.A=None
        self.B=None
        self.c=None
        self.set_state(q)
        self.f=Matrix([f1, f2,f3,f4])
        self.temp_q=None
        self.u=u
    
    def set_state(self,new_q):
        if isinstance(new_q, dict):
            self.q=new_q
        else:
            temp_dict={}
            temp_dict['x']=new_q[0]
            temp_dict['y']=new_q[1]
            temp_dict['θ']=new_q[2]
            temp_dict['ϕ_1']=new_q[3]
            self.q=temp_dict
        if self.A is not None:
            self.compute_linearized_A()
        if self.B is not None:
            self.compute_linearized_B()
        if self.c is not None:
            self.compute_linearized_c()

    
    def get_vector_state(self):
        return np.array(list(self.q.values()))
    
    def get_state(self):
        return self.q
    
    def get_input(self):
        return self.u
    
    def get_linearized_A(self):
        if self.A is None:
            print("richiedo A")
            self.compute_linearized_A()
        return self.A

    def get_linearized_B(self):
        if self.B is None:
            self.compute_linearized_B()
        return self.B

    def get_linearized_c(self):
        if self.c is None:
            self.get_linearized_A()
            self.compute_linearized_c()
        return self.c
    
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

    def is_controllable(self):
        n = self.get_linearized_A().shape[0]  # Dimensione dello stato
        m = self.get_linearized_B().shape[1]  # Dimensione dell'ingresso

        # Costruisci la matrice di raggiungibilità
        R = np.zeros((n, n*m))
        for i in range(n):
            R[:, i*m:(i+1)*m] = np.linalg.matrix_power(self.get_linearized_A(), i+1) @ self.get_linearized_B()

        # Controlla se il rango della matrice di raggiungibilità è uguale alla dimensione dello stato
        return np.linalg.matrix_rank(R) == n

class Dynamic_Model(Base_Model):
    def __init__(self,q,u):
        # Instance attributes
        super().__init__(q,u)
        self.f=Matrix([f1, f2,f3,f4,f5,f6])

    def set_state(self,new_q):
        if isinstance(new_q, dict):
            self.q=new_q
        else:
            temp_dict={}
            temp_dict['x']=new_q[0]
            temp_dict['y']=new_q[1]
            temp_dict['θ']=new_q[2]
            temp_dict['ϕ_1']=new_q[3]
            temp_dict['v_1']=new_q[4]
            temp_dict['ω']=new_q[5]
            self.q=temp_dict
        if self.A is not None:
            self.compute_linearized_A()
        if self.B is not None:
            self.compute_linearized_B()
        if self.c is not None:
            self.compute_linearized_c()
    
    def compute_jac_A(self):
        jac_A = self.f.jacobian([x, y,θ, ϕ_1,v_1,ω])
        return jac_A
    
    def compute_linearized_A(self):
        jac_A = self.f.jacobian([x, y,θ, ϕ_1,v_1,ω])
        A=jac_A.subs({x: self.q['x'],y: self.q['y'],θ: self.q['θ'], ϕ_1: self.q['ϕ_1'], v_1: self.q['v_1'],ω: self.q['ω'], a_v_1: 0, v_ϕ_1: 0,a_ω:0}).evalf() ##qua sostituiamo i valori che ci servono
        self.A=np.array(A)
        return self.A
    

    def compute_jac_B(self):
        jac_B = self.f.jacobian([v_ϕ_1,a_v_1, a_ω])
        return jac_B
    
    def compute_linearized_B(self):
        jac_B = self.f.jacobian([v_ϕ_1,a_v_1, a_ω])
        B = jac_B.subs({x: self.q['x'],y: self.q['y'],θ: self.q['θ'], ϕ_1: self.q['ϕ_1'], v_1: self.q['v_1'],ω: self.q['ω'], a_v_1: 0, v_ϕ_1: 0,a_ω:0}).evalf()
        self.B=np.array(B)
        return self.B

    def compute_linearized_c(self):
        computed_f= self.f.subs({x: self.q['x'],y: self.q['y'],θ: self.q['θ'], ϕ_1: self.q['ϕ_1'], v_1: self.q['v_1'],ω: self.q['ω'], a_v_1: 0, v_ϕ_1: 0,a_ω:0}).evalf()
        #self.c=np.array(computed_f).ravel() - np.dot(np.array(self.A),self.get_vector_state())
        self.c=np.array(computed_f).ravel() - self.A @ (self.get_vector_state()) #scegliere o quello sopra o quello sotto
        # print(np.array(computed_f).ravel())
        # print(np.dot(np.array(self.A),self.get_vector_state()))
        # print(self.A @ (self.get_vector_state()))
        return self.c
    def get_symbolic_Matrix(self):
        jac_A = self.f.jacobian([x, y,θ, ϕ_1,v_1,ω])
        jac_B = self.f.jacobian([v_ϕ_1,a_v_1, a_ω])
        return jac_A, jac_B

if __name__ == "__main__":
    initial_q={
    'x':1,
    'y':1,
    'θ':1,
    'ϕ_1':1
        }        
    expanded_q={
    'x':10,
    'y':20,
    'θ':3,
    'ϕ_1':4,
    'v_1':5,
    'ω':6
    }
    my_u="prova"
    b=Dynamic_Model(expanded_q,my_u)
    print("A:",b.get_linearized_A())
    print("B:",b.get_linearized_B())
    print("c:",b.get_linearized_c())
