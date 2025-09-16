import sys
sys.path.append(r'C:\Users\Luca\AMR23-FP7-MPSWMR\AMR23-FP7-MPSWMR\Model')
from old_model_creator import *
import sympy
import time
import numpy as np
from decimal import Decimal


class prova():
    def __init__(self):

        expanded_q={
            'x':1,
            'y':1,
            'θ':1,
            'ϕ_1':1,
            'v_1':1,
            'ω':1
        }
        my_u=np.array([0,0,0])
        self.model=Dynamic_Model(expanded_q,my_u)
        self.R=[[1,0,0],[0,1,0],[0,0,1]]
        return

    def optimal_trajectory(self,q0,q1,tau_step,frequency):

        self.model.set_state(q0)

        tau = Decimal('0.0')
        tau_step_s=str(tau_step)
        #tau = 0.0
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
            #tau+= tau_step
            iter += 1
            q_bar[iter] = self.RK4(self.q_bar_d, q_bar[iter-1], tau_step,frequency)
            G[iter] = self.RK4(self.G_d, G[iter-1], tau_step,frequency) #qua però facendo così q_bar e G escono più precisi perchè in ogni intervallo ne calcolo tanti
            c[iter] = self.cost_fn(G[iter], q1 ,q_bar[iter], tau)
            print(c[iter])
            print(tau)

            if c[iter] < c[iter_star]:
                if c[iter]>0:
                    iter_star = iter
                    tau_star = tau
        d_star = np.linalg.pinv(G[iter_star]) @ (q1 - q_bar[iter_star])
        state_f = np.concatenate([q1, d_star])
        print("taustar",tau_star)

        temp_path, temp_y = self.back_RK4(self.ODE_state, state_f, tau_star,frequency) #valutare se fare anche qua come sopra

        computed_tau=np.arange(0,len(temp_path),frequency*tau_step, dtype=int)
        if len(temp_path)-1 not in computed_tau:
            computed_tau=np.concatenate((computed_tau,np.array([len(temp_path)-1])))
        path=[temp_path[i] for i in computed_tau]
        y=[temp_y[i] for i in computed_tau]

        RB = np.linalg.inv(self.R) @ self.model.get_linearized_B().T
        print("y_len",len(temp_y))
        print("path_len",len(temp_path))
        #print(path)
        #inputs = [RB @ temp_y[i] for i in range(len(temp_y))]
        inputs=temp_y
        return c[iter_star],temp_path,inputs,tau_star
    
    def evolution(self,inputs,tf,freq):
        t = np.arange(0, float(tf), 1/(freq))
        if tf not in t:
            t = np.concatenate((t, np.array([tf])))
        x=np.zeros((len(inputs),*self.model.get_vector_state().shape))
        x[0]=self.model.get_vector_state()
        for i in range(len(inputs)-1):
            h = t[i+1] - t[i]
            k1=self.model.get_linearized_A()@ x[i] + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c()
            k2=self.model.get_linearized_A()@ (x[i] + k1*h/2.) + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c()
            k3=self.model.get_linearized_A()@ (x[i] + k2*h/2.) + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c()
            k4=self.model.get_linearized_A()@ (x[i] + k3*h) + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c()
            x[i+1]=x[i] + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)
            #x[i+1]= x[i] + (self.model.get_linearized_A()@ x[i] + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c())*h
        print(len(x))
        print(x)
        return x[-1]
    
    def evolution_2(self,x_0,inputs,tf,freq):
        t = np.arange(0, float(tf), 1/(freq))
        if tf not in t:
            t = np.concatenate((t, np.array([float(tf)])))
        n=len(t)
        x=np.zeros((n,*self.model.get_vector_state().shape))
        x[0]=x_0
        for i in range(n-1):
            h = t[i+1] - t[i]
            k1=self.model.get_linearized_A()@ x[i] + self.model.get_linearized_B()@ np.linalg.inv(self.R) @ self.model.get_linearized_B().T @inputs[i] + self.model.get_linearized_c()
            k2=self.model.get_linearized_A()@ (x[i] + k1*h/2.) + self.model.get_linearized_B()@ np.linalg.inv(self.R) @ self.model.get_linearized_B().T @(inputs[i]+k1*h/2. )+ self.model.get_linearized_c()
            k3=self.model.get_linearized_A()@ (x[i] + k2*h/2.) + self.model.get_linearized_B()@ np.linalg.inv(self.R) @ self.model.get_linearized_B().T @(inputs[i]+k2*h/2. ) + self.model.get_linearized_c()
            k4=self.model.get_linearized_A()@ (x[i] + k3*h) + self.model.get_linearized_B()@ np.linalg.inv(self.R) @ self.model.get_linearized_B().T @(inputs[i]+k3*h ) + self.model.get_linearized_c()
            x[i+1]=x[i] + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)
            #x[i+1]= x[i] + (self.model.get_linearized_A()@ x[i] + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c())*h
        print(len(x))
        print(x[:10])
        return x[-1]
    
    def evolution_3(self,p_0,inputs,tf,freq):
        t = np.arange(0, float(tf), 1/(freq))
        if tf not in t:
            t = np.concatenate((t, np.array([float(tf)])))
        n=len(t)
        p=np.zeros((n,*self.model.get_vector_state().shape))
        p[0]=p_0
        for i in range(n-1):
            h = t[i+1] - t[i]
            p[i+1]= p[i] + (self.model.get_linearized_A()@ p[i] + self.model.get_linearized_B()@ np.linalg.inv(self.R) @ self.model.get_linearized_B().T @ inputs[i] + self.model.get_linearized_c())*h
        print(len(p))
        print(p[:10])
        return p[-1]


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
        t = np.arange(0, float(tf), 1/(freq))
        if tf not in t:
            t = np.concatenate((t, np.array([float(tf)])))
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
        # print("state iniziale:", flipped_state[0])
        # print("state finale:",flipped_state[n-1])
        # tempo=np.zeros((2,12))
        # tempo[0]=flipped_state[0]
        # tempo[1]=flipped_state[n-1]
        # print(tempo[:,:6])
        # print(tempo[:,6:])
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
        # det=np.linalg.det(G)
        # if det==0:
        #     print("non si può invertire")
        #     return t
        c = float(t) + (q1 - q_bar).T @ np.linalg.pinv(G) @ (q1 - q_bar)
        
        return c

azz=prova()
q_iniziale=np.array([45, 45, 0, 0,1,1]) 
q_finale=np.array([43, 43, 0, 0,1,1]) 
#q_finale=np.array([6, 6, 6, 6]) #quando i valori finali sono uguali approssima bene, altrimenti no
start_time = time.time()
ciao,app,ul,finale_taim=azz.optimal_trajectory(q_iniziale,q_finale,0.1,100)

end_time = time.time()
elapsed_time = end_time - start_time

print("Tempo impiegato:", elapsed_time, "secondi")


print("tempo finale:",finale_taim)
print("costo",ciao)
print("lunghezza del tragitto", len(app))
print(app)







# kk1=A@q0 +c
# print(kk1)
# print(A@(q0+kk1) +c)
# print()
#G_0= np.ones((azz.model.get_linearized_A().shape))
#azz.RK4(azz.q_bar_d(),q_,q0,tau,freq)
# kk1=A@G_0 +G_0@A.T + B @ np.linalg.inv(azz.R) @B.T
# print(kk1)
# print(A@(G_0+kk1*1/2) +(G_0 +kk1*1/2)@A.T + B @ np.linalg.inv(azz.R) @B.T)
# print()
#azz.RK4(azz.G_d(),G_,G_0,tau,freq)
#azz.c_star(azz.model.get_vector_state(),q_final)