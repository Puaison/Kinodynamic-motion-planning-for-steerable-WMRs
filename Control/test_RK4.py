import sys
sys.path.append(r'C:\Users\Luca\AMR23-FP7-MPSWMR\AMR23-FP7-MPSWMR\Model')
from model_creator import *
import sympy
import time
import numpy as np

x_, y_, θ_, ϕ_1_, v_1_, ω_=symbols('x_ y_ θ_ ϕ_1_ v_1_ ω_')
G_=sympy.MatrixSymbol('G_',6,6)
#G__=np.array(G_)
q_=np.array([x_, y_, θ_, ϕ_1_, v_1_, ω_])


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
        my_u="prova"
        self.model=Dynamic_Model(expanded_q,my_u)
        self.R=[[1,0,0],[0,1,0],[0,0,1]]
        return

    def optimal_trajectory(self,q0,q1,tau_step,frequency):

        self.model.set_state(q0)

        tau = 0.0
        #tau_step = 0.1 #the sampling time
        iter_star = 0
        tau_star=0
        iter_max = 100000
        #frequency=100 #how many samples in one second
        #sampling_time=1/frequency

        c = np.zeros((iter_max))
        q_bar = np.zeros((iter_max, *self.model.get_vector_state().shape))
        G = np.zeros((iter_max, *self.model.get_linearized_A().shape))

        c[iter_star] = np.inf
        G[0] = np.zeros((self.model.get_linearized_A().shape))
        iter = 0
        q_bar[0] = q0
        while tau < c[iter_star] and iter<iter_max:
            
            tau+= tau_step
            #print("G=",G[iter])
            iter += 1
            q_bar[iter] = self.RK4(self.q_bar_d(),q_, q_bar[iter-1], tau_step,frequency)
            G[iter] = self.RK4(self.G_d(),G_, G[iter-1], tau_step,frequency) #qua però facendo così q_bar e G escono più precisi perchè in ogni intervallo ne calcolo tanti
            #print("G=",G[iter])
            c[iter] = self.cost_fn(G[iter], q1 ,q_bar[iter], tau)
            print(c[iter])
            print(tau)

            if c[iter] < c[iter_star]:
                iter_star = iter
                tau_star = tau
                #print(c[iter_star])
                #print(tau_star)
        #print(G[iter_star])
        d_star = np.linalg.inv(G[iter_star]) @ (q1 - q_bar[iter_star])
        state_f = np.concatenate([q1, d_star])
        print("taustar",tau_star)

        temp_path, temp_y = self.back_RK4(self.ODE_state, state_f, tau_star,frequency) #valutare se fare anche qua come sopra
        computed_tau=np.arange(0,len(temp_path),frequency*tau_step, dtype=int)
        if len(temp_path)-1 not in computed_tau:
            computed_tau=np.concatenate((computed_tau,np.array([len(temp_path)-1])))
        path=[temp_path[i] for i in computed_tau]
        y=[temp_y[i] for i in computed_tau]

        RB = np.linalg.inv(self.R) @ self.model.get_linearized_B().T
        print("y_len",len(y))
        print("path_len",len(path))
        #print(path)
        inputs = [RB @ y[i] for i in range(len(y))]
        return c[iter_star],path,inputs,tau_star
    
    def evolution(self,h,inputs):
        x=np.zeros((len(inputs)+1,*self.model.get_vector_state().shape))
        x[0]=self.model.get_vector_state()
        for i in range(len(inputs)):
            k1=self.model.get_linearized_A()@ x[i] + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c()
            k2=self.model.get_linearized_A()@ (x[i] + k1*h/2.) + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c()
            k3=self.model.get_linearized_A()@ (x[i] + k2*h/2.) + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c()
            k4=self.model.get_linearized_A()@ (x[i] + k3*h) + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c()
            x[i+1]=x[i] + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)
            #x[i+1]= x[i] + (self.model.get_linearized_A()@ x[i] + self.model.get_linearized_B()@inputs[i] + self.model.get_linearized_c())*sampling
        print(len(x))
        print(x)
        return x[-1]

                
    def from_values_to_dict(self,symbols, values):
        symbols_and_values={}
        if isinstance(symbols, sympy.MatrixSymbol):
            symbols_and_values[symbols]=sympy.Matrix(values)
            return symbols_and_values
        for symbol, value in zip(symbols,values):
            symbols_and_values[symbol]=value
        return symbols_and_values

    def RK4(self,f,variables_list,y0,tf,freq): #tf mi dice quali sono i secondi finali
        t = np.arange(0, tf, 1/(freq))
        if tf not in t:
            t = np.concatenate((t, np.array([tf])))
        n = len(t)
        y = np.zeros((n, *y0.shape))
        y[0] = y0
        for i in range(n - 1):
            h = t[i+1] - t[i] #h sarebbe l'intervallo do tempo. se h=2, vuol dire che passano due secondi
            k1=np.array(f.subs(self.from_values_to_dict(variables_list,y[i])))
            k2=np.array(f.subs(self.from_values_to_dict(variables_list,y[i]+ k1*h/2.)))
            k3=np.array(f.subs(self.from_values_to_dict(variables_list,y[i]+ k2*h/2.)))
            k4=np.array(f.subs(self.from_values_to_dict(variables_list,y[i]+ k3*h)))
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
            k1=f(state[i])
            k2 = f(state[i] - k1 * h / 2.)
            k3 = f(state[i] - k2 * h / 2.)
            k4 = f(state[i] - k3 * h)
            state[i+1] = state[i] - (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
        #poi qui dobbiamo flippare il valore state e suddividerlo in q e y
        flipped_state=state[::-1]
        q=flipped_state[:,:self.model.get_vector_state().shape[0]]
        y=flipped_state[:,self.model.get_vector_state().shape[0]:]
        return q,y
    
    
    def q_bar_d(self):
        f=self.model.get_linearized_A()@q_ + self.model.get_linearized_c()
        return sympy.Array(f)
    
    def G_d(self):
        G_d = self.model.get_linearized_A() @ G_ + G_ @ self.model.get_linearized_A().T + self.model.get_linearized_B() @ np.linalg.inv(self.R) @ self.model.get_linearized_B().T #transition function
        return sympy.Matrix(G_d)
    
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
        c = t + (q1 - q_bar).T @ np.linalg.inv(G) @ (q1 - q_bar)
        
        return c

azz=prova()
q_finale=np.array([2, 2, 2, 0, 0, 0])
import time
start_time = time.time()
ciao,app,ul,finale_taim=azz.optimal_trajectory(azz.model.get_vector_state(),q_finale,0.1,23)
end_time = time.time()
elapsed_time = end_time - start_time

print("Tempo impiegato:", elapsed_time, "secondi")
print("tempo finale:",finale_taim)
print("costo",ciao)
print("lunghezza del tragitto", len(app))
print(app)

#cost,final_path,u,star_tau=obj.optimal_trajectory(q0,q1,time_step,frquency)
#uau=azz.evolution(1/100,ul)
#print(uau)
#print(app[-1])
#print(app)







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