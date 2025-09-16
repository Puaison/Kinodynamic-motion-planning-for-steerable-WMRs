import numpy as np
import matplotlib.pyplot as plt
import math

def uniciclo_cinematica(v, omega, x,y,theta, dt, duration):
    x, y, theta = x, y, theta
    x_history, y_history, theta_history = [x], [y], [theta]
    

    for _ in np.arange(0, duration, dt):
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        
        x += x_dot * dt
        y += y_dot * dt
        theta += omega * dt

        x_history.append(x)
        y_history.append(y)
        theta_history.append(theta)

    return x_history, y_history, theta_history

def uniciclo_cinematica_rk4(v, omega, x,y,theta, dt, duration):
    x, y, theta = x, y, theta
    x_history, y_history, theta_history = [x], [y], [theta]
    

    for _ in np.arange(0, duration, dt):

        k1=v * np.cos(theta)
        k2= v * np.cos(theta +(1/2)*k1*dt)
        k3= v * np.cos(theta +(1/2)*k2*dt)
        k4= v * np.cos(theta +(k3*dt))


        x += (dt/6)*(k1+2*k2+2*k3+k4)


        k1=v * np.sin(theta)
        k2= v * np.sin(theta +(1/2)*k1*dt)
        k3= v * np.sin(theta +(1/2)*k2*dt)
        k4= v * np.sin(theta +(k3*dt))


        y += (dt/6)*(k1+2*k2+2*k3+k4)


        theta += omega * dt

        x_history.append(x)
        y_history.append(y)
        theta_history.append(theta)

    return x_history, y_history, theta_history

def simulazione_multipla(num_unicicli, v, omega, dt, duration):
    plt.ion()
    plt.figure(figsize=(8, 8))
    # plt.ylim(top=8,bottom=-8)
    # plt.xlim(right=8,left=-8)
    unicicli=[]
    x1_start=0.0
    y1_start=0.0
    x2_start=0.0
    y2_start=0.0
    theta1_start=0.0
    theta2_start=0.0
    #theta2_start=math.pi/2
    unicicli.append({'x': x1_start, 'y': y1_start, 'theta': theta1_start, 'n':1})
    unicicli.append({'x': x2_start, 'y': y2_start, 'theta': theta2_start, 'n':2})
    #print(unicicli[0]['x'])

    t=0
    for _ in np.arange(0, duration, dt):
        plt.clf()  # Pulisce il grafico ad ogni iterazione
        # plt.ylim(top=8,bottom=-8)
        # plt.xlim(right=8,left=-8)
        t+=dt
        for uniciclo in unicicli:
            if uniciclo['n']==1:
                x_hist, y_hist, _ = uniciclo_cinematica(v, omega,x1_start,y1_start, theta1_start, dt, t)
            else:
                x_hist, y_hist, _ = uniciclo_cinematica_rk4(v, omega,x2_start,y2_start, theta2_start, dt, t)
            uniciclo['x'] = x_hist[-1]
            uniciclo['y'] = y_hist[-1]
            plt.plot(x_hist, y_hist, label=f'Uniciclo {uniciclo["theta"]:.2f}')
            plt.scatter(uniciclo['x'], uniciclo['y'], color='red')
            
        plt.title('Simulazione di più unicicli')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.pause(dt)

    plt.show()
    plt.ioff()

# Parametri della simulazione
num_unicicli = 2
v = 1.0  # Velocità lineare
omega = 0.1  # Velocità angolare
dt = 0.1  # Passo di campionamento
duration = 10.0  # Durata della simulazione

# plt.ylim(top=8,bottom=-8)
# plt.xlim(right=8,left=-8)
simulazione_multipla(num_unicicli, v, omega, dt, duration)
plt.show()
