import numpy as np
import matplotlib.pyplot as plt

def uniciclo_cinematica(v, omega, theta, dt, duration):
    x, y, theta = 10.0, 0.0, theta
    x_history, y_history, theta_history = [x], [y], [theta]
    for _ in np.arange(0, duration, dt):
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot= omega
        
        x_history.append(x)
        y_history.append(y)
        theta_history.append(theta)

        # Aggiorna le variabili di stato
        x += x_dot * dt
        y += y_dot * dt
        theta += theta_dot* dt
        # Aggiorna il grafico interattivo
        plt.plot(x_history, y_history, color='blue')
        plt.grid(True)
        plt.scatter(x_history[-1], y_history[-1], color='red')
        plt.pause(dt)  # Pausa per rendere il grafico interattivo
        

    plt.show()
    #return x_history, y_history, theta_history

# Parametri del uniciclo
v = 1.0  # Velocità lineare
omega = 1.0  # Velocità angolare
theta = 0.0  # Angolo di orientamento iniziale

# Parametri della simulazione
dt = 0.1  # Passo di campionamento
duration = 10.0  # Durata della simulazione

# Simula il movimento del uniciclo
plt.ion()
#plt.figure(figsize=(8, 8))
plt.ylim(top=8,bottom=-8)
plt.xlim(right=8,left=-8)

uniciclo_cinematica(v, omega, theta, dt, duration)

# Visualizza i risultati
# plt.plot(x_history, y_history)
# plt.scatter(x_history, y_history, color='red', label='Punti di Campionamento')
# plt.title('Simulazione del movimento del uniciclo')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
plt.ioff()
plt.show() ##per mantenere il grafico
