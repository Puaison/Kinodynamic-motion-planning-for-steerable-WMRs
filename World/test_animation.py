import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# Dimensioni e posizioni iniziali
car_width, car_height = 4.0, 2.0
wheel_radius = 0.5
wheel_positions = [(1, 0.5), (4, 0.5), (1, 2.5), (4, 2.5)]  # Angoli della macchina

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, 5)
ax.set_ylim(0, 3)

# Disegna il corpo della macchina
car_body = Rectangle((0.5, 0.5), car_width, car_height, fc='blue', ec='black')
ax.add_patch(car_body)

# Disegna le ruote e le linee di orientamento
wheels = [Circle(pos, wheel_radius, fc='black', ec='black') for pos in wheel_positions]
# wheels = [Line2D([pos[0], pos[0]+wheel_radius], [y, y], color="green", lw=20) for pos in wheel_positions]
orientation_lines = []

for wheel in wheels:
    ax.add_patch(wheel)
    # Calcola il centro della ruota
    x, y = wheel.center
    print(f'wheel center: {x}, {y}')
    line = Line2D([x, x + wheel_radius], [y, y], color="green", lw=2)
    ax.add_line(line)
    orientation_lines.append(line)

for line in orientation_lines:
  print(line.get_xdata(), line.get_ydata())

def init():
    for line in orientation_lines:
      x0, x1 = line.get_xdata()
      y0, y1 = line.get_ydata()
      line.set_data([x0, x1], [y0, y1])
    return orientation_lines

# Funzione di aggiornamento per l'animazione
def update(frame):
    print("change")
    for line in orientation_lines:
        x, y = line.get_data()
        x0, x1 = line.get_xdata()
        y0, y1 = line.get_ydata()

        print(x0, x1, y0, y1)
        
        if frame % 2 == 0:
          line.set_data([x0, x0 + wheel_radius], [y0, y0])
        else:
          line.set_data([x0, x0], [y0, y0 + wheel_radius])


        # Aggiorna l'orientamento della linea
        # angle = np.deg2rad(frame * 10)  # Cambia l'angolo con il tempo
        # dx = wheel_radius * np.cos(angle)
        # dy = wheel_radius * np.sin(angle)
        # line.set_data([x[0], x[0] + dx], [y[0], y[0] + dy])
      
    return orientation_lines

ani = FuncAnimation(fig, update, frames=range(0, 36, 1), init_func=init, blit=True, interval=50, repeat=False)
plt.show()
