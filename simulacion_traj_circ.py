from vpython import *
import numpy as np
from quadcopter_3Dmodel import Quadcopter
import pandas as pd

# =========================================
# Configurar escena
# =========================================
scene = canvas(title="Quadcopter Simulation", width=900, height=600, background=color.white)
scene.lights = []
distant_light(direction=vector(1,1,1), color=color.white)
distant_light(direction=vector(-1,-1,-0.5), color=color.gray(0.5))

# =========================================
# Crear grilla en el piso
# =========================================
grid_size = 5
grid_step = 0.5

def create_grid():
    for x in arange(-grid_size, grid_size + grid_step, grid_step):
        curve(pos=[vector(x,0,-grid_size), vector(x,0,grid_size)], color=color.gray(0.85))
    for z in arange(-grid_size, grid_size + grid_step, grid_step):
        curve(pos=[vector(-grid_size,0,z), vector(grid_size,0,z)], color=color.gray(0.85))

create_grid()

# === Tiempo en pantalla (overlay) ===
time_label = wtext(text="t = 0.00 s")

# =========================================
# Crear quadcopter
# =========================================
drone = Quadcopter()
drone.set_position([0, 1.0, 0])

# =========================================
# Recoger datos de trayectoria circular desde CSV
# =========================================
data = pd.read_csv("drone_simulation_results.csv")
positions = data[['x', 'z', 'y']].to_numpy()
orientations = data[['phi', 'theta', 'psi']].to_numpy()  # en radianes
t = data['time'].to_numpy()

# =========================================
# Crear traza del dron
# =========================================
trajectory = curve(color=color.red, radius=0.008)  # rojo, un poco gruesa

# =========================================
# Bucle de animación
# =========================================
fps = len(t[t<1.0])
velocidad_reproduccion = 5.0  # 1x velocidad real
for i in range(len(t)):
    rate(fps*velocidad_reproduccion)  # limita la animación a 60 FPS

    # --- Actualizar tiempo en pantalla ---
    time_label.text = f"t = {t[i]:.2f} s --> x{velocidad_reproduccion}"

    # --- Movimiento ---
    pos = positions[i]
    drone.set_position(pos)

    # --- Agregar punto a la traza ---
    trajectory.append(vector(*pos))

    # --- Rotación ---
    roll, pitch, yaw = orientations[i]
    drone.set_taitbryan_yxz(roll, pitch, yaw)  # usa tu función Tait-Bryan
