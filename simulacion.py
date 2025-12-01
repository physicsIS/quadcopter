from vpython import *
import numpy as np
from quadcopter_3Dmodel import Quadcopter


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
# wtext crea texto en el panel HTML del canvas (no rotante), ideal para mostrar el tiempo.
time_label = wtext(text="t = 0.00 s")

# =========================================
# Crear quadcopter
# =========================================
drone = Quadcopter()
drone.set_position([0, 0.2, 0])

# =========================================
# Movimiento de prueba (suave y natural)
# =========================================
t = 0

def smooth_trajectory(t):
    """Genera una trayectoria suave para demostrar el movimiento."""
    x = 0.8 * np.sin(0.3 * t)
    y = 0.2 + 0.1 * np.sin(0.5 * t)
    z = 0.8 * np.cos(0.3 * t)
    return np.array([x, y, z])


def smooth_orientation(t):
    """Ángulos pequeños y suaves para parecer realistas."""
    roll  = 0.20 * np.sin(0.5 * t)
    pitch = 0.18 * np.sin(0.4 * t + 1.0)
    yaw   = 0.30 * np.sin(0.2 * t)
    return roll, pitch, yaw


# =========================================
# Bucle principal
# =========================================
while True:
    rate(60)
    dt = 0.016  # 60 FPS
    t += dt
    time_label.text = f"t = {t:.2f} s"

    # --- Movimiento ---
    pos = smooth_trajectory(t)
    drone.set_position(pos)

    # --- Rotación natural ---
    roll, pitch, yaw = smooth_orientation(t)
    drone.set_euler(roll, pitch, yaw)

    # --- Hélices girando ---
    motor_speeds = [40, 40, 40, 40]  # velocidad fija
    drone.spin_props(motor_speeds, dt)
