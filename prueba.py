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

# =========================================
# Crear quadcopter
# =========================================
drone = Quadcopter()
drone.set_position([0, 1, 0])