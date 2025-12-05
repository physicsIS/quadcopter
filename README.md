# 🛸 Simulación y Control de Quadcopter en Python

![Quadcopter](imgs/quadcopter_example.png)
*Así es el quadcopter que usamos en este proyecto.*

¡Hola! 👋 Este proyecto es sobre **simular y controlar un quadcopter** usando Python. Aquí combinamos matemáticas, física y visualización 3D para que puedas ver cómo se mueve el dron siguiendo distintas trayectorias, en particular, una trayectoría circular.

### Qué hace este proyecto

- Construye un **modelo matemático completo** del dron (ecuaciones de movimiento, linealización, matrices de transferencia).  
- Implementa un **controlador PID** para que el dron siga trayectorias suavizadas.  
- Permite ver la **simulación 3D en VPython**, con posición, orientación y hasta hélices girando.  
- Permite hacer análisis en **tiempo** y **frecuencia**.

---

## 📂 Estructura del repositorio

### Archivos de código

| Archivo | Qué hace |
|---------|----------|
| `math_model.ipynb` | Notebook que hace todo el análisis dinámico y de control, y genera la simulación de trayectorias. |
| `quadcopter_3D.py` | Modelo 3D del quadcopter y funciones para moverlo y rotarlo. |
| `funciones.py` | Funciones auxiliares, como convertir matrices simbólicas de SymPy para `control`. |
| `simulacion_traj_circ.py` | Lee CSV con posiciones y orientaciones y genera la animación 3D. |
| `drone_lib.py` | Controlador del dron, suavizado de trayectoria y resolución de ecuaciones diferenciales (RK4). |
| `prueba_simulacion.py` | Ejemplo independiente de cómo usar el modelo 3D y mover el dron. |

### Carpetas

| Carpeta | Contenido |
|---------|----------|
| `docs/` | Artículos y papers usados como referencia. |
| `imgs/` | Imágenes o PDFs generados, esquemas del dron y trayectorias. |
| `paper/` | Informe final del proyecto. |

---

## ⚙️ Cómo usarlo

1. Instala dependencias (Python 3.10+ recomendado):

```bash
pip install numpy sympy control vpython pandas matplotlib scienceplots
```
2. Ejecutar la simulación 3D:
```bash
python simulacion_traj_circ.py
```


---



# 🛸 Quadcopter Simulation and Control in Python

![Quadcopter](imgs/quadcopter_example.png) 
*This is the quadcopter used in the project.*

Hello! 👋 This project is about **simulating and controlling a quadcopter** using Python. We combine math, physics, and 3D visualization so you can see the drone following different trajectories.

### What this project does

- Builds a **complete mathematical model** of the drone (equations of motion, linearization, transfer matrices).  
- Implements a **controller** to follow smooth reference trajectories.  
- Allows **3D visualization in VPython**, showing position, orientation, and even spinning propellers.  
- Supports **time-domain** and **frequency-domain** analysis.

---

## 📂 Repository structure

### Code files

| File | Description |
|------|------------|
| `math_model.ipynb` | Notebook with all dynamic analysis, control, and trajectory simulations. |
| `quadcopter_3D.py` | 3D quadcopter model and functions to move and rotate it. |
| `funciones.py` | Helper functions, like converting symbolic SymPy matrices for `control`. |
| `simulacion_traj_circ.py` | Reads CSV with positions and orientations and animates the quadcopter. |
| `drone_lib.py` | Drone controller, trajectory smoothing, and differential equation solver (RK4). |
| `prueba_simulacion.py` | Independent example showing how to use the 3D model and move the drone. |

### Folders

| Folder | Content |
|--------|--------|
| `docs/` | Articles and papers used as reference. |
| `imgs/` | Images or PDFs generated, drone schemes and trajectories. |
| `paper/` | Final project report. |

---

## ⚙️ How to use

1. Install dependencies (Python 3.10+ recommended):

```bash
pip install numpy sympy control vpython pandas matplotlib scienceplots
```

2. Run 3D simulation:
```bash
python simulacion_traj_circ.py
```
