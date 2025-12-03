"""
Simulación de dron con control PID y dinámica linealizada (convención A).

Este módulo incluye:
- Clase PID para controladores generales.
- Funciones de trayectorias (círculo) y blending.
- Dinámica linealizada del dron.
- Controladores externos (posición) e internos (actitud).
- Simulación genérica con integración RK4.
"""

import numpy as np

# ===============================
# Clase PID
# ===============================

class PID:
    """
    Controlador PID general para regular errores de forma proporcional, integral y derivativa.

    Calcula la señal de control como u = Kp*error + Ki*integral(error) + Kd*derivada(error).

    Args:
        kp (float): Ganancia proporcional.
        ki (float): Ganancia integral.
        kd (float): Ganancia derivativa.
        integ_lim (float): Límite absoluto del término integral para evitar windup.
        out_lim (float, optional): Límite absoluto de la salida de control.

    Methods:
        reset(): Reinicia los estados internos (integral y error previo).
        __call__(error, dt): Calcula la señal de control para un error dado y paso temporal dt.

    Example:
        >>> pid = PID(kp=1.0, ki=0.1, kd=0.5)
        >>> u = pid(error=0.2, dt=0.01)
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, integ_lim=10.0, out_lim=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integ, self.prev = 0.0, None
        self.integ_lim, self.out_lim = integ_lim, out_lim

    def reset(self):
        """Reinicia los estados internos del PID (integral y error previo)."""
        self.integ, self.prev = 0.0, None

    def __call__(self, error, dt):
        """Calcula la señal de control PID para un error dado y paso temporal dt."""
        P = self.kp * error
        self.integ += error * dt
        self.integ = np.clip(self.integ, -self.integ_lim, self.integ_lim)
        I = self.ki * self.integ
        D = 0.0 if self.prev is None else self.kd * (error - self.prev) / dt
        self.prev = error
        u = P + I + D
        if self.out_lim is not None:
            u = np.clip(u, -self.out_lim, self.out_lim)
        return u

# ===============================
# Funciones de trayectoria
# ===============================

def trajectory_circle(t, R=1.0, w=0.03, z0=1.0):
    """
    Genera una trayectoria circular con velocidad y aceleración.

    Calcula la posición, velocidad y aceleración deseada para un dron siguiendo un círculo horizontal
    de radio R a velocidad angular w y altura constante z0.

    Args:
        t (float): Tiempo actual de la simulación.
        R (float): Radio del círculo.
        w (float): Velocidad angular de la trayectoria.
        z0 (float): Altura constante de la trayectoria.

    Returns:
        tuple:
            p_ref (np.ndarray): Posición deseada [x, y, z, psi].
            v_ref (np.ndarray): Velocidad deseada [vx, vy, vz, wz].
            a_ref (np.ndarray): Aceleración deseada [ax, ay, az, az].
    
    Example:
        >>> p_ref, v_ref, a_ref = trajectory_circle(1.0, R=2.0, w=0.1, z0=1.5)
    """
    xd, yd, zd = R*np.cos(w*t), R*np.sin(w*t), z0
    vxd, vyd, vzd = -R*w*np.sin(w*t), R*w*np.cos(w*t), 0.0
    axd, ayd, azd = -R*w*w*np.cos(w*t), -R*w*w*np.sin(w*t), 0.0
    return np.array([xd, yd, zd, 0.0]), np.array([vxd, vyd, vzd, 0.0]), np.array([axd, ayd, azd, 0.0])

def trajectory_lemniscate(t, a=1.0, w=0.2, z0=1.0):
    """
    Trayectoria lemniscata (∞) en el plano z=z0.

    Calcula posición, velocidad y aceleración para un dron siguiendo 
    una lemniscata de Gerono: x = a*cos(theta), y = a*cos(theta)*sin(theta)

    Args:
        t (float): Tiempo actual.
        a (float): Escala de la lemniscata (tamaño del lóbulo).
        w (float): Velocidad angular de recorrido.
        z0 (float): Altura constante del dron.

    Returns:
        p_ref, v_ref, a_ref (np.ndarray): Posición, velocidad y aceleración [x,y,z,psi].
    """
    theta = w*t
    x = a * np.cos(theta)
    y = a * np.cos(theta) * np.sin(theta)
    z = z0

    # Derivadas exactas
    dx = -a * w * np.sin(theta)
    dy = a * w * (np.cos(theta)**2 - np.sin(theta)**2)
    dz = 0.0

    ddx = -a * w**2 * np.cos(theta)
    ddy = -4 * a * w**2 * np.cos(theta) * np.sin(theta)
    ddz = 0.0

    p_ref = np.array([x, y, z, 0.0])
    v_ref = np.array([dx, dy, dz, 0.0])
    a_ref = np.array([ddx, ddy, ddz, 0.0])
    return p_ref, v_ref, a_ref



# ===============================
# Blending de trayectoria
# ===============================

def blend_coeff(t, Tr):
    """
    Calcula coeficientes de blending suave usando polinomio trigonométrico.

    Args:
        t (float): Tiempo actual de la simulación.
        Tr (float): Tiempo total de transición/blending.

    Returns:
        tuple:
            s (float): Coeficiente de blending de posición.
            s_dot (float): Derivada temporal del coeficiente (velocidad de blending).
            s_ddot (float): Segunda derivada temporal del coeficiente (aceleración de blending).

    Example:
        >>> s, s_dot, s_ddot = blend_coeff(2.0, 5.0)
    """
    if t <= 0: return 0.0,0.0,0.0
    if t >= Tr: return 1.0,0.0,0.0
    s = 0.5*(1 - np.cos(np.pi*t/Tr))
    s_dot = 0.5*(np.pi/Tr)*np.sin(np.pi*t/Tr)
    s_ddot = 0.5*(np.pi/Tr)**2 * np.cos(np.pi*t/Tr)
    return s, s_dot, s_ddot

def blended_trajectory(t, traj_func, Tr=5.0, p_init=None, v_init=None, **traj_params):
    """
    Genera una trayectoria suavemente blendada desde la posición inicial del dron a la deseada.

    Combina la trayectoria deseada con un blending polinómico para evitar saltos iniciales.

    Args:
        t (float): Tiempo actual de la simulación.
        traj_func (function): Función de trayectoria que devuelve posición, velocidad y aceleración.
        Tr (float): Tiempo de blending.
        p_init (np.ndarray, optional): Posición inicial del dron [x, y, z, psi]. Por defecto [0,0,0,0].
        v_init (np.ndarray, optional): Velocidad inicial del dron [vx, vy, vz, wz]. Por defecto [0,0,0,0].
        **traj_params: Parámetros adicionales para la función de trayectoria.

    Returns:
        tuple:
            p_ref (np.ndarray): Posición blendada [x, y, z, psi].
            v_ref (np.ndarray): Velocidad blendada [vx, vy, vz, wz].
            a_ref (np.ndarray): Aceleración blendada [ax, ay, az, az].
    """
    p_c, v_c, a_c = traj_func(t, **traj_params)

    if p_init is None:
        p0 = np.zeros(4)
    else:
        p0 = np.array(p_init)

    if v_init is None:
        v0 = np.zeros(4)
    else:
        v0 = np.array(v_init)

    s, s_dot, s_ddot = blend_coeff(t, Tr)

    # Blending desde la posición inicial actual
    p_ref = (1-s)*p0 + s*p_c
    v_ref = (1-s)*v0 + s_dot*(p_c - p0) + s*v_c
    a_ref = s_ddot*(p_c - p0) + 2*s_dot*(v_c - v0) + s*a_c

    return p_ref, v_ref, a_ref


# ===============================
# Dinámica linealizada del dron
# ===============================

def dynamics(state, Tprime, Nx, Ny, Nz, A, B, C, D, g, gamma, epsilon):
    """
    Calcula la dinámica linealizada del dron (12 estados).

    Estados: [x, vx, y, vy, z, vz, phi, wx, theta, wy, psi, wz]

    Args:
        state (np.ndarray): Vector de estado actual (12 elementos).
        Tprime (float): Comando de empuje linealizado.
        Nx, Ny, Nz (float): Comandos de aceleración angular.
        Av, Bv, Cv, Dv (float): Coeficientes de la dinámica linealizada.
        g (float): Gravedad.
        gamma, epsilon (float): Parámetros físicos adicionales (no utilizados directamente aquí).

    Returns:
        np.ndarray: Derivadas temporales de los estados (dx/dt).

    Example:
        >>> dxdt = dynamics(state, Tprime, Nx, Ny, Nz, A, B, C, D, g, gamma, epsilon)
    """
    x, vx, y, vy, z, vz, phi, wx, theta, wy, psi, wz = state
    dxdt = vx; dvxdt = -B*vx + g*theta
    dydt = vy; dvydt = -C*vy - g*phi
    dzdt = vz; dvzdt = A*Tprime - D*vz
    dphidt = wx; dwxdt = Nx*gamma
    dthetadt = wy; dwydt = Ny*gamma
    dpsidt = wz; dwzdt = Nz*epsilon
    return np.array([dxdt,dvxdt,dydt,dvydt,dzdt,dvzdt,
                    dphidt,dwxdt,dthetadt,dwydt,dpsidt,dwzdt])

# ===============================
# Control externo e interno
# ===============================

def outer_controller(state, p_ref, v_ref, a_ref, params):
    """
    Control PID de posición -> aceleraciones deseadas.

    Args:
        state (np.ndarray): Estado actual del dron.
        p_ref (np.ndarray): Posición de referencia.
        v_ref (np.ndarray): Velocidad de referencia.
        a_ref (np.ndarray): Aceleración de referencia.
        params (dict): Diccionario de parámetros, incluyendo PIDs y dt.

    Returns:
        tuple: Aceleraciones deseadas (a_des_x, a_des_y, a_des_z).

    Example:
        >>> a_des_x, a_des_y, a_des_z = outer_controller(state, p_ref, v_ref, a_ref, params)
    """
    x,vx,y,vy,z,vz = state[0],state[1],state[2],state[3],state[4],state[5]
    pid_x, pid_y, pid_z = params["pid_x"], params["pid_y"], params["pid_z"]
    ex,ey,ez = p_ref[0]-x, p_ref[1]-y, p_ref[2]-z
    a_des_x = a_ref[0] + pid_x(ex, params["dt"])
    a_des_y = a_ref[1] + pid_y(ey, params["dt"])
    a_des_z = a_ref[2] + pid_z(ez, params["dt"])
    return a_des_x, a_des_y, a_des_z

def inner_controller(state, angles_ref, params):
    """
    Control PD de actitud -> aceleraciones angulares.

    Args:
        state (np.ndarray): Estado actual del dron.
        angles_ref (tuple): Ángulos deseados (phi_des, theta_des, psi_des).
        params (dict): Diccionario de parámetros con ganancias PD.

    Returns:
        tuple: Comandos de aceleración angular (Nx, Ny, Nz).

    Example:
        >>> Nx, Ny, Nz = inner_controller(state, (phi_des, theta_des, psi_des), params)
    """
    phi, theta, psi = state[6], state[8], state[10]
    wx, wy, wz = state[7], state[9], state[11]
    phi_des, theta_des, psi_des = angles_ref
    Nx =(params["kp_phi"]*(phi_des - phi) - params["kd_phi"]*wx)/params["gamma"]
    Ny = (params["kp_theta"]*(theta_des - theta) - params["kd_theta"]*wy)/params["gamma"]
    Nz = (params["kp_psi"]*(psi_des - psi) - params["kd_psi"]*wz)/params["epsilon"]
    return Nx, Ny, Nz

# ===============================
# Simulación RK4
# ===============================

def simulate(T, dt, dynamics_func, trajectory_func, outer_controller, inner_controller, state0, params):
    """
    Simulador genérico de dron usando integración RK4.

    Args:
        T (float): Tiempo total de simulación.
        dt (float): Paso temporal.
        dynamics_func (function): Función de dinámica.
        trajectory_func (function): Función de trayectoria.
        outer_controller (function): Controlador de posición.
        inner_controller (function): Controlador de actitud.
        state0 (np.ndarray): Estado inicial del dron.
        params (dict): Diccionario con todos los parámetros necesarios (PIDs, límites, constantes físicas).

    Returns:
        tuple: (tvec, hist, ref_hist, u_hist, Tprime_hist)
            - tvec (np.ndarray): Vector de tiempos.
            - hist (np.ndarray): Historial de estados (n,12).
            - ref_hist (np.ndarray): Historial de posiciones de referencia (n,4).
            - u_hist (np.ndarray): Historial de comandos [Tprime, Nx, Ny, Nz].
            - Tprime_hist (np.ndarray): Historial del comando de empuje Tprime.

    Example:
        >>> tvec, hist, ref_hist, u_hist, Tprime_hist = simulate(10.0, 0.01, dynamics, blended_trajectory, outer_controller, inner_controller, state0, params)
    """
    n = int(T/dt)+1
    tvec = np.linspace(0,T,n)
    hist = np.zeros((n,12))
    ref_hist = np.zeros((n,4))
    u_hist = np.zeros((n,4))
    Tprime_hist = np.zeros(n)
    state = np.array(state0, dtype=float)
    params["dt"] = dt
    for k in ["pid_x","pid_y","pid_z"]:
        if k in params: params[k].reset()
    for i,t in enumerate(tvec):
        p_ref, v_ref, a_ref = trajectory_func(t, **params.get("traj_params",{}))
        ref_hist[i,:] = p_ref
        a_des_x,a_des_y,a_des_z = outer_controller(state,p_ref,v_ref,a_ref,params)
        theta_des = (a_des_x + params["B"]*state[1]) / params["g"]
        phi_des   = -(a_des_y + params["C"]*state[3]) / params["g"]
        Tprime_cmd = (a_des_z + params["D"]*state[5]) / params["A"]
        theta_des = float(np.clip(theta_des, -params["max_angle"], params["max_angle"]))
        phi_des   = float(np.clip(phi_des, -params["max_angle"], params["max_angle"]))
        Tprime_cmd = float(np.clip(Tprime_cmd, params["Tprime_min"], params["Tprime_max"]))
        Nx_cmd, Ny_cmd, Nz_cmd = inner_controller(state,(phi_des,theta_des,p_ref[3]),params)
        Nx_cmd = float(np.clip(Nx_cmd, -params["max_ang_acc"], params["max_ang_acc"]))
        Ny_cmd = float(np.clip(Ny_cmd, -params["max_ang_acc"], params["max_ang_acc"]))
        Nz_cmd = float(np.clip(Nz_cmd, -params["max_ang_acc"], params["max_ang_acc"]))
        u_hist[i,:] = [Tprime_cmd,Nx_cmd,Ny_cmd,Nz_cmd]
        Tprime_hist[i] = Tprime_cmd
        k1 = dynamics_func(state,Tprime_cmd,Nx_cmd,Ny_cmd,Nz_cmd, params["A"],params["B"],params["C"],params["D"],params["g"],params["gamma"],params["epsilon"])
        k2 = dynamics_func(state+0.5*dt*k1,Tprime_cmd,Nx_cmd,Ny_cmd,Nz_cmd, params["A"],params["B"],params["C"],params["D"],params["g"],params["gamma"],params["epsilon"])
        k3 = dynamics_func(state+0.5*dt*k2,Tprime_cmd,Nx_cmd,Ny_cmd,Nz_cmd, params["A"],params["B"],params["C"],params["D"],params["g"],params["gamma"],params["epsilon"])
        k4 = dynamics_func(state+dt*k3,Tprime_cmd,Nx_cmd,Ny_cmd,Nz_cmd, params["A"],params["B"],params["C"],params["D"],params["g"],params["gamma"],params["epsilon"])
        state += (dt/6.0)*(k1+2*k2+2*k3+k4)
        state[7]*=0.999; state[9]*=0.999; state[11]*=0.999
        hist[i,:] = state
    return tvec,hist,ref_hist,u_hist,Tprime_hist
