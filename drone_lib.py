"""
Simulación de dron con control PID y dinámica linealizada.

Este módulo incluye:
- Clase PID para controladores generales.
- Funciones de trayectorias (círculo y lemniscata) y blending.
- Dinámica linealizada del dron (12 estados) alrededor del hover usando T' (Tprime).
- Controladores externos (posición → referencias) e internos (actitud → torques).
- Simulación genérica con integración RK4 que usa inputs físicos:
	T' (thrust incremental) y Nx, Ny, Nz (torques físicos).
"""

import numpy as np

# ===============================
# Clase PID
# ===============================

class PID:
	"""
	Controlador PID general.

	Calcula la señal de control:
		u = Kp*error + Ki*integral(error) + Kd*derivada(error)

	La clase implementa límite en el término integral (anti-windup por saturación)
	y un límite opcional de salida.

	Args:
		kp (float): Ganancia proporcional.
		ki (float): Ganancia integral.
		kd (float): Ganancia derivativa.
		integ_lim (float): Límite absoluto para la integral (previene windup).
		out_lim (float|None): Límite absoluto de la salida u (None = sin límite), es un límite simétrico.

	Methods:
		reset(): Reinicia integral y error previo.
		__call__(error, dt): Calcula salida del PID para un error y paso dt.

	Example:
		>>> pid = PID(kp=1.0, ki=0.1, kd=0.01, integ_lim=5.0, out_lim=10.0)
		>>> u = pid(0.2, 0.01)
	"""
	def __init__(self, kp=0.0, ki=0.0, kd=0.0, integ_lim=10.0, out_lim=None):
		self.kp =kp
		self.ki = ki
		self.kd = kd
		self.integ = 0.0 
		self.prev = None
		self.integ_lim = integ_lim
		self.out_lim = out_lim

	def reset(self):
		"""
		Reinicia los estados internos del PID.

		Restablece integral y valor previo del error a su estado inicial.
		"""
		self.integ = 0.0
		self.prev = None

	def __call__(self, error, dt):
		"""
		Calcula la salida del PID para un error dado y paso temporal dt.

		Args:
			error (float): Error de referencia - medida.
			dt (float): Paso temporal (segundos).

		Returns:
			u (float): Señal de control calculada por el PID (posiblemente saturada).
		"""

		# P
		P = self.kp * error
		# I
		self.integ += error * dt
		self.integ = np.clip(self.integ, -self.integ_lim, self.integ_lim)
		I = self.ki * self.integ
		# D
		D = 0.0 if self.prev is None else self.kd * (error - self.prev) / dt
		self.prev = error
		# PID 
		u = P + I + D
		if self.out_lim is not None:
			u = np.clip(u, -self.out_lim, self.out_lim)
		return u

# ===============================
# Funciones de trayectoria
# ===============================

def trajectory_circle(t: float, R: float = 1.0, w: float = 0.03, z0: float = 1.0):
	"""
	Genera una trayectoria circular horizontal (posición, velocidad, aceleración).

	La trayectoria es un círculo en el plano XY con radio R, velocidad angular w
	y altura constante z0. Devuelve posición, velocidad y aceleración en el formato
	[x, y, z, psi], [vx, vy, vz, wz], [ax, ay, az, az].

	Args:
		t (float): Tiempo actual (s).
		R (float): Radio del círculo (m).
		w (float): Velocidad angular de recorrido (rad/s).
		z0 (float): Altura constante (m).

	Returns:
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

def trajectory_lemniscate(t: float, a: float = 1.0, w: float = 0.2, z0: float = 1.0):
	"""
	Genera una trayectoria tipo lemniscata (Gerono) en el plano z = z0.

	Devuelve posición, velocidad y aceleración en el formato [x, y, z, psi].

	Args:
		t (float): Tiempo actual (s).
		a (float): Escala de la lemniscata (m).
		w (float): Velocidad angular paramétrica (rad/s).
		z0 (float): Altura constante (m).

	Returns:
		p_ref (np.ndarray): Posición deseada [x, y, z, psi].
		v_ref (np.ndarray): Velocidad deseada [vx, vy, vz, wz].
		a_ref (np.ndarray): Aceleración deseada [ax, ay, az, az].

	Example:
		>>> p_ref, v_ref, a_ref = trajectory_lemniscate(0.5, a=1.0, w=0.5, z0=1.2)
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

def trajectory_spiral(
		t: float,
		R0: float = 0.5,     # radio inicial
		R1: float = 2.0,     # radio final
		w: float = 0.2,      # velocidad angular
		z0: float = 1.0,     # altura inicial
		vz: float = 0.1      # velocidad vertical constante
	):
	"""
	Trayectoria helicoidal (espiral 3D) suave en formato [p, v, a].

	El radio crece linealmente:
		R(t) = R0 + (R1 - R0) * t / T_growth    (implícito)

	Args:
		t (float): Tiempo actual.
		R0 (float): Radio inicial.
		R1 (float): Radio final.
		w (float): Velocidad angular (rad/s).
		z0 (float): Altura inicial.
		vz (float): Velocidad vertical constante.

	Returns:
		p_ref: [x, y, z, psi]
		v_ref: [vx, vy, vz, wz]
		a_ref: [ax, ay, az, az]
	"""

	# Radio cambiando linealmente con t
	R = R0 + (R1 - R0) * t * 0.1
	dR = (R1 - R0) * 0.1        # derivada del radio
	ddR = 0.0                   # aceleración radial nula

	theta = w * t
	dtheta = w
	ddtheta = 0.0

	# Posición
	x = R * np.cos(theta)
	y = R * np.sin(theta)
	z = z0 + vz * t

	# Velocidad
	vx = dR * np.cos(theta) - R * np.sin(theta) * dtheta
	vy = dR * np.sin(theta) + R * np.cos(theta) * dtheta
	vz_val = vz

	# Aceleración
	ax = (ddR - R * dtheta**2) * np.cos(theta) \
		 - (2 * dR * dtheta + R * ddtheta) * np.sin(theta)

	ay = (ddR - R * dtheta**2) * np.sin(theta) \
		 + (2 * dR * dtheta + R * ddtheta) * np.cos(theta)

	az = 0.0

	p_ref = np.array([x, y, z, 0.0])
	v_ref = np.array([vx, vy, vz_val, 0.0])
	a_ref = np.array([ax, ay, az, 0.0])

	return p_ref, v_ref, a_ref


# ===============================
# Blending de trayectoria
# ===============================

def blend_coeff(t: float, Tr: float):
	"""
	Coeficientes suaves de blending entre 0 y 1 usando función trigonométrica.

	Args:
		t (float): Tiempo actual (s).
		Tr (float): Duración del blending (s).

	Returns:
		s (float): Coeficiente de blending (0..1).
		s_dot (float): Primera derivada de s respecto al tiempo.
		s_ddot (float): Segunda derivada de s respecto al tiempo.

	Example:
		>>> s, sd, sdd = blend_coeff(1.0, 5.0)
	"""
	if t <= 0: return 0.0,0.0,0.0
	if t >= Tr: return 1.0,0.0,0.0

	# Hay dos opciones comunes para blending suave:
	# Descomentar según preferencia

	# Opciones trigonométricas (suave pero no continua en t=0,Tr)

	#s = 0.5*(1 - np.cos(np.pi*t/Tr))
	#s_dot = 0.5*(np.pi/Tr)*np.sin(np.pi*t/Tr)
	#s_ddot = 0.5*(np.pi/Tr)**2 * np.cos(np.pi*t/Tr)
	
	# Opciones polinómicas ontinua (suave y continua en t=0,Tr)
	tau = t / Tr
	s = 10*tau**3 - 15*tau**4 + 6*tau**5
	s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / Tr
	s_ddot = (60*tau - 180*tau**2 + 120*tau**3) / (Tr**2)
	return s, s_dot, s_ddot

def blended_trajectory(t: float, traj_func, Tr: float = 5.0, p_init=None, v_init=None, **traj_params):
	"""
	Genera trayectoria blendada desde (p_init, v_init) hacia la trayectoria objetivo.

	Se usa para hacer transición suave al inicio evitando saltos en referencia. Esta función es útil para evitar saltos y oscilaciones indeseadas al iniciar la simulación desde una posición y velocidad iniciales distintas a las de la trayectoria deseada.

	Args:
		t (float): Tiempo actual (s).
		traj_func (callable): Función de trayectoria que devuelve (p_ref, v_ref, a_ref).
		Tr (float): Duración del blending (s).
		p_init (np.ndarray|None): Posición inicial [x,y,z,psi]. Si None, se usa cero.
		v_init (np.ndarray|None): Velocidad inicial [vx,vy,vz,wz]. Si None, se usa cero.
		**traj_params: Parámetros adicionales para traj_func.

	Returns:
		p_ref (np.ndarray): Posición blendada [x,y,z,psi].
		v_ref (np.ndarray): Velocidad blendada [vx,vy,vz,wz].
		a_ref (np.ndarray): Aceleración blendada [ax,ay,az,az].

	Example:
		>>> p_ref, v_ref, a_ref = blended_trajectory(t, trajectory_circle, Tr=3.0, R=2.0)
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

def dynamics(state: np.ndarray, Tprime: float, Nx: float, Ny: float, Nz: float,
			A: float, B: float, C: float, D: float, g: float, gamma: float, epsilon: float) -> np.ndarray:
	"""
	Dinámica linealizada del quadcopter alrededor del hover (12 estados).
	Las ecuaciones están escritas en forma de un sistema de ecuaciones de primer orden, además, las ecuaciones traslaciones están escritas en el marco inercial, mientras que las ecuaciones de rotación (ecuaciones de derivada de omega) están escritas en el marco del cuerpo y se relacionan con los ángulos de Tait-Bryan (marco inercial) mediante su respectiva matriz.

	Notación de estados:
		state = [ x, vx, y, vy, z, vz, phi, wx, theta, wy, psi, wz ]
	donde wx, wy, wz son las velocidades angulares en el marco del cuerpo.

	Ecuaciones linealizadas usadas:
		dx/dt   = vx
		dvx/dt  = -B*vx + g * theta
		dy/dt   = vy
		dvy/dt  = -C*vy - g * phi
		dz/dt   = vz
		dvz/dt  = A * Tprime - D * vz
		dphi/dt = wx
		dwx/dt  = gamma * Nx        # gamma = 1/Ixx = 1/Iyy
		dtheta/dt = wy
		dwy/dt  = gamma * Ny
		dpsi/dt = wz
		dwz/dt  = epsilon * Nz      # epsilon = 1/Izz

	Args:
		state (np.ndarray): Estado actual (12,).
		Tprime (float): Thrust incremental (T' = T - m*g) en newton.
		Nx, Ny, Nz (float): Torques aplicados en roll, pitch, yaw (ejes x,y,z) respectivamente.
		A, B, C, D (float): Coeficientes lineales de la dinámica (definición propia del modelo).
		g (float): Gravedad (positivo, p.ej. 9.81).
		gamma (float): 1/Ixx (y = Iyy) — inverso de la inercia en roll/pitch.
		epsilon (float): 1/Izz — inverso de la inercia en yaw.

	Returns:
		np.ndarray: Vector derivadas del estado (12,).

	Example:
		>>> dxdt = dynamics(state, Tprime, Nx, Ny, Nz, A,B,C,D,9.81,gamma,epsilon)
	"""
	x, vx, y, vy, z, vz, phi, wx, theta, wy, psi, wz = state # Vector de estado

	# Ecuaciones de la dinámica linealizada
	# Ecuaciones de traslación
	dxdt = vx; dvxdt = -B*vx + g*theta
	dydt = vy; dvydt = -C*vy - g*phi
	dzdt = vz; dvzdt = A*Tprime - D*vz
	# Ecuaciones de rotación
	dphidt = wx; dwxdt = Nx*gamma
	dthetadt = wy; dwydt = Ny*gamma
	dpsidt = wz; dwzdt = Nz*epsilon

	# Vector de estado derivado
	return np.array([dxdt, dvxdt, dydt, dvydt, dzdt, dvzdt,
					dphidt, dwxdt, dthetadt, dwydt, dpsidt, dwzdt])

# ===============================
# Control externo e interno
# ===============================

def outer_controller(state: np.ndarray, p_ref: np.ndarray, v_ref: np.ndarray, a_ref: np.ndarray, params: dict):
	"""
	Outer loop: controlador de posición.

	Función de control de la trayectoria que calcula las *referencias* que usará
	el inner loop y el input vertical Tprime.

	Flujo:
		1) Calcula errores de posición (x,y,z).
		2) Usa pid_x/pid_y/pid_z para obtener aceleraciones deseadas ax_des, ay_des, az_des.
		3) Convierte ax_des, ay_des en ángulos deseados (theta_des, phi_des) usando aproximación small-angle:
			theta_des = (ax_des + B*vx)/g
			phi_des   = -(ay_des + C*vy)/g
		4) Convierte az_des en Tprime via:
			Tprime = (az_des + D*vz) / A
	5) Devuelve (Tprime, phi_des, theta_des, psi_des).

	Args:
		state (np.ndarray): Estado actual (12,).
		p_ref (np.ndarray): Posición de referencia [x,y,z,psi].
		v_ref (np.ndarray): Velocidad de referencia (no usada directamente).
		a_ref (np.ndarray): Aceleración de referencia (feedforward).
		params (dict): Parámetros y PIDs. Debe contener:
			- 'pid_x', 'pid_y', 'pid_z' (instancias PID)
			- 'g','A','B','C','D'
			- 'dt', 'max_angle', 'Tprime_min', 'Tprime_max' (opcionales según uso)

	Returns:
		Tprime (float): Thrust incremental (entrada física vertical).
		phi_des (float): Ángulo roll deseado (rad).
		theta_des (float): Ángulo pitch deseado (rad).
		psi_des (float): Referencia de yaw (rad), tomada de p_ref[3].

	Example:
		>>> Tprime, phi_d, theta_d, psi_d = outer_controller(state, p_ref, v_ref, a_ref, params)
	"""
	x, vx, y, vy, z, vz = state[0], state[1], state[2], state[3], state[4], state[5]

	pid_x = params["pid_x"]
	pid_y = params["pid_y"]
	pid_z = params["pid_z"]

	dt = params["dt"]

	# Errores de posición
	ex = p_ref[0] - x
	ey = p_ref[1] - y
	ez = p_ref[2] - z

	# Aceleraciones deseadas usando PIDs (feedforward + feedback)
	ax_des = a_ref[0] + pid_x(ex, dt)
	ay_des = a_ref[1] + pid_y(ey, dt)
	az_des = a_ref[2] + pid_z(ez, dt)

	# Parámetros de la dinámica
	g = params["g"]
	A = params["A"]
	B = params["B"]
	C = params["C"]
	D = params["D"]

	# Conversión a ángulos (small-angle)
	theta_des = (ax_des + B * vx) / g              # pitch
	phi_des   = -(ay_des + C * vy) / g             # roll

	# Empuje linealizado (input físico)
	min_Tprime = params.get('Tprime_min',-np.inf)
	max_Tprime =  params.get('Tprime_max',np.inf)

	Tprime    = (az_des + D * vz) / A
	Tprime = float(np.clip(Tprime, min_Tprime,max_Tprime))

	# psi_des proviene de la referencia
	psi_des = p_ref[3]

	return Tprime, phi_des, theta_des, psi_des

def inner_controller(state: np.ndarray, phi_des: float, theta_des: float, psi_des: float, params: dict):
	"""
	Inner loop: controlador de actitud que calcula torques.

	Implementación:
		- Calcula errores en ángulos (phi, theta, psi).
		- Utiliza PIDs en espacio angular para obtener aceleraciones angulares deseadas:
			alpha = PID_angle(error, dt)
		- Convierte aceleraciones angulares deseadas alpha a torque físico tau usando
			las constantes gamma (1/Ixx) y epsilon (1/Izz):
				tau_phi   = alpha_phi   / gamma
				tau_theta = alpha_theta / gamma
				tau_psi   = alpha_psi   / epsilon

	Args:
		state (np.ndarray): Estado actual (12,).
		phi_des (float): Roll deseado (rad).
		theta_des (float): Pitch deseado (rad).
		psi_des (float): Yaw deseado (rad).
		params (dict): Parámetros que deben contener:
			- 'pid_phi', 'pid_theta', 'pid_psi' (instancias PID)
			- 'gamma' (1/Ixx), 'epsilon' (1/Izz)
			- 'dt', y límites opcionales 'max_tau', 'max_tau_psi'

	Returns:
		tau_phi (float): Torque físico en roll (Nm).
		tau_theta (float): Torque físico en pitch (Nm).
		tau_psi (float): Torque físico en yaw (Nm).

	Example:
		>>> tau_phi, tau_theta, tau_psi = inner_controller(state, phi_d, theta_d, psi_d, params)
	"""
	phi, theta, psi = state[6], state[8], state[10]
	wx, wy, wz = state[7], state[9], state[11]

	dt = params["dt"]

	e_phi   = phi_des - phi
	e_theta = theta_des - theta
	e_psi   = psi_des - psi

	# PID en espacio angular devuelve aceleración angular deseada (rad/s^2)
	alpha_phi   = params["pid_phi"](e_phi, dt)
	alpha_theta = params["pid_theta"](e_theta, dt)
	alpha_psi   = params["pid_psi"](e_psi, dt)

	# Convertir aceleración angular -> torque físico
	tau_phi   = alpha_phi   / params["gamma"]
	tau_theta = alpha_theta / params["gamma"]
	tau_psi   = alpha_psi   / params["epsilon"]

	# Saturación opcional de torques físicos
	max_tau = params.get("max_tau", np.inf)
	max_tau_psi = params.get("max_tau_psi", np.inf)
	tau_phi   = float(np.clip(tau_phi,   -max_tau, max_tau))
	tau_theta = float(np.clip(tau_theta, -max_tau, max_tau))
	tau_psi   = float(np.clip(tau_psi,   -max_tau_psi, max_tau_psi))

	return tau_phi, tau_theta, tau_psi

# ===============================
# Simulación RK4
# ===============================

def simulate(T: float, dt: float, dynamics_func, trajectory_func, outer_controller, inner_controller, state0: np.ndarray, params: dict):
	"""
	Simulador de dron usando integración RK4 y control en lazo cascada.

	Arquitectura:
		1) trajectory_func(t) -> p_ref, v_ref, a_ref
		2) outer_controller(state, p_ref, v_ref, a_ref, params) -> (Tprime, phi_des, theta_des, psi_des)
		3) inner_controller(state, phi_des, theta_des, psi_des, params) -> (Nx, Ny, Nz)
		4) dynamics(state, Tprime, Nx, Ny, Nz, ...) -> state_dot
		5) Integración RK4 con dt

	Args:
		T (float): Tiempo total de simulación (s).
		dt (float): Paso temporal (s).
		dynamics_func (callable): Función de dinámica con firma dynamics(state, Tprime, Nx, Ny, Nz, A,B,C,D,g,gamma,epsilon).
		trajectory_func (callable): Función de trayectoria con firma (p_ref, v_ref, a_ref) = trajectory_func(t, **traj_params).
		outer_controller (callable): Función outer_controller(state, p_ref, v_ref, a_ref, params).
		inner_controller (callable): Función inner_controller(state, phi_des, theta_des, psi_des, params).
		state0 (np.ndarray): Estado inicial (12,).
		params (dict): Diccionario con parámetros (A,B,C,D,g,gamma,epsilon,PIDs, límites, etc.).

	Returns:
		tvec (np.ndarray): Vector de tiempos (n,).
		hist (np.ndarray): Historial de estados (n,12).
		ref_hist (np.ndarray): Historial de referencias p_ref (n,4).
		u_hist (np.ndarray): Historial de inputs físicos [Tprime, Nx, Ny, Nz] (n,4).
		Tprime_hist (np.ndarray): Historial de Tprime (n,).
		params (dict): Diccionario con parámetros usados en la simulación.

	Example:
		>>> tvec, hist, ref_hist, u_hist, Tprime_hist, params = simulate(10.0, 0.01, dynamics, blended_trajectory, outer_controller, inner_controller, state0, params)
	"""
	n = int(T/dt) + 1
	tvec = np.linspace(0, T, n)

	# Historiales
	hist = np.zeros((n, 12))
	ref_hist = np.zeros((n, 4))
	u_hist = np.zeros((n, 4))        # [Tprime, Nx, Ny, Nz]
	Tprime_hist = np.zeros(n)

	# Estado inicial
	state = np.array(state0, dtype=float)

	# Asegurar dt en params
	params["dt"] = dt

	# Resetear PIDs de posición
	for k in ["pid_x", "pid_y", "pid_z"]:
		if k in params and hasattr(params[k], "reset"):
			params[k].reset()
	# Resetear PIDs de actitud si existen
	for k in ["pid_phi", "pid_theta", "pid_psi"]:
		if k in params and hasattr(params[k], "reset"):
			params[k].reset()

	# Loop de simulación
	for i, t in enumerate(tvec):

		# 1) Trayectoria deseada
		p_ref, v_ref, a_ref = trajectory_func(t, **params.get("traj_params", {}))
		ref_hist[i, :] = p_ref

		# 2) Outer controller -> produce referencias y Tprime
		Tprime_cmd, phi_des, theta_des, psi_des = outer_controller(state, p_ref, v_ref, a_ref, params)

		# Aplicar límites a referencias y Tprime
		phi_des = float(np.clip(phi_des, -params.get("max_angle", np.inf), params.get("max_angle", np.inf)))
		theta_des = float(np.clip(theta_des, -params.get("max_angle", np.inf), params.get("max_angle", np.inf)))
		Tprime_cmd = float(np.clip(Tprime_cmd, params.get("Tprime_min", -np.inf), params.get("Tprime_max", np.inf)))

		# 3) Inner controller -> torques físicos
		Nx, Ny, Nz = inner_controller(state, phi_des, theta_des, psi_des, params)

		# Opcional: límites adicionales sobre torques
		Nx = float(np.clip(Nx, -params.get("max_tau", np.inf), params.get("max_tau", np.inf)))
		Ny = float(np.clip(Ny, -params.get("max_tau", np.inf), params.get("max_tau", np.inf)))
		Nz = float(np.clip(Nz, -params.get("max_tau_psi", np.inf), params.get("max_tau_psi", np.inf)))

		# Guardar inputs físicos
		u_hist[i, :] = [Tprime_cmd, Nx, Ny, Nz]
		Tprime_hist[i] = Tprime_cmd

		# 4) Integración RK4 con inputs físicos
		k1 = dynamics_func(state, Tprime_cmd, Nx, Ny, Nz,
							params["A"], params["B"], params["C"], params["D"],
							params["g"], params["gamma"], params["epsilon"])
		k2 = dynamics_func(state + 0.5*dt*k1, Tprime_cmd, Nx, Ny, Nz,
							params["A"], params["B"], params["C"], params["D"],
							params["g"], params["gamma"], params["epsilon"])
		k3 = dynamics_func(state + 0.5*dt*k2, Tprime_cmd, Nx, Ny, Nz,
							params["A"], params["B"], params["C"], params["D"],
							params["g"], params["gamma"], params["epsilon"])
		k4 = dynamics_func(state + dt*k3, Tprime_cmd, Nx, Ny, Nz,
							params["A"], params["B"], params["C"], params["D"],
							params["g"], params["gamma"], params["epsilon"])

		state += (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

		# Ligerísimo damping numérico para estabilidad de rates
		state[7] *= 0.999
		state[9] *= 0.999
		state[11] *= 0.999

		# Guardar estado
		hist[i, :] = state

	return tvec, hist, ref_hist, u_hist, Tprime_hist, params
