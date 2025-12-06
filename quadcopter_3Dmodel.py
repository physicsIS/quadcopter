from vpython import *
import numpy as np

class Quadcopter:
	"""Clase Quadcopter 3D para VPython.

	Características principales:
	- Modelo 3D completo (cuerpo, brazos, motores, hélices)
	- Compound para agrupar la geometría
	- Sistema interno de orientación basado en una matriz de rotación 3x3 (self.R)
	- Métodos para set_position, set_orientation (matriz), set_euler y set_quaternion
	- Textos 3D (labels) actualizados cada vez que cambia la pose
	- spin_props(speeds, dt) para animar hélices

	Notas:
	- Esta clase solo gestiona visualización. No implementa dinámica ni control.
	"""

	def __init__(self,
				arm_length=0.4,
				arm_radius=0.015,
				body_radius=0.12,
				body_height=0.05,
				prop_radius=0.12,
				prop_thickness=0.015,
				motor_height=0.05):

		# Parámetros geométricos
		self.arm_length = arm_length
		self.motor_height = motor_height
		self.prop_thickness = prop_thickness
		self.axis_length = 0.6

		# Lista temporal de partes para crear el compound
		parts = []

		# -----------------
		# Cuerpo
		# -----------------
		body = cylinder(pos=vector(0, -body_height/2, 0),
						axis=vector(0, body_height, 0),
						radius=body_radius,
						color=color.gray(0.6))
		parts.append(body)

		# -----------------
		# Brazos (convención: +x frontal, +y izquierda, -x trasero, -y derecha)
		# -----------------
		arm1 = cylinder(pos=vector(0,0,0), axis=vector(arm_length,0,0), radius=arm_radius, color=color.gray(0.3))
		arm2 = cylinder(pos=vector(0,0,0), axis=vector(0,0,arm_length), radius=arm_radius, color=color.gray(0.3))
		arm3 = cylinder(pos=vector(0,0,0), axis=vector(-arm_length,0,0), radius=arm_radius, color=color.gray(0.3))
		arm4 = cylinder(pos=vector(0,0,0), axis=vector(0,0,-arm_length), radius=arm_radius, color=color.gray(0.3))
		parts += [arm1, arm2, arm3, arm4]

		# -----------------
		# Motores y hélices
		# -----------------
		motor_positions = [a.pos + a.axis for a in (arm1, arm2, arm3, arm4)]

		props = []
		motors = []

		for i, pos in enumerate(motor_positions):
			motor = cylinder(pos=pos,
							axis=vector(0, motor_height, 0),
							radius=0.04,
							color=color.gray(0.5))

			prop = cylinder(pos=pos + vector(0, motor_height + prop_thickness/2, 0),
							axis=vector(0, prop_thickness, 0),
							radius=prop_radius,
							color=color.blue if i % 2 == 0 else color.red)

			parts.append(motor)
			parts.append(prop)
			motors.append(motor)
			props.append(prop)

		# -----------------
		# Ejes corporales (como arrows locales)
		# -----------------
		xB = arrow(pos=vector(0,0,0), axis=vector(self.axis_length,0,0), color=color.red, shaftwidth=0.02)
		yB = arrow(pos=vector(0,0,0), axis=vector(0,0,self.axis_length), color=color.green, shaftwidth=0.02)
		zB = arrow(pos=vector(0,0,0), axis=vector(0,self.axis_length,0), color=color.blue, shaftwidth=0.02)
		parts += [xB, yB, zB]

		# -----------------
		# Textos 3D (labels). Usamos text() para que puedan ser posicionados en 3D
		# No los ponemos como label() ya que label no rota con la escena.
		# billboard=True hace que el texto siempre mire a la cámara (opcional).
		# -----------------
		txt_x = text(text='x_B', pos=xB.axis * 1.05, height=0.08, color=color.red, billboard=True)
		txt_y = text(text='y_B', pos=yB.axis * 1.05, height=0.08, color=color.green, billboard=True)
		txt_z = text(text='z_B', pos=zB.axis * 1.05, height=0.08, color=color.blue, billboard=True)
		parts += [txt_x, txt_y, txt_z]

		# -----------------
		# Compound: agrupa todas las partes en un solo objeto visual
		# -----------------
		self._parts = parts  # guardamos la lista por si hace falta
		self.compound = compound(parts, pos=vector(0,0,0), origin = vector(0,0,0))

		# Guardar referencias a hélices (estos objetos siguen existiendo aunque formen parte del compound)
		self.props = props
		self.motors = motors

		# Guardar referencias a los ejes y textos (acceso a través del compound requiere desplazamiento relativo)
		# Para actualizaciones, guardamos vectores base en coordenadas del cuerpo
		self._base_x = np.array([1.0, 0.0, 0.0])
		self._base_y = np.array([0.0, 0.0, 1.0])
		self._base_z = np.array([0.0, 1.0, 0.0])

		# Mantener una matriz de rotación interna (3x3). Identidad al inicio.
		self.R = np.eye(3)

		# Guardar referencias a objetos de ejes/textos para actualizar sus posiciones cada frame
		# Buscamos los objetos por tipo/posición dentro del compound: mejor asignarlos antes.
		# Como ya tenemos los objetos txt_x, etc., los referenciamos por índice relativo en self._parts
		# (últimos 6 elementos añadidos fueron xB, yB, zB, txt_x, txt_y, txt_z)
		self._xB_obj = xB
		self._yB_obj = yB
		self._zB_obj = zB
		self._txt_x = txt_x
		self._txt_y = txt_y
		self._txt_z = txt_z

		# Inicial pose (posicion zero y orientacion identidad)
		self.set_position([0.0, 0.0, 0.0])
		self.set_orientation_matrix(self.R)

	# -----------------
	# Pose: posición
	# -----------------
	def set_position(self, pos):
		"""pos: lista/array [x,y,z] o vector VPython"""
		if isinstance(pos, vector):
			self.compound.pos = pos
		else:
			self.compound.pos = vector(pos[0], pos[1], pos[2])
		# actualizar labels
		self._update_axis_and_labels()

	# -----------------
	# Pose: orientación mediante matriz de rotación 3x3
	# -----------------
	def set_orientation_matrix(self, R):
		"""R: numpy array 3x3 (convención: transforma vectores en coordenadas de cuerpo -> mundo)

		En vez de aplicar rotaciones incrementales, aquí asignamos la orientación absoluta
		modificando los vectores axis y up del compound de acuerdo a R. Esto evita errores por acumulación.
		"""
		R = np.array(R)
		if R.shape != (3,3):
			raise ValueError('R debe ser una matriz 3x3')

		self.R = R.copy()

		# Definir forward (eje x del cuerpo) y up (eje z del cuerpo) en coordenadas mundo
		forward = R @ np.array([1.0, 0.0, 0.0])
		up_vec  = R @ np.array([0.0, 1.0, 0.0])

		# Asignar a compound de forma absoluta
		self.compound.axis = vector(forward[0], forward[1], forward[2])
		self.compound.up   = vector(up_vec[0], up_vec[1], up_vec[2])

		# Actualizar ejes/textos que dependen de la orientación
		self._update_axis_and_labels()

	# -----------------
	# Metodos de conveniencia: Euler (ZYX), Tait-Bryan (YXZ) y cuaterniones
	# -----------------
	def set_euler(self, roll, pitch, yaw):
		"""Roll, pitch, yaw (rad). Convención ZYX: R = Rz(yaw) * Ry(pitch) * Rx(roll)"""
		cr, sr = np.cos(roll), np.sin(roll)
		cp, sp = np.cos(pitch), np.sin(pitch)
		cy, sy = np.cos(yaw), np.sin(yaw)

		Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
		Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
		Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

		R = Rz @ Ry @ Rx
		self.set_orientation_matrix(R)


	def set_taitbryan_yxz(self, phi, theta, psi):
		"""
		Roll (phi), Pitch (theta), Yaw (psi) en radianes.
		Convención Tait-Bryan Y-X-Z: R = Rx(phi) @ Ry(theta) @ Rz(psi)
		Pasa de coordenadas inerciales a no inerciales.
		"""
		cphi, sphi   = np.cos(phi), np.sin(phi)
		ctheta, stheta = np.cos(theta), np.sin(theta)
		cpsi, spsi   = np.cos(psi), np.sin(psi)

		# Matrices elementales
		Rx = np.array([
			[1.0, 0.0, 0.0],
			[0.0, cphi, sphi],
			[0.0, -sphi, cphi]
		])

		Ry = np.array([
			[ctheta, 0.0, -stheta],
			[0.0, 1.0, 0.0],
			[stheta, 0.0, ctheta]
		])

		Rz = np.array([
			[cpsi, spsi, 0.0],
			[-spsi, cpsi, 0.0],
			[0.0, 0.0, 1.0]
		])

		# Rotación compuesta
		R = Rx @ Ry @ Rz
		self.set_orientation_matrix(R)


	def set_quaternion(self, q):
		"""q = [w, x, y, z] o array-like. Convierte quaternion a matriz y llama a set_orientation_matrix"""
		q = np.array(q, dtype=float)
		if q.size != 4:
			raise ValueError('Quaternion debe tener 4 componentes [w,x,y,z]')
		w, x, y, z = q
		# Normalizar
		n = np.linalg.norm(q)
		if n == 0:
			raise ValueError('Quaternion de norma cero')
		w, x, y, z = q / n
		# Matriz de rotación (WXYZ)
		R = np.array([
			[1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
			[2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
			[2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
		])
		self.set_orientation_matrix(R)

	# -----------------
	# Actualizar posición de ejes (arrows) y labels (text)
	# -----------------
	def _update_axis_and_labels(self):
		"""Calcula las posiciones actuales de los ejes corporales y coloca los textos en consecuencia."""
		# Obtener ejes en coordenadas mundo aplicando R a los vectores base
		rx = self.R @ self._base_x
		ry = self.R @ self._base_y
		rz = self.R @ self._base_z

		# Update arrow objects: sus posiciones deben ser relativas a self.compound.pos
		# las arrows se crearon en coordenadas locales iniciales, pero como están dentro del compound,
		# podemos mover sus attributes axis y pos relativos al compound.pos
		self._xB_obj.pos = self.compound.pos
		self._xB_obj.axis = vector(rx[0]*self.axis_length, rx[1]*self.axis_length, rx[2]*self.axis_length)

		self._yB_obj.pos = self.compound.pos
		self._yB_obj.axis = vector(ry[0]*self.axis_length, ry[1]*self.axis_length, ry[2]*self.axis_length)

		self._zB_obj.pos = self.compound.pos
		self._zB_obj.axis = vector(rz[0]*self.axis_length, rz[1]*self.axis_length, rz[2]*self.axis_length)

		# Actualizar textos en la punta de cada eje (un poco más allá del axis)
		self._txt_x.pos = self.compound.pos + vector(rx[0]*(self.axis_length*1.05), rx[1]*(self.axis_length*1.05), rx[2]*(self.axis_length*1.05))
		self._txt_y.pos = self.compound.pos + vector(ry[0]*(self.axis_length*1.05), ry[1]*(self.axis_length*1.05), ry[2]*(self.axis_length*1.05))
		self._txt_z.pos = self.compound.pos + vector(rz[0]*(self.axis_length*1.05), rz[1]*(self.axis_length*1.05), rz[2]*(self.axis_length*1.05))

	# -----------------
	# Hélices
	# -----------------
	def spin_props(self, speeds, dt):
		"""Gira cada hélice.

		speeds: iterable con 4 velocidades angulares en rad/s (o cualquier escala que uses)
		dt: paso temporal (s)
		"""
		for i, p in enumerate(self.props):
			direction = 1 if i % 2 == 0 else -1
			# Como las hélices están dentro del compound, su posición ya está en coordenadas mundo
			p.rotate(angle=direction * speeds[i] * dt, axis=self.compound.up, origin=self.compound.pos)

	# -----------------
	# Utilidades
	# -----------------
	def get_pose(self):
		"""Devuelve (pos, R)"""
		pos = np.array([self.compound.pos.x, self.compound.pos.y, self.compound.pos.z])
		return pos, self.R.copy()
