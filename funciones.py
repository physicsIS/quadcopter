import sympy as smp
import numpy as np
import control as ctl


def convert_sympy_tf_matrix(
		G_sym,
		subs_dict={},
		s_symbol=None,
		return_type="tf",
		as_ndarray=False,
		verbose=True
	):
	"""
	Convierte una matriz de funciones de transferencia simbólicas a objetos de python-control.

	Convierte una matriz simbólica cuyas entradas son funciones racionales en la variable compleja s 
	(típicamente generadas con SymPy) en sus representaciones equivalentes dentro de la librería 
	python-control. Antes de la conversión, permite sustituir valores numéricos para parámetros 
	simbólicos. La función puede retornar objetos del tipo TransferFunction o StateSpace según se solicite.

	Args:
		G_sym (sympy.Matrix): 
			Matriz simbólica donde cada entrada es una expresión racional en s.

		subs_dict (dict): 
			Diccionario con sustituciones de la forma {símbolo: valor} que se aplicarán antes 
			de realizar la conversión.

		s_symbol (sympy.Symbol, optional): 
			Símbolo utilizado como variable compleja s. Si no se especifica, la función intenta 
			detectar automáticamente un símbolo llamado 's' o 'S'.

		return_type (str): 
			Tipo de objeto a retornar.
			- "tf": retorna objetos TransferFunction.
			- "ss": retorna objetos StateSpace.

		as_ndarray (bool): 
			Si es True, la salida se retorna como un arreglo de NumPy.
			Si es False, la salida es una lista de listas.

		verbose (bool): 
			Si es True, imprime información detallada del proceso de conversión.

	Returns:
		matriz (numpy.ndarray or list[list]): 
			Matriz que contiene los objetos TransferFunction o StateSpace, según el tipo solicitado.

	Example:
		>>> from sympy import symbols, Matrix
		>>> s, k = symbols('s k')
		>>> G = Matrix([[k/(s+1)]])
		>>> out = convert_transfer_matrix(G, {k: 2.0}, s_symbol=s, return_type="tf")
	"""


	# -----------------------------------------
	# 1. Detectar el símbolo s automáticamente
	# -----------------------------------------
	if s_symbol is None:
		candidates = [sym for sym in G_sym.free_symbols if sym.name.lower() == 's']
		if len(candidates) == 0:
			raise ValueError("No pude detectar el símbolo 's'. Pásalo explícitamente.")
		s_symbol = candidates[0]
		if verbose:
			print(f"[INFO] Detectado símbolo de Laplace: {s_symbol}")

	rows, cols = G_sym.shape

	# Crear contenedor
	if as_ndarray:
		G_out = np.empty((rows, cols), dtype=object)
	else:
		G_out = [[None for _ in range(cols)] for _ in range(rows)]

	# -----------------------------------------
	# 2. Recorrer todas las entradas simbólicas
	# -----------------------------------------
	for i in range(rows):
		for j in range(cols):

			expr = smp.simplify(G_sym[i, j].subs(subs_dict))

			# Caso: entrada exactamente cero
			if expr == 0:
				tf = ctl.TransferFunction([0.0], [1.0])
				G_out[i][j] = tf
				continue

			# Caso: la expresión es un número (5, -3.2, etc.)
			if expr.is_number:
				tf = ctl.TransferFunction([float(expr)], [1.0])
				G_out[i][j] = tf
				continue

			# Verificar que es racional en s
			if not expr.has(s_symbol):
				raise ValueError(f"La entrada G[{i},{j}] no depende de s.")

			# Separar numerador y denominador simbólicos
			num_sym, den_sym = smp.fraction(expr)

			# Convertir a polinomios en s
			try:
				num_poly = smp.Poly(num_sym, s_symbol)
				den_poly = smp.Poly(den_sym, s_symbol)
			except smp.polys.polyerrors.PolynomialError:
				raise ValueError(
					f"La expresión G[{i},{j}] = {expr} no es racional en s."
				)

			# Obtener coeficientes numéricos
			num_coeffs = [float(c) for c in num_poly.all_coeffs()]
			den_coeffs = [float(c) for c in den_poly.all_coeffs()]

			# Construir TF
			tf = ctl.TransferFunction(num_coeffs, den_coeffs)

			# Convertir a StateSpace si se pidió
			if return_type == "ss":
				tf = ctl.ss(tf)

			# Guardar
			G_out[i][j] = tf

	if verbose:
		print("\n[INFO] Conversión completada. Matriz obtenida:")
		for row in G_out:
			print(row)

	return G_out