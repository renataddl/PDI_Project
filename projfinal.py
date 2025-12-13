# https://www.teses.usp.br/teses/disponiveis/18/18152/tde-15102008-135808/publico/Ricardo.pdf
# https://github.com/renataddl/PDI_Project
import numpy as np
import cv2

# PIPELINE
# Pre-processamento:
	# Equalizacao
	# Aprimoramento fft
	# fFiltros suavizacao
	# Binarizacao
	# Imagem direcional
# Extracao de minucias:
	# Regiao interesse
	# Afinamento
	# Filtros morfologicos
	# Extracao de minucias
# Pos-processamento:
	# Remocao de minucias

### Filtros morfologicos ########################

def filtroMorf(binary_image):
	# Variações dos kernels
	filter_kernels = [
		# clean
		np.array([
			[0, 0, 0],
			[0, 1, 0],
			[0, 0, 0]
		], dtype="int"),
		
		#hbreak vertical
		np.array([
			[1, 0, 1],
			[1, 1, 1],
			[1, 0, 1]
		], dtype="int"),
		
		#hbreak horizontal
		np.array([
			[1, 1, 1],
			[0, 1, 0],
			[1, 1, 1]
		], dtype="int"),
	]

	''' 4 rotações de Spur em 2 variações, vão ser mais 8 filtros... talvez pular este se já estiver muito pesado?
	também tem que ver a lógica de substituição, que é diferente porque é tamanho par, não elimina o pixel central
	como os outros:'''

	# spur1 = np.array([
	# 	[0, 0, 0, 0],
	# 	[0, 0, 1, 0],
	# 	[0, 1, 0, 0],
	# 	[1, 1, 0, 0]
	# ], dtype="int"),

	# spur2 = np.array([
	# 	[0, 0, 0, 0],
	# 	[0, 0, 1, 0],
	# 	[1, 1, 0, 0],
	# 	[1, 0, 0, 0]
	# ], dtype="int"),

	# Adaptando os kernels spur para o anchor point ficar no centro
	# já com a transformação de 0 em -1 no background, pois o novo
	# padding deve permanecer 0.
	spur1 = np.array([
		[ 0,  0,  0,  0,  0],
		[-1, -1, -1, -1,  0],
		[-1, -1,  1, -1,  0],
		[-1,  1, -1, -1,  0],
		[ 1,  1, -1, -1,  0],
	], dtype="int")

	spur2 = np.array([
		[ 0,  0,  0,  0,  0],
		[-1, -1, -1, -1,  0],
		[-1, -1,  1, -1,  0],
		[ 1,  1, -1, -1,  0],
		[ 1, -1, -1, -1,  0],
	], dtype="int")

	spur_filter_kernels = []

	for _ in range(4):
		spur1 = np.rot90(spur1)
		spur_filter_kernels.append(spur1)

	for _ in range(4):
		spur2 = np.rot90(spur2)
		spur_filter_kernels.append(spur2)

	# Conferir os kernels:
	print('filter_kernels:')
	for kernel in filter_kernels:
		print(kernel)
		print()

	print('spur_filter_kernels:')
	for kernel in spur_filter_kernels:
		print(kernel)
		print()

	result_image = binary_image.copy()

	print(binary_image.shape, result_image.shape) #debug
	for kernel in filter_kernels:
		kernel = np.where(kernel == 0, -1, kernel) # -1 indica background, 1 foreground, 0 tanto faz (por isso precisa evitar zeros)
		match_map = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel, borderType=cv2.BORDER_REPLICATE) # replicar a borda para restringir kernel dentro da imagem
		result_image[match_map > 0] = 0

		cv2.imshow(f'{kernel} matches', match_map)
		cv2.imshow(f'{kernel} result', result_image)

	print(binary_image.shape, result_image.shape) #debug
	for kernel in spur_filter_kernels:
		match_map = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel, borderType=cv2.BORDER_REPLICATE)
		result_image[match_map > 0] = 0

		cv2.imshow(f'{kernel} matches', match_map)
		cv2.imshow(f'{kernel} result', result_image)	

	cv2.imshow('Testcase Visualization', binary_image)
	cv2.imshow('Filter', result_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return result_image

def testarFiltroMorf():
	testcase = [
		[0, 0, 0, 1, 0, 0, 1],
		[0, 1, 0, 1, 1, 0, 0],
		[0, 0, 1, 0, 0, 1, 1],
		[0, 1, 1, 0, 1, 0, 0],
		[0, 1, 0, 1, 0, 1, 1],
		[0, 1, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 1],
		[1, 1, 1, 0, 1, 1, 1],
		[0, 1, 0, 0, 1, 1, 1],
		[1, 1, 1, 0, 1, 1, 1],
		[0, 0, 0, 0, 0, 1, 0],
		[1, 0, 1, 0, 0, 0, 0],
		[1, 1, 1, 1, 0, 1, 0],
		[1, 0, 1, 0, 0, 0, 0],
	]
	
	test_array = np.array(testcase, dtype=np.uint8)
	binary_image = test_array * 255

	return filtroMorf(binary_image)


### Extracao de minucias ########################

def extrairMinucias(binary_image):
	# Variações dos kernels
	bifurcation_kernel_types = [
		np.array([
			[1, 0, 1],
			[0, 1, 0],
			[0, 1, 0]
		], dtype="int"),

		np.array([
			[0, 1, 0],
			[1, 1, 0],
			[0, 0, 1]
		], dtype="int"),
	]
	
	termination_kernel_types = [
		np.array([
			[0, 1, 0],
			[0, 1, 0],
			[0, 0, 0]
		], dtype="int"),

		np.array([
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 0]
		], dtype="int"),
	]

	# Computar as rotações de cada variante
	bifurcation_kernels = []
	for kernel in bifurcation_kernel_types:
		for _ in range(4):
			kernel = np.rot90(kernel)
			bifurcation_kernels.append(kernel)

	termination_kernels = []
	for kernel in termination_kernel_types:
		for _ in range(4):
			kernel = np.rot90(kernel)
			termination_kernels.append(kernel)

	# Conferir os kernels
	print('bifurcation_kernels:')
	for kernel in bifurcation_kernels:
		print(kernel)
		print()

	print('termination_kernels:')
	for kernel in termination_kernels:
		print(kernel)
		print()

	# Preparar saidas
	color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
	result_image = np.zeros_like(binary_image)

	# Varrer a imagem com cada kernel
	for kernel in bifurcation_kernels:
		hit_miss_result = np.zeros_like(binary_image)
		kernel = np.where(kernel == 0, -1, kernel) # -1 indica background, 1 foreground, 0 tanto faz (por isso precisa evitar zeros)
		hit_miss_result = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel, borderValue=0) # borda zero para restringir kernel dentro da imagem
		matches = np.argwhere(hit_miss_result == 255)
		for match in matches:
			color_image[match[0], match[1]] = [0, 0, 255]  # Pintar bifurcações de vermelho BGR
		cv2.imshow(f'{kernel} hits do kernel', hit_miss_result)
		cv2.imshow(f'{kernel} hits acumulado', color_image)
		result_image = np.maximum(result_image, hit_miss_result)
	
	for kernel in termination_kernels:
		hit_miss_result = np.zeros_like(binary_image)
		kernel = np.where(kernel == 0, -1, kernel) # -1 indica background, 1 foreground, 0 tanto faz (por isso precisa evitar zeros)
		hit_miss_result = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel, borderValue=0) # borda zero para restringir kernel dentro da imagem
		matches = np.argwhere(hit_miss_result == 255)
		for match in matches:
			color_image[match[0], match[1]] = [0, 255, 0]  # Pintar terminações de verde BGR
		cv2.imshow(f'{kernel} hits do kernel', hit_miss_result)
		cv2.imshow(f'{kernel} hits acumulado', color_image)
		result_image = np.maximum(result_image, hit_miss_result)

	cv2.imshow('Testcase Visualization', binary_image)
	cv2.imshow('Hit-Miss Result', result_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	# https://docs.opencv.org/4.x/db/d06/tutorial_hitOrMiss.html
	# https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f


def testarExtracao():
	testcase = [
		[0, 0, 0, 1, 0, 0, 1],
		[0, 1, 0, 1, 1, 0, 0],
		[0, 0, 1, 0, 0, 1, 1],
		[0, 1, 1, 0, 1, 0, 0],
		[0, 1, 0, 1, 0, 1, 1],
		[0, 1, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 1],
		[1, 1, 1, 0, 1, 1, 1],
		[0, 1, 0, 0, 1, 1, 1],
		[1, 1, 1, 0, 1, 1, 1],
		[0, 0, 0, 0, 0, 1, 0],
		[1, 0, 1, 0, 0, 0, 0],
		[1, 1, 1, 1, 0, 1, 0],
		[1, 0, 1, 0, 0, 0, 0],
	]

	test_array = np.array(testcase, dtype=np.uint8)
	binary_image = test_array * 255

	extrairMinucias(binary_image)

testarExtracao() # Extração no test_case sem filtrar

filtradateste = testarFiltroMorf()
extrairMinucias(filtradateste) # Extração no test_case filtrado