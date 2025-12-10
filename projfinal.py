# https://www.teses.usp.br/teses/disponiveis/18/18152/tde-15102008-135808/publico/Ricardo.pdf
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
def testarFiltrosMorf():
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
	como os outros:

	spur1 = np.array([
		[0, 0, 0, 0],
		[0, 0, 1, 0],
		[0, 1, 0, 0],
		[1, 1, 0, 0]
	], dtype="int"),

	spur2 = np.array([
		[0, 0, 0, 0],
		[0, 0, 1, 0],
		[1, 1, 0, 0],
		[1, 0, 0, 0]
	], dtype="int"),

	for _ in range(4):
		kernel = np.rot90(spur1)
		filter_kernels.append(kernel)

	for _ in range(4):
		kernel = np.rot90(spur2)
		filter_kernels.append(kernel)
	'''

	# Conferir os kernels:
	print('filter_kernels:')
	for kernel in filter_kernels:
		print(kernel)
		print()

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
	result_image = binary_image.copy()

	for kernel in filter_kernels:
		kernel = np.where(kernel == 0, -1, kernel) # -1 indica background, 1 foreground, 0 tanto faz (por isso precisa evitar zeros)
		match_map = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel, borderType=cv2.BORDER_REPLICATE) # replicar a borda para restringir kernel dentro da imagem
		result_image[match_map > 0] = 0

		cv2.imshow(f'{kernel} matches', match_map)
		cv2.imshow(f'{kernel} result', result_image)

	cv2.imshow('Testcase Visualization', binary_image)
	cv2.imshow('Filter', result_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

testarFiltrosMorf()






### Extracao de minucias ########################
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

	bifurcation_kernels = [
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
	
	termination_kernels = [
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

	extraction_kernels = []

	# Compute all rotations
	for kernel in bifurcation_kernels:
		for _ in range(4):
			kernel = np.rot90(kernel)
			extraction_kernels.append(kernel)

	for kernel in termination_kernels:
		for _ in range(4):
			kernel = np.rot90(kernel)
			extraction_kernels.append(kernel)

	# Conferir os kernels
	print('extraction_kernels:')
	for kernel in extraction_kernels:
		print(kernel)
		print()

	test_array = np.array(testcase, dtype=np.uint8)
	binary_image = test_array * 255
	color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
	result_image = np.zeros_like(binary_image)

	for kernel in extraction_kernels:
		hit_miss_result = np.zeros_like(binary_image)
		kernel = np.where(kernel == 0, -1, kernel) # -1 indica background, 1 foreground, 0 tanto faz (por isso precisa evitar zeros)
		hit_miss_result = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel, borderValue=0) # borda zero para restringir kernel dentro da imagem
		matches = np.argwhere(hit_miss_result == 255)
		for match in matches:
			color_image[match[0], match[1]] = [0, 0, 255]  # Pintar hit de vermelho BGR
		cv2.imshow(f'{kernel} hits do kernel', hit_miss_result)
		cv2.imshow(f'{kernel} hits acumulado', color_image)
		result_image = np.maximum(result_image, hit_miss_result)

	cv2.imshow('Testcase Visualization', binary_image)
	cv2.imshow('Hit-Miss Result', result_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	# https://docs.opencv.org/4.x/db/d06/tutorial_hitOrMiss.html
	# https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f

testarExtracao()