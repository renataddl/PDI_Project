import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt 

# --- CONFIGURAÇÃO ---
ARQUIVO_ALVO = "101_3.tif" 

# ==========================================
# FUNÇÕES DE PROCESSAMENTO BÁSICO
# ==========================================

def aplicar_fft(img, block_size=32, k=0.45):
    img_float = img.astype(np.float32)
    rows, cols = img.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    padded_img = np.pad(img_float, ((0, pad_rows), (0, pad_cols)), mode='constant')
    
    enhanced_img = np.zeros_like(padded_img)
    padded_rows, padded_cols = padded_img.shape
    
    for r in range(0, padded_rows, block_size):
        for c in range(0, padded_cols, block_size):
            block = padded_img[r:r+block_size, c:c+block_size]
            f = np.fft.fft2(block)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            enhanced_f = fshift * (magnitude ** k)
            f_ishift = np.fft.ifftshift(enhanced_f)
            img_back = np.fft.ifft2(f_ishift)
            enhanced_img[r:r+block_size, c:c+block_size] = np.abs(img_back)

    result = enhanced_img[:rows, :cols]
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def aplicar_suavizacao(img):
    img_media = cv2.blur(img, (3, 3))
    img_gauss = cv2.GaussianBlur(img_media, (5, 5), 0)
    return img_gauss

def binarizar_otsu_local(img, block_size=32):
    rows, cols = img.shape
    binaria = np.zeros_like(img)
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            block = img[r:r_end, c:c_end]
            if block.size == 0: continue
            thresh, bin_block = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binaria[r:r_end, c:c_end] = bin_block
    return binaria

def espalhamento_contraste_local(img, kernel_size=(5, 5)):
    img_media = cv2.blur(img, kernel_size)
    img_saida = img.copy()
    mascara_fundo = img < img_media
    img_saida[mascara_fundo] = 0
    return img_saida

def estimar_roi_variancia(img, block_size=16, threshold_std=20.0):
    rows, cols = img.shape
    mask = np.zeros_like(img)
    img_float = img.astype(np.float32)
    
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            block = img_float[r:r_end, c:c_end]
            if block.size == 0: continue
            if np.std(block) > threshold_std:
                mask[r:r_end, c:c_end] = 255

    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size*2, block_size*2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large)
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel_small, iterations=4)
    return mask

def estimar_imagem_direcional(img, roi_mask, block_size=16):
    rows, cols = img.shape
    vis_orientacao = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            cx, cy = c + block_size // 2, r + block_size // 2
            if cx < cols and cy < rows:
                if roi_mask[cy, cx] == 0: continue

            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            gxb = gx[r:r_end, c:c_end]
            gyb = gy[r:r_end, c:c_end]
            val_xy = 2 * np.sum(gxb * gyb)
            val_xx_yy = np.sum(gxb**2 - gyb**2)
            
            if val_xx_yy == 0 and val_xy == 0: angle = 0
            else: angle = 0.5 * np.arctan2(val_xy, val_xx_yy)
            
            length = block_size // 2 - 2
            x2 = int(cx + length * np.cos(angle + np.pi/2))
            y2 = int(cy + length * np.sin(angle + np.pi/2))
            x1 = int(cx - length * np.cos(angle + np.pi/2))
            y1 = int(cy - length * np.sin(angle + np.pi/2))
            cv2.line(vis_orientacao, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return vis_orientacao

def aplicar_afinamento(img_binaria):
    _, img_thresh = cv2.threshold(img_binaria, 127, 255, cv2.THRESH_BINARY)
    try:
        skeleton = cv2.ximgproc.thinning(img_thresh, thinningType=0) # Zhang-Suen
        return skeleton
    except AttributeError:
        print("[AVISO] cv2.ximgproc não encontrado. Rodando sem afinamento.")
        return img_thresh

# ==========================================
# NOVAS FUNÇÕES: LIMPEZA E MINÚCIAS
# ==========================================

def aplicar_filtros_morfologicos(binary_image):
    """
    Remove artefatos do esqueleto (spurs, breaks) usando Hit-Miss.
    """
    # Definição dos Kernels para limpeza
    # 0 -> Deve ser PRETO (Fundo)
    # 1 -> Deve ser BRANCO (Frente)
    filter_kernels = [
        # Clean: remove pixel isolado (1 cercado de 0s)
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype="int"),
        # Hbreak vertical
        np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype="int"),
        # Hbreak horizontal
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype="int")
    ]

    # Spurs (Espinhos)
    spur1 = np.array([[0,0,0,0,0], [-1,-1,-1,-1,0], [-1,-1,1,-1,0], [-1,1,-1,-1,0], [1,1,-1,-1,0]], dtype="int")
    spur2 = np.array([[0,0,0,0,0], [-1,-1,-1,-1,0], [-1,-1,1,-1,0], [1,1,-1,-1,0], [1,-1,-1,-1,0]], dtype="int")
    
    spur_filter_kernels = []
    for _ in range(4):
        spur1 = np.rot90(spur1); spur_filter_kernels.append(spur1)
        spur2 = np.rot90(spur2); spur_filter_kernels.append(spur2)

    result_image = binary_image.copy()

    # Aplica filtros básicos
    for kernel in filter_kernels:
        # Prepara kernel para Hit-Miss: 1=FG, -1=BG
        k = np.where(kernel == 0, -1, kernel) 
        # Aumenta um pouco a borda para garantir que funcione
        match_map = cv2.morphologyEx(result_image, cv2.MORPH_HITMISS, k)
        # Remove pixels que deram match
        result_image[match_map > 0] = 0

    # Aplica filtros de Spurs
    for kernel in spur_filter_kernels:
        # A lógica do seu colega usava -1 explicitamente nos spurs
        # Vamos manter a lógica dele de conversão para garantir compatibilidade
        # Nota: Se o kernel já tem -1, o np.where vai mantê-lo. Se tem 0, vira -1.
        k = np.where(kernel == 0, -1, kernel)
        match_map = cv2.morphologyEx(result_image, cv2.MORPH_HITMISS, k)
        result_image[match_map > 0] = 0

    return result_image

def extrair_minucias(skeleton_image, roi_mask, margin=10):
    """
    Encontra bifurcações e terminações, IGNORANDO as bordas da ROI.
    margin: distância em pixels da borda para ignorar (evita falsas terminações).
    """
    # 1. Cria uma "Zona Segura" para ignorar as bordas
    # Erode a ROI pela margem especificada. Tudo fora disso é ignorado.
    kernel_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin*2, margin*2))
    zona_segura = cv2.erode(roi_mask, kernel_margin)
    
    # Bifurcação: pixel central conectado a 3 vizinhos
    bifurcation_kernels_base = [
        np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype="int"), # T
        np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype="int")  # Y
    ]
    
    # Terminação: pixel central conectado a apenas 1 vizinho
    termination_kernels_base = [
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype="int"),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype="int")
    ]

    # Gera rotações
    bifurcation_kernels = []
    for k in bifurcation_kernels_base:
        for _ in range(4): k = np.rot90(k); bifurcation_kernels.append(k)

    termination_kernels = []
    for k in termination_kernels_base:
        for _ in range(4): k = np.rot90(k); termination_kernels.append(k)

    # Imagem para desenho (BGR)
    minutiae_map = cv2.cvtColor(skeleton_image, cv2.COLOR_GRAY2BGR)
    
    # Detecta Bifurcações (Vermelho)
    for kernel in bifurcation_kernels:
        k = np.where(kernel == 0, -1, kernel)
        matches = cv2.morphologyEx(skeleton_image, cv2.MORPH_HITMISS, k)
        
        # Filtra pela Zona Segura
        matches = cv2.bitwise_and(matches, matches, mask=zona_segura)
        
        # Dilata para visualização
        matches_vis = cv2.dilate(matches, np.ones((3,3), np.uint8)) 
        minutiae_map[matches_vis > 0] = [0, 0, 255] # Red

    # Detecta Terminações (Verde)
    for kernel in termination_kernels:
        k = np.where(kernel == 0, -1, kernel)
        matches = cv2.morphologyEx(skeleton_image, cv2.MORPH_HITMISS, k)
        
        # Filtra pela Zona Segura (AQUI É O TRUQUE DO CORTE)
        matches = cv2.bitwise_and(matches, matches, mask=zona_segura)
        
        matches_vis = cv2.dilate(matches, np.ones((3,3), np.uint8))
        minutiae_map[matches_vis > 0] = [0, 255, 0] # Green

    return minutiae_map

# ==========================================
# VISUALIZAÇÃO E MAIN
# ==========================================

def visualizar_resultados_plt(orig, fft, roi, binaria, esq_bruto, esq_limpo, minucias, direcional):
    plt.figure(figsize=(16, 8))
    
    # Lista de 8 imagens para grid 2x4
    data = [
        (orig, "1. Original Inv.", 'gray'),
        (fft, "2. FFT/Suave", 'gray'),
        (roi, "3. ROI", 'gray'),
        (binaria, "4. Binária", 'gray'),
        (esq_bruto, "5. Esqueleto Bruto", 'gray'),
        (esq_limpo, "6. Esqueleto Limpo", 'gray'),
        (minucias, "7. Minúcias (V=Fim, Vm=Bif)", None),
        (cv2.cvtColor(direcional, cv2.COLOR_BGR2RGB), "8. Direção", None)
    ]
    
    for i, (img, title, cmap) in enumerate(data):
        plt.subplot(2, 4, i+1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def processar_imagem(path):
    print(f"Lendo imagem: {path}")
    img_bgr = cv2.imread(path)
    if img_bgr is None: 
        print("ERRO: Imagem não encontrada.")
        return

    # Pipeline de Pré-processamento
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.mean(img_gray) > 127: # Inversão se fundo branco
        img_gray = 255 - img_gray

    img_eq = cv2.equalizeHist(img_gray)
    img_limpa = espalhamento_contraste_local(img_eq)
    img_fft = aplicar_fft(img_limpa, k=0.45)
    img_suave = aplicar_suavizacao(img_fft)
    
    # Binarização e ROI
    print(" > Binarizando e calculando ROI...")
    roi_mask = estimar_roi_variancia(img_suave, threshold_std=7.0)
    img_bin = binarizar_otsu_local(img_suave)
    img_bin_roi = cv2.bitwise_and(img_bin, img_bin, mask=roi_mask)

    # Afinamento (Esqueleto)
    print(" > Afinamento (Zhang-Suen)...")
    img_esqueleto = aplicar_afinamento(img_bin_roi)

    # Limpeza Morfológica (NOVO)
    print(" > Limpeza Morfológica do Esqueleto...")
    img_esqueleto_limpo = aplicar_filtros_morfologicos(img_esqueleto)

    # Extração de Minúcias (AGORA COM CORTE DE BORDAS)
    print(" > Extraindo Minúcias (com margem de segurança)...")
    # margin=15 remove minúcias a 15 pixels da borda da digital
    img_minucias = extrair_minucias(img_esqueleto_limpo, roi_mask, margin=15)

    # Direção (apenas para visualização)
    vis_dir = estimar_imagem_direcional(img_suave, roi_mask)

    # Visualização Completa (2x4)
    visualizar_resultados_plt(
        img_gray, img_fft, roi_mask, img_bin_roi, 
        img_esqueleto, img_esqueleto_limpo, img_minucias, vis_dir
    )

if __name__ == "__main__":
    if os.path.exists(ARQUIVO_ALVO):
        processar_imagem(ARQUIVO_ALVO)
    else:
        print(f"Arquivo '{ARQUIVO_ALVO}' não encontrado.")