import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt # Necessário instalar: pip install matplotlib

# --- CONFIGURAÇÃO ---
# Coloque o nome exato do seu arquivo aqui. 
# Se estiver na mesma pasta do script, só o nome basta.
ARQUIVO_ALVO = "101_3.tif" 

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
    """
    Suavização aprimorada para remover artefatos de bloco da FFT.
    """
    # 1. Filtro Gaussiano um pouco maior para conectar as cristas quebradas pelos blocos
    # Aumentar o kernel de (5,5) para (7,7) ajuda a "fundir" as bordas dos blocos.
    img_gauss = cv2.GaussianBlur(img, (7, 7), 1.0)
    
    # 2. Opcional: Filtro de Mediana para remover ruído "sal e pimenta" residual
    # Isso limpa os pontinhos pretos/brancos que sobram dentro das cristas.
    img_mediana = cv2.medianBlur(img_gauss, 3)
    
    return img_mediana

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

def estimar_roi_variancia(img, block_size=16, threshold_std=10.0):
    rows, cols = img.shape
    mask = np.zeros_like(img)
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            block = img[r:r_end, c:c_end]
            if block.size == 0: continue
            if np.std(block) > threshold_std:
                mask[r:r_end, c:c_end] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size*2, block_size*2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, kernel_erode, iterations=1)
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

def visualizar_resultados_plt(orig, limpa, fft, roi, binaria, direcional):
    """
    Exibe os resultados usando Matplotlib (Melhor para zoom e análise)
    """
    plt.figure(figsize=(12, 8))
    
    titulos = ["1. Original Invertida", "2. Limpa (Hong et al.)", "3. FFT + Suavizada", 
               "4. Máscara ROI", "5. Binária Final", "6. Direção"]
    imagens = [orig, limpa, fft, roi, binaria, cv2.cvtColor(direcional, cv2.COLOR_BGR2RGB)]
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(imagens[i], cmap='gray' if i < 5 else None)
        plt.title(titulos[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def aplicar_afinamento(img_binaria):
    """
    Aplica o algoritmo de Zhang-Suen para reduzir as cristas a 1 pixel de largura.
    Requer: pip install opencv-contrib-python
    """
    # Verifica se a imagem está no formato correto (0 e 255)
    # Zhang-Suen espera cristas BRANCAS (255) e fundo PRETO (0)
    
    try:
        # O OpenCV tem o Zhang-Suen nativo no módulo ximgproc
        # THINNING_ZHANGSUEN = 0
        skeleton = cv2.ximgproc.thinning(img_binaria, thinningType=0)
        return skeleton
    except AttributeError:
        print("\n[ERRO] O módulo 'cv2.ximgproc' não foi encontrado.")
        print("Para usar o Zhang-Suen, instale a versão contrib do OpenCV:")
        print("   pip uninstall opencv-python")
        print("   pip install opencv-contrib-python\n")
        return img_binaria # Retorna sem afinar para não travar    

def processar_imagem(path):
    print(f"Lendo imagem: {path}")
    img_bgr = cv2.imread(path)
    if img_bgr is None: 
        print("ERRO: Imagem não encontrada.")
        return

    # --- PIPELINE ---
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Inversão para garantir Fundo Preto / Crista Branca
    if np.mean(img_gray) > 127:
        print(" > Invertendo cores...")
        img_gray = 255 - img_gray

    img_eq = cv2.equalizeHist(img_gray)
    img_limpa = espalhamento_contraste_local(img_eq)
    img_fft = aplicar_fft(img_limpa, k=0.45)
    img_suave = aplicar_suavizacao(img_fft)
    
    print(" > Calculando ROI e Binarização...")
    roi_mask = estimar_roi_variancia(img_suave, threshold_std=5.0)
    img_bin = binarizar_otsu_local(img_suave)
    
    # Aplica ROI na Binária
    img_bin_roi = cv2.bitwise_and(img_bin, img_bin, mask=roi_mask)

    # --- NOVO PASSO: AFINAMENTO ---
    print(" > Aplicando Afinamento (Zhang-Suen)...")
    img_esqueleto = aplicar_afinamento(img_bin_roi)

    # Direção (apenas visualização)
    vis_dir = estimar_imagem_direcional(img_suave, roi_mask)

    # --- VISUALIZAÇÃO ---
    visualizar_resultados_plt(img_gray, img_fft, roi_mask, img_bin_roi, img_esqueleto, vis_dir)

if __name__ == "__main__":
    # Verifica se o arquivo existe no diretório atual
    if os.path.exists(ARQUIVO_ALVO):
        processar_imagem(ARQUIVO_ALVO)
    else:
        # Se não achar o arquivo fixo, tenta abrir o seletor ou avisa
        print(f"Arquivo '{ARQUIVO_ALVO}' não encontrado na pasta do script.")
        # Se quiser fallback para seletor, descomente as linhas abaixo:
        # root = tk.Tk(); root.withdraw()
        # path = filedialog.askopenfilename()
        # if path: processar_imagem(path)