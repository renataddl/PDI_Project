import cv2
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np 
import math

def select_image(start_dir=None):
    root = tk.Tk()
    root.withdraw()
    start = start_dir or os.path.expanduser("~")
    path = filedialog.askopenfilename(title="Escolha uma imagem",
                                      initialdir=start,
                                      filetypes=[("Imagens", "*.tif *.png *.jpg *.bmp")])
    root.destroy()
    return path

def aplicar_fft(img, block_size=32, k=0.45):
    rows, cols = img.shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    padded_img = np.pad(img, ((0, pad_rows), (0, pad_cols)), mode='constant')
    
    padded_rows, padded_cols = padded_img.shape
    enhanced_img = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    
    for r in range(0, padded_rows, block_size):
        for c in range(0, padded_cols, block_size):
            block = padded_img[r:r+block_size, c:c+block_size]
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue
            f = np.fft.fft2(block)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            enhanced_f = fshift * (magnitude ** k)
            f_ishift = np.fft.ifftshift(enhanced_f)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            enhanced_img[r:r+block_size, c:c+block_size] = img_back

    enhanced_img = enhanced_img[:rows, :cols]
    enhanced_img = cv2.normalize(enhanced_img, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced_img.astype(np.uint8)

def aplicar_suavizacao(img):
    img_media = cv2.blur(img, (3, 3))
    img_gauss = cv2.GaussianBlur(img_media, (5, 5), 0)
    return img_gauss

def binarizar_otsu(img):
    #TODO: BINARIZAR DENTRO DA GRID
    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria

def estimar_imagem_direcional(img, block_size=16):
    rows, cols = img.shape
    orientations = np.zeros((rows, cols), dtype=np.float32)
    
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    # Imagem para visualização das linhas de direção
    vis_orientacao = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 2. Divide a imagem em blocos W x W
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # Define limites do bloco
            r_end = min(r + block_size, rows)
            c_end = min(c + block_size, cols)
            
            # Extrai gradientes do bloco
            block_gx = gx[r:r_end, c:c_end]
            block_gy = gy[r:r_end, c:c_end]
            
            # 4. Estima orientação local (Eq 5.6, 5.7 e 5.14 para corrigir ambiguidade)
            # Numerador: 2 * Gx * Gy
            # Denominador: Gx^2 - Gy^2
            # Theta = 0.5 * atan2(numerador, denominador)
            
            val_xy = 2 * np.sum(block_gx * block_gy)
            val_xx_yy = np.sum(block_gx**2 - block_gy**2)
            
            if val_xx_yy == 0 and val_xy == 0:
                angle = 0
            else:
                # np.arctan2 já lida com os quadrantes corretamente
                angle = 0.5 * np.arctan2(val_xy, val_xx_yy)
            
            # Salva o ângulo no mapa de orientações
            orientations[r:r_end, c:c_end] = angle
            
            # --- Visualização (Desenhar linhas) ---
            # Centro do bloco
            cx = c + block_size // 2
            cy = r + block_size // 2
            
            if cx < cols and cy < rows:
                # Comprimento da linha para visualização
                length = block_size // 2 - 2
                
                # Coordenadas da linha baseadas no ângulo (ortogonal ao gradiente)
                # O gradiente aponta na direção de maior mudança (borda), 
                # a crista corre perpendicular a isso. 
                # A fórmula acima já dá o ângulo da crista (theta).
                
                x2 = int(cx + length * np.cos(angle + np.pi/2)) 
                y2 = int(cy + length * np.sin(angle + np.pi/2))
                x1 = int(cx - length * np.cos(angle + np.pi/2))
                y1 = int(cy - length * np.sin(angle + np.pi/2))
                
                cv2.line(vis_orientacao, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return orientations, vis_orientacao

def processar_imagem(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Erro ao carregar a imagem " + str(path))
    
    #img = 255 - img 
    
    escala_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    equalizada = cv2.equalizeHist(escala_cinza)   
    imagem_fft = aplicar_fft(equalizada, k=0.45)
    imagem_suavizada = aplicar_suavizacao(imagem_fft)

    imagem_binaria = binarizar_otsu(imagem_suavizada)

    mapa_orientacao, vis_orientacao = estimar_imagem_direcional(imagem_suavizada, block_size=16)

    step1 = cv2.hconcat([
        cv2.cvtColor(equalizada, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(imagem_fft, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(imagem_suavizada, cv2.COLOR_GRAY2BGR)
    ])
    
    step2 = cv2.hconcat([
        cv2.cvtColor(imagem_binaria, cv2.COLOR_GRAY2BGR),
        vis_orientacao
    ])

    cv2.imshow("Pre-processamento (Equalizada -> FFT -> Suavizada)", step1)
    cv2.imshow("Binarizacao (Otsu) e Orientacao (Sobel)", step2)
    
    print("Pipeline executado.")
    print("Janela 1: Pre-processamento.")
    print("Janela 2: Esquerda = Binarizacao Otsu, Direita = Mapa Direcional estimado.")
    print("Pressione qualquer tecla para fechar...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return imagem_binaria, mapa_orientacao

if __name__ == "__main__":
    diretorio = r"/home/scalcon/Desktop/facul/pdi/PDI_Project"
    
    path_imagem = select_image(start_dir=diretorio)
    
    if path_imagem:
        print(f"Processando: {path_imagem}")
        processar_imagem(path_imagem)
    else:
        print("Nenhuma imagem selecionada.")