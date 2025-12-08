import cv2
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np 

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

def processar_imagem(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Erro ao carregar a imagem " + str(path))
    img = 255 - img 
    
    escala_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalizada = cv2.equalizeHist(escala_cinza)   
    imagem_fft = aplicar_fft(equalizada, k=0.45)
    imagem_suavizada = aplicar_suavizacao(imagem_fft)

    step1 = cv2.hconcat([
        cv2.cvtColor(escala_cinza, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(equalizada, cv2.COLOR_GRAY2BGR)
    ])
    step2 = cv2.hconcat([
        cv2.cvtColor(equalizada, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(imagem_fft, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(imagem_suavizada, cv2.COLOR_GRAY2BGR)
    ])
    cv2.imshow("Etapa 1: Pre-processamento inicial", step1)
    cv2.imshow("Etapa 2 e 3: FFT e Suavizacao", step2)
    
    print("Pressione qualquer tecla na janela da imagem para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return imagem_suavizada

if __name__ == "__main__":
    diretorio = r"/home/scalcon/Desktop/facul/pdi/PDI_Project"
    
    path_imagem = select_image(start_dir=diretorio)
    
    if path_imagem:
        print(f"Processando: {path_imagem}")
        processar_imagem(path_imagem)
    else:
        print("Nenhuma imagem selecionada.")