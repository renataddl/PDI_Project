import cv2
import os
import numpy as np 
import math

# --- CONFIGURAÇÃO ---
ARQUIVO_ALVO = "DB1_B/101_3.tif" 

# ==========================================
# 1. FUNÇÕES DE PRÉ-PROCESSAMENTO
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

def estimar_roi_variancia(img, block_size=16, threshold_std=10.0):
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
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, kernel_small, iterations=1)
    return mask

def aplicar_afinamento(img_binaria):
    _, img_thresh = cv2.threshold(img_binaria, 127, 255, cv2.THRESH_BINARY)
    try:
        skeleton = cv2.ximgproc.thinning(img_thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return skeleton
    except AttributeError:
        print("[AVISO] cv2.ximgproc não encontrado. Instale 'opencv-contrib-python'.")
        return img_thresh

def aplicar_filtros_morfologicos(binary_image):
    filter_kernels = [
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype="int"),
        np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype="int"),
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype="int")
    ]
    result_image = binary_image.copy()
    for kernel in filter_kernels:
        k = np.where(kernel == 0, -1, kernel) 
        match_map = cv2.morphologyEx(result_image, cv2.MORPH_HITMISS, k)
        result_image[match_map > 0] = 0
    return result_image

# ==========================================
# 2. EXTRAÇÃO E PÓS-PROCESSAMENTO
# ==========================================

def compute_crossing_number(skeleton):
    rows, cols = skeleton.shape
    padded = np.pad(skeleton, 1, mode='constant')
    minutiae_list = []
    skel_bool = (padded > 0).astype(int)

    idx = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if skel_bool[r, c] == 1:
                cn_val = 0
                for i in range(8):
                    val_curr = skel_bool[r + idx[i][0], c + idx[i][1]]
                    val_next = skel_bool[r + idx[i+1][0], c + idx[i+1][1]]
                    cn_val += abs(val_curr - val_next)
                cn_val = cn_val // 2

                if cn_val == 1:
                    minutiae_list.append({'x': c-1, 'y': r-1, 'type': 'Terminacao'})
                elif cn_val == 3:
                    minutiae_list.append({'x': c-1, 'y': r-1, 'type': 'Bifurcacao'})
    return minutiae_list

def filtrar_minucias(minutiae_list, roi_mask, distance_thresh=10):
    valid_minutiae = []
    
    # 1. Filtro de ROI Segura
    kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    roi_safe = cv2.erode(roi_mask, kernel_safe)

    temp_list = []
    for m in minutiae_list:
        if 0 <= m['y'] < roi_safe.shape[0] and 0 <= m['x'] < roi_safe.shape[1]:
            if roi_safe[m['y'], m['x']] > 0:
                temp_list.append(m)
    
    # 2. Filtro de Distância Euclidiana
    ignore_indices = set()
    for i in range(len(temp_list)):
        if i in ignore_indices: continue
        m1 = temp_list[i]
        for j in range(i + 1, len(temp_list)):
            if j in ignore_indices: continue
            m2 = temp_list[j]
            dist = math.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
            if dist < distance_thresh:
                ignore_indices.add(i)
                ignore_indices.add(j)
    
    for i in range(len(temp_list)):
        if i not in ignore_indices:
            valid_minutiae.append(temp_list[i])

    return valid_minutiae

def desenhar_minucias(img_rgb, minutiae_list):
    vis = img_rgb.copy()
    for m in minutiae_list:
        # BGR: Verde=(0,255,0), Vermelho=(0,0,255)
        color = (0, 255, 0) if m['type'] == 'Terminacao' else (0, 0, 255)
        cv2.circle(vis, (m['x'], m['y']), 4, color, 1)
        if m['type'] == 'Bifurcacao':
            cv2.rectangle(vis, (m['x']-3, m['y']-3), (m['x']+3, m['y']+3), color, 1)
    return vis

# ==========================================
# 3. VISUALIZAÇÃO COM HCONCAT (3 em 3)
# ==========================================

def mostrar_imagens_agrupadas(etapas_dict):
    """
    Agrupa imagens de 3 em 3 e exibe em janelas unificadas.
    """
    print("\n--- Visualização ---")
    print("Pressione qualquer tecla para encerrar.")
    
    # Lista auxiliar para processar
    chaves = list(etapas_dict.keys())
    imgs_processadas = []

    # Prepara todas as imagens (converte para BGR e adiciona texto)
    for titulo in chaves:
        img = etapas_dict[titulo]
        
        # Converte grayscale para BGR para poder concatenar com as coloridas
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()
            
        # Adiciona o título na própria imagem (canto superior esquerdo)
        # Fundo preto no texto para legibilidade
        cv2.rectangle(img_bgr, (0,0), (img_bgr.shape[1], 40), (0,0,0), -1)
        cv2.putText(img_bgr, titulo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        imgs_processadas.append(img_bgr)

    # Loop para criar os painéis de 3 em 3
    grupo_id = 1
    for i in range(0, len(imgs_processadas), 3):
        # Pega fatia de 3 imagens
        batch = imgs_processadas[i : i+3]
        
        # Se for o último e não tiver 3, o hconcat funciona igual (concatena o que tiver)
        if len(batch) > 0:
            painel = cv2.hconcat(batch)
            
            nome_janela = f"Painel {grupo_id} (Imagens {i+1}-{i+len(batch)})"
            cv2.imshow(nome_janela, painel)
            
            # Posiciona janelas em cascata para não sobrepor totalmente
            cv2.moveWindow(nome_janela, 50 + (grupo_id-1)*40, 50 + (grupo_id-1)*40)
            grupo_id += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processar_imagem(path):
    print(f"Lendo: {path}")
    img_bgr = cv2.imread(path)
    if img_bgr is None: print("Erro arquivo."); return

    etapas = {}

    # 1. Leitura
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.mean(img_gray) > 127: img_gray = 255 - img_gray
    etapas["1. Original"] = img_gray

    # 2. Pré-processamento
    img_eq = cv2.equalizeHist(img_gray)
    img_limpa = espalhamento_contraste_local(img_eq)
    etapas["2. Contraste"] = img_limpa

    img_fft = aplicar_fft(img_limpa, k=0.45)
    etapas["3. FFT"] = img_fft
    
    img_suave = aplicar_suavizacao(img_fft)
    etapas["4. Suavizacao"] = img_suave

    # 3. ROI e Binarização
    roi_mask = estimar_roi_variancia(img_suave, threshold_std=10.0) 
    etapas["5. ROI Mask"] = roi_mask

    img_bin = binarizar_otsu_local(img_suave)
    img_bin_roi = cv2.bitwise_and(img_bin, img_bin, mask=roi_mask)
    etapas["6. Binaria+ROI"] = img_bin_roi

    # 4. Esqueleto
    img_esqueleto = aplicar_afinamento(img_bin_roi)
    etapas["7. Esqueleto"] = img_esqueleto

    img_esqueleto_limpo = aplicar_filtros_morfologicos(img_esqueleto)
    etapas["8. Esq. Limpo"] = img_esqueleto_limpo

    # 5. Extração e Filtragem
    lista_bruta = compute_crossing_number(img_esqueleto_limpo)
    
    # Desenho das brutas
    esq_bgr_bruto = cv2.cvtColor(img_esqueleto_limpo, cv2.COLOR_GRAY2BGR)
    vis_bruta = desenhar_minucias(esq_bgr_bruto, lista_bruta)
    etapas["9. Minucias Brutas"] = vis_bruta

    # Filtragem
    lista_filtrada = filtrar_minucias(lista_bruta, roi_mask, distance_thresh=10)
    print(f"Minúcias Finais: {len(lista_filtrada)}")
    
    # Desenho final
    esq_bgr_final = cv2.cvtColor(img_esqueleto_limpo, cv2.COLOR_GRAY2BGR)
    vis_final = desenhar_minucias(esq_bgr_final, lista_filtrada)
    etapas["10. Minucias Finais"] = vis_final

    # Resultado sobreposto na original
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = desenhar_minucias(overlay, lista_filtrada)
    etapas["11. Overlay Final"] = overlay

    # Exibe agrupado
    mostrar_imagens_agrupadas(etapas)

if __name__ == "__main__":
    if os.path.exists(ARQUIVO_ALVO):
        processar_imagem(ARQUIVO_ALVO)
    else:
        print(f"Arquivo '{ARQUIVO_ALVO}' não encontrado.")