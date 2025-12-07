import cv2 
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np

def fft_enhance(gray, hp_radius=30, use_butterworth=False, order=2):
    """
    Aplica filtro passa-alta no domínio da frequência.
    hp_radius: raio de corte em pixels (quanto maior, menos alto-frequência é removida).
    use_butterworth: se True usa Butterworth HP, caso contrário usa máscara ideal.
    order: ordem do filtro Butterworth.
    Retorna imagem realçada (uint8).
    """
    rows, cols = gray.shape
    # DFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # construir máscara
    crow, ccol = rows // 2, cols // 2
    if use_butterworth:
        u = np.arange(cols) - ccol
        v = np.arange(rows) - crow
        U, V = np.meshgrid(u, v)
        D = np.sqrt(U**2 + V**2)
        # Butterworth high-pass: H = 1 / (1 + (D0/D)^(2n))
        H = 1.0 / (1.0 + (hp_radius / (D + 1e-8))**(2 * order))
        mask = np.dstack([H, H]).astype(np.float32)
    else:
        mask = np.ones((rows, cols, 2), np.float32)
        Y, X = np.ogrid[:rows, :cols]
        mask_area = (Y - crow)**2 + (X - ccol)**2 <= (hp_radius**2)
        mask[mask_area] = 0  # remove componentes de baixa frequência

    # aplicar e voltar ao espaço imagem
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_back

def block_fft_enhance(gray, block_size=32, k=0.45, stride=None, post_smooth_sigma=1.0):
    """
    Aprimora imagem em blocos seguindo Chikkerur:
    - usa overlap-add (stride, por padrão 50% do bloco) para reduzir artefatos de blocos;
    - normaliza a magnitude por bloco antes de aplicar mag**k;
    - opcional suavização final (post_smooth_sigma > 0).
    Retorna uint8 normalizado (0..255).
    """
    rows, cols = gray.shape
    bs = block_size
    if stride is None:
        stride = bs // 2  # overlap 50%
    # pad para garantir cobertura com o stride
    pad_r = (bs - (rows - bs) % stride - bs) % stride if rows > bs else (bs - rows)
    pad_c = (bs - (cols - bs) % stride - bs) % stride if cols > bs else (bs - cols)
    if pad_r == 0 and rows < bs:
        pad_r = bs - rows
    if pad_c == 0 and cols < bs:
        pad_c = bs - cols
    gray_p = np.pad(gray.astype(np.float32), ((0, pad_r), (0, pad_c)), mode='reflect')
    r_p, c_p = gray_p.shape

    out = np.zeros_like(gray_p, dtype=np.float32)
    weight = np.zeros_like(gray_p, dtype=np.float32)
    win1d = np.hanning(bs)
    window = np.outer(win1d, win1d).astype(np.float32)
    eps = 1e-8

    # percorre com overlap
    i = 0
    while i <= r_p - bs:
        j = 0
        while j <= c_p - bs:
            block = gray_p[i:i+bs, j:j+bs].astype(np.float32)
            F = np.fft.fft2(block)
            mag = np.abs(F)
            # normalizar magnitude por bloco para evitar diferenças extremas
            mag_norm = mag / (np.mean(mag) + eps)
            gain = (mag_norm + eps) ** k
            Fp = F * gain
            recon = np.real(np.fft.ifft2(Fp))
            out[i:i+bs, j:j+bs] += recon * window
            weight[i:i+bs, j:j+bs] += window
            j += stride
        # garantir cobertura da borda direita
        if j < c_p:
            j = c_p - bs
            block = gray_p[i:i+bs, j:j+bs].astype(np.float32)
            F = np.fft.fft2(block)
            mag = np.abs(F)
            mag_norm = mag / (np.mean(mag) + eps)
            gain = (mag_norm + eps) ** k
            recon = np.real(np.fft.ifft2(F * gain))
            out[i:i+bs, j:j+bs] += recon * window
            weight[i:i+bs, j:j+bs] += window
        i += stride

    # garantir cobertura da borda inferior
    if i < r_p:
        i = r_p - bs
        j = 0
        while j <= c_p - bs:
            block = gray_p[i:i+bs, j:j+bs].astype(np.float32)
            F = np.fft.fft2(block)
            mag = np.abs(F)
            mag_norm = mag / (np.mean(mag) + eps)
            gain = (mag_norm + eps) ** k
            recon = np.real(np.fft.ifft2(F * gain))
            out[i:i+bs, j:j+bs] += recon * window
            weight[i:i+bs, j:j+bs] += window
            j += stride
        if j < c_p:
            j = c_p - bs
            block = gray_p[i:i+bs, j:j+bs].astype(np.float32)
            F = np.fft.fft2(block)
            mag = np.abs(F)
            mag_norm = mag / (np.mean(mag) + eps)
            gain = (mag_norm + eps) ** k
            recon = np.real(np.fft.ifft2(F * gain))
            out[i:i+bs, j:j+bs] += recon * window
            weight[i:i+bs, j:j+bs] += window

    out = out / (weight + eps)
    out = out[:rows, :cols]
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return out

def select_image(start_dir=None):
    root = tk.Tk()
    root.withdraw()
    start = start_dir or os.path.expanduser("~")
    path = filedialog.askopenfilename(title="Escolha uma imagem",
                                       initialdir=start,
                                       filetypes=[("Imagens","*.tif *.png *.jpg *.jpeg *.bmp"), ("Todos os arquivos","*.*")])
    root.destroy()
    return path

def equalizar(path, save_output=True, use_fft=False, use_block_fft=False, hp_radius=30, use_butterworth=False, block_size=32, k=0.45):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Erro ao carregar a imagem" + str(path))
    escala_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalizada = cv2.equalizeHist(escala_cinza)

    global_fft = None
    block_fft = None
    if use_fft:
        # calcula FFT global sempre (opcional) e também o aprimoramento por blocos se pedido
        global_fft = fft_enhance(equalizada, hp_radius=hp_radius, use_butterworth=use_butterworth, order=2)
        if use_block_fft:
            block_fft = block_fft_enhance(equalizada, block_size=block_size, k=k)

    # criar 4 janelas e mostrar (usar janela vazia se não houver resultado)
    h, w = escala_cinza.shape
    gap = 10
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Equalizada", cv2.WINDOW_NORMAL)
    cv2.namedWindow("FFT Block", cv2.WINDOW_NORMAL)
    cv2.namedWindow("FFT Global", cv2.WINDOW_NORMAL)

    cv2.imshow("Original", escala_cinza)
    cv2.imshow("Equalizada", equalizada)
    cv2.imshow("FFT Block", block_fft if block_fft is not None else np.zeros_like(equalizada))
    cv2.imshow("FFT Global", global_fft if global_fft is not None else np.zeros_like(equalizada))

    # posicionar janelas em 2x2 para melhor visualização
    try:
        cv2.moveWindow("Original", 0, 0)
        cv2.moveWindow("Equalizada", w + gap, 0)
        cv2.moveWindow("FFT Block", 0, h + gap)
        cv2.moveWindow("FFT Global", w + gap, h + gap)
    except Exception:
        # alguns backends podem ignorar moveWindow; não é crítico
        pass

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # salvar resultados separados se solicitado
    if save_output:
        base, ext = os.path.splitext(os.path.basename(path))
        out_dir = os.path.dirname(path)
        cv2.imwrite(os.path.join(out_dir, f"{base}_original{ext}"), escala_cinza)
        cv2.imwrite(os.path.join(out_dir, f"{base}_equalizada{ext}"), equalizada)
        if block_fft is not None:
            cv2.imwrite(os.path.join(out_dir, f"{base}_fft_block{ext}"), block_fft)
            print(f"Salvo: {base}_fft_block{ext}")
        if global_fft is not None:
            cv2.imwrite(os.path.join(out_dir, f"{base}_fft_global{ext}"), global_fft)
            print(f"Salvo: {base}_fft_global{ext}")
        print(f"Salvo: {base}_equalizada{ext}")

    # retorna o resultado principal (prioriza block_fft quando presente)
    if use_fft:
        return block_fft if block_fft is not None else global_fft
    return equalizada

if __name__ == "__main__":
    diretorio = r"C:\Users\Renata\Documents\Faculdade\pdi\Trabalho Final\DB1_B"
    path_imagem = select_image(start_dir=diretorio)
    if path_imagem:
        equalizar(path_imagem, use_fft=True, use_block_fft=True, block_size=32, k=0.45)
    else:
        print("Nenhuma imagem selecionada.")
