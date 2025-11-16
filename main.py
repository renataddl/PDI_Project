import cv2 
import os
import tkinter as tk
from tkinter import filedialog

def select_image(start_dir=None):
    root = tk.Tk()
    root.withdraw()
    start = start_dir or os.path.expanduser("~")
    path = filedialog.askopenfilename(title="Escolha uma imagem",
                                       initialdir=start,
                                       filetypes=[("Imagens","*.tif")])
    root.destroy()
    return path

def equalizar(path, save_output=True):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Erro ao carregar a imagem" + str(path))
    escala_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalizada = cv2.equalizeHist(escala_cinza)
    mostrar = cv2.hconcat([cv2.cvtColor(escala_cinza,cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(equalizada,cv2.COLOR_GRAY2BGR)])
    cv2.imshow("original vs equalizada", mostrar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save_output:
        base, ext = os.path.splitext(os.path.basename(path))
        out_path = os.path.join(os.path.dirname(path), f"{base}_equalizada{ext}")
        cv2.imwrite(out_path, equalizada)
        print(f"Imagem equalizada salva em: {out_path}")
    return equalizada

if __name__ == "__main__":
    diretorio = r"C:\Users\Renata\Documents\Faculdade\pdi\Trabalho Final\DB1_B"
    path_imagem = select_image(start_dir=diretorio)
    if path_imagem:
        equalizar(path_imagem)
    else:
        print("Nenhuma imagem selecionada.")
