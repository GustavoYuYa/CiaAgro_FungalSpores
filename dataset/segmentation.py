import cv2
import numpy as np
import os


ORIGINAL_DATASET = r"D:\CiaAgro2026\dataset_huggingface\cropped_images"
FILTERED_DATASET = r"D:\CiaAgro2026\dataset_huggingface\filtered_cropped_images"

NORMAL = os.path.join(FILTERED_DATASET, "Normal");
SEGMENTED = os.path.join(FILTERED_DATASET, "Segmented");
FAILED = os.path.join(FILTERED_DATASET, "FilteredImgs");

os.makedirs(NORMAL, exist_ok = True);
os.makedirs(SEGMENTED, exist_ok = True);
os.makedirs(FAILED, exist_ok = True);

targets = {"Phakopsora_pachyrhizi": 0,
          "Peronospora_manshurica": 0,
          "Microsphaera_diffusa" : 0}

low_threshold = 30
ratio = 3           
canny_kernel_size = 3
morphological_kernel_size = 15 # Tamanho do kernel para o Fechamento Morfológico
min_area = 200   # Área mínima para ignorar sujeiras menores
max_area = 8000  # Área máxima para ignorar "monstros" que pegam a tela toda

kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphological_kernel_size, morphological_kernel_size))

for classe in sorted(os.listdir(ORIGINAL_DATASET)):

    if classe not in targets:
        continue

    ORIGINAL_CLASS = os.path.join(ORIGINAL_DATASET, classe)

    NORMAL_CLASS = os.path.join(NORMAL, classe)
    SEGMENTED_CLASS = os.path.join(SEGMENTED, classe)
    FAILED_CLASS = os.path.join(FAILED, classe)

    os.makedirs(NORMAL_CLASS, exist_ok = True)
    os.makedirs(SEGMENTED_CLASS, exist_ok = True)
    os.makedirs(FAILED_CLASS, exist_ok = True)

    for dir in sorted(os.listdir(ORIGINAL_CLASS)):

        ORIGINAL_CLASS_DIR = os.path.join(ORIGINAL_CLASS, dir)
        
        NORMAL_CLASS_DIR = os.path.join(NORMAL_CLASS, dir)
        SEGMENTED_CLASS_DIR = os.path.join(SEGMENTED_CLASS, dir)
        FAILED_CLASS_DIR = os.path.join(FAILED_CLASS, dir)

        os.makedirs(NORMAL_CLASS_DIR, exist_ok = True)
        os.makedirs(SEGMENTED_CLASS_DIR, exist_ok = True)
        os.makedirs(FAILED_CLASS_DIR, exist_ok = True)
        
        for img in sorted(os.listdir(ORIGINAL_CLASS_DIR)):

            ORIGINAL_PATH_IMG = os.path.join(ORIGINAL_CLASS_DIR, img)

            NORMAL_PATH_IMG = os.path.join(NORMAL_CLASS_DIR, img)
            SEGMENTED_PATH_IMG = os.path.join(SEGMENTED_CLASS_DIR, img)
            FAILED_PATH_IMG = os.path.join(FAILED_CLASS_DIR, img)
            
            src = cv2.imread(ORIGINAL_PATH_IMG)
            
            if src is None:
                    print(f"Erro: Imagem não encontrada - {ORIGINAL_PATH_IMG}")
            else:
                src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
                src_blur = cv2.blur(src_gray, (5,5))
        
                detected_edges = cv2.Canny(src_blur, low_threshold, low_threshold * ratio, canny_kernel_size)
                
                closed_edges = cv2.morphologyEx(detected_edges, cv2.MORPH_CLOSE, kernel_closing)

                contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                valid_contours = []

                for cnt in contours:
                    area = cv2.contourArea(cnt)

                    if min_area < area < max_area:
                        hull = cv2.convexHull(cnt)
                        valid_contours.append(hull)

                mask = np.zeros_like(src_gray)
                cv2.drawContours(mask, valid_contours, -1, 255, -1) 

                if np.max(mask) == 0:
                    cv2.imwrite(FAILED_PATH_IMG, src)
                    targets[classe] += 1

                else:
                    masked_img = cv2.bitwise_and(src, src, mask=mask)
                    cv2.imwrite(NORMAL_PATH_IMG, src)
                    cv2.imwrite(SEGMENTED_PATH_IMG, masked_img)

print(targets)
