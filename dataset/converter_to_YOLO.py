import shutil
from pathlib import Path


def convert_all_to_yolo():

    script_dir = Path(__file__).resolve().parent
    
    data_dir = script_dir.parent
    
    whole_slide_dir = data_dir / "whole_slide"
    draw_yolo_dir = script_dir / "draw-YOLO-box-main"
    
    img_out = draw_yolo_dir / "raw_images"
    lbl_out = draw_yolo_dir / "labels"
    classes_file_path = draw_yolo_dir / "classes.txt"
    
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    processed_count = 0
    
    class_mapping = {}
    next_class_id = 0
    
    print(f"Buscando arquivos em: {whole_slide_dir}...\n")
    
    for txt_file in whole_slide_dir.rglob('*.txt'):
        if txt_file.name == 'classes.txt':
            continue
            
        image_file = None
        for ext in valid_extensions:
            potential_image = txt_file.parent / (txt_file.stem + ext)
            if potential_image.exists():
                image_file = potential_image
                break
                
        if not image_file:
            print(f"Aviso: Imagem não encontrada para {txt_file.name}. Pulando...")
            continue
            
        new_lines = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                class_name_parts = []
                coordinates = []

                for part in parts:
                    try:
                        float(part)
                        coordinates.append(part)
                    except ValueError:
                        class_name_parts.append(part)
                        
                class_name = "_".join(class_name_parts)
                
                if class_name not in class_mapping:
                    class_mapping[class_name] = next_class_id
                    print(f"Nova classe encontrada: '{class_name}' -> ID: {next_class_id}")
                    next_class_id += 1
                    
                class_id = class_mapping[class_name]

                new_line = f"{class_id} " + " ".join(coordinates)
                new_lines.append(new_line)
                
        out_txt_path = lbl_out / txt_file.name
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            for new_line in new_lines:
                f.write(new_line + "\n")
                
        shutil.copy2(image_file, img_out / image_file.name)
        processed_count += 1

    sorted_classes = sorted(class_mapping.items(), key=lambda item: item[1])
    with open(classes_file_path, 'w', encoding='utf-8') as f:
        for name, _ in sorted_classes:
            f.write(name + "\n")
            
    print(f"\nSucesso! {processed_count} pares de imagem/label copiados e formatados.")
    print(f"Arquivo de classes gerado com sucesso em: {classes_file_path}")


if __name__ == "__main__":
    convert_all_to_yolo()