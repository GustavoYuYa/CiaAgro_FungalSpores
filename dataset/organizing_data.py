import shutil
from pathlib import Path


def organize_dataset_by_class(src_dir, dest_dir):
    src = Path(src_dir)
    dest = Path(dest_dir)
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    processed_count = 0
    
    for txt_file in src.glob('*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
            
        if not first_line:
            continue  # Skip if the file is empty
            
        # Extract the class name by separating text from floating point numbers
        parts = first_line.split()
        class_name_parts = []
        
        for part in parts:
            try:
                # If the part can be converted to float, we hit the coordinates
                float(part)
                break
            except ValueError:
                # If it's not a number, it's part of the class name
                class_name_parts.append(part)
        
        if not class_name_parts:
            class_name = "Unknown_Class"
        else:
            class_name = "_".join(class_name_parts)
            
        class_dir = dest / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        image_file = None
        for ext in valid_extensions:
            potential_image = src / (txt_file.stem + ext)
            if potential_image.exists():
                image_file = potential_image
                break
        
        # If the image is found, copy both files to the class directory
        if image_file:
            shutil.copy2(txt_file, class_dir / txt_file.name)
            shutil.copy2(image_file, class_dir / image_file.name)
    
            print(f"Organized: {txt_file.stem} -> {class_name}/")
            processed_count += 1
        else:
            print(f"Warning: Corresponding image not found for {txt_file.name}")
            
    print(f"\nDone! {processed_count} image/text pairs were organized into classes.")


source_directory = r"C:\Users\yudibioinfo\Downloads\Anotações-20260909T163402Z-1-001\Anotações"
destination_directory = r"C:\GitHub_CiaAgro\CiaAgro_FungalSpores\data\whole_slide"

organize_dataset_by_class(source_directory, destination_directory)