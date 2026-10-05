import shutil
from pathlib import Path


def copy_and_rename_images(src_dir, dest_dir, base_name):

    src = Path(src_dir)
    dest = Path(dest_dir)

    dest.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    counter = 1
    
    for file in src.iterdir():

        if file.is_file() and file.suffix.lower() in valid_extensions:

            extension = file.suffix
            
            new_name = f"{base_name}_{counter:05d}{extension}"
            dest_path = dest / new_name
            
            shutil.copy2(file, dest_path)
            
            counter += 1
            
    print(f"\nDone! {counter - 1} images were copied and renamed.")


src_dir = r"C:\Analise_Esporos_2026\Dataset_Recortado_4_classes\phakopsora"     # Source directory adress
dest_dir = r"C:\GitHub_CiaAgro\CiaAgro_FungalSpores\data\dataset_huggingface\cropped_images\phakops"      # Destination directory adress
base_name = "Phakopsora_pachyrhizi"     # Fungal spore class: "Phakopsora_pachyrhizi", "Microsphaera_diffusa", "Peronospora_manshurica"

# Execute the function
copy_and_rename_images(src_dir, dest_dir, base_name)
