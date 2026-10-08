from pathlib import Path
import numpy as np
import csv

from src.datasets.spore_dataset import criar_splits

BASE_DIR = Path(r"C:\GitHub_CiaAgro\dataset_huggingface\filtered_cropped_images")

SAVE_DIR = Path(r"C:\GitHub_CiaAgro\dataset_huggingface")

NORMAL_DIR = BASE_DIR / "Normal"
BLACK_DIR = BASE_DIR / "Segmented"

OUTPUT_DIR = SAVE_DIR / "splits"

HF_NORMAL_ROOT = Path("filtered_cropped_images") / "Normal"
HF_BLACK_ROOT = Path("filtered_cropped_images") / "Segmented"

RANDOM_STATE = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def collect_dataset(root_dir):

    samples = []

    class_dirs = sorted(
        [
            p for p in root_dir.iterdir()
            if p.is_dir()
        ]
    )

    if not class_dirs:
        raise RuntimeError(
            f"No class directories found in:\n{root_dir}"
        )

    for label, class_dir in enumerate(class_dirs):

        subdirs = sorted(
            [
                p for p in class_dir.iterdir()
                if p.is_dir()
            ]
        )

        if not subdirs:

            image_paths = sorted(
                [
                    p for p in class_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
            )

            for image_path in image_paths:

                samples.append({
                    "id": image_path.stem,
                    "class_name": class_dir.name,
                    "label": label,
                    "path": image_path
                })

        else:

            for subdir in subdirs:

                image_paths = sorted(
                    [
                        p for p in subdir.iterdir()
                        if p.is_file()
                        and p.suffix.lower() in IMAGE_EXTENSIONS
                    ]
                )

                for image_path in image_paths:

                    samples.append({
                        "id": image_path.stem,
                        "class_name": class_dir.name,
                        "label": label,
                        "path": image_path
                    })

    return samples


def normal_to_hf_path(local_path):

    local_path = Path(local_path)

    relative_to_normal = local_path.relative_to(NORMAL_DIR)

    return HF_NORMAL_ROOT / relative_to_normal


def black_to_hf_path(local_path):

    local_path = Path(local_path)

    relative_to_black = local_path.relative_to(BLACK_DIR)

    return HF_BLACK_ROOT / relative_to_black


def create_black_index(black_samples):
    """
    Create an index:

        image_id -> black image information
    """

    index = {}

    for sample in black_samples:

        image_id = sample["id"]

        if image_id in index:
            raise ValueError(
                f"Duplicate image ID in segmented dataset: "
                f"{image_id}"
            )

        index[image_id] = sample

    return index


def convert_to_black_paths(
    normal_files,
    normal_labels,
    black_index
):

    black_files = []

    for normal_path, label in zip(
        normal_files,
        normal_labels
    ):

        normal_path = Path(normal_path)

        image_id = normal_path.stem

        if image_id not in black_index:

            raise FileNotFoundError(
                "\nCould not find corresponding segmented "
                f"image for:\n{normal_path}\n"
                f"Image ID: {image_id}"
            )

        black_sample = black_index[image_id]

        if black_sample["label"] != label:

            raise ValueError(
                f"Label mismatch for image {image_id}: "
                f"Normal={label}, "
                f"Black={black_sample['label']}"
            )

        black_files.append(
            str(black_sample["path"])
        )

    return black_files


def convert_normal_split_to_hf_paths(files):

    return [
        str(normal_to_hf_path(path)).replace("\\", "/")
        for path in files
    ]


def convert_black_split_to_hf_paths(files):

    return [
        str(black_to_hf_path(path)).replace("\\", "/")
        for path in files
    ]


def save_split(
    output_dir,
    train_files,
    train_labels,
    val_files,
    val_labels,
    test_files,
    test_labels
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_dir / "train_files.npy",
        np.array(train_files, dtype=str)
    )

    np.save(
        output_dir / "train_labels.npy",
        np.array(train_labels, dtype=np.int64)
    )

    np.save(
        output_dir / "val_files.npy",
        np.array(val_files, dtype=str)
    )

    np.save(
        output_dir / "val_labels.npy",
        np.array(val_labels, dtype=np.int64)
    )

    np.save(
        output_dir / "test_files.npy",
        np.array(test_files, dtype=str)
    )

    np.save(
        output_dir / "test_labels.npy",
        np.array(test_labels, dtype=np.int64)
    )


def save_metadata(
    output_dir,
    train_files,
    train_labels,
    val_files,
    val_labels,
    test_files,
    test_labels
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = output_dir / "split_metadata.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "image_id",
            "label",
            "split",
            "path"
        ])

        for split_name, files, labels in [
            ("train", train_files, train_labels),
            ("val", val_files, val_labels),
            ("test", test_files, test_labels)
        ]:

            for path, label in zip(files, labels):

                path = Path(path)

                writer.writerow([
                    path.stem,
                    label,
                    split_name,
                    str(path).replace("\\", "/")
                ])


def print_statistics(
    name,
    train_files,
    train_labels,
    val_files,
    val_labels,
    test_files,
    test_labels
):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Train: {len(train_files):,}"
    )

    print(
        f"Val:   {len(val_files):,}"
    )

    print(
        f"Test:  {len(test_files):,}"
    )

    print(
        f"Total: {len(train_files) + len(val_files) + len(test_files):,}"
    )

    print("\nClass distribution:")

    for split_name, labels in [
        ("Train", train_labels),
        ("Val", val_labels),
        ("Test", test_labels)
    ]:

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        print(f"\n{split_name}:")

        for label, count in zip(unique, counts):

            print(
                f"  Class {label}: {count:,}"
            )


def main():

    print("=" * 60)
    print("CREATING DATASET SPLITS")
    print("=" * 60)

    print("\nNormal dataset:")
    print(NORMAL_DIR)

    print("\nSegmented Black dataset:")
    print(BLACK_DIR)

    # Load Normal

    print("\nCollecting Normal images...")

    normal_samples = collect_dataset(
        NORMAL_DIR
    )

    print(
        f"Found {len(normal_samples):,} Normal images."
    )

    # Load Segmented Black

    print("\nCollecting Segmented Black images...")

    black_samples = collect_dataset(
        BLACK_DIR
    )

    print(
        f"Found {len(black_samples):,} Segmented Black images."
    )

    normal_ids = {
        sample["id"]
        for sample in normal_samples
    }

    black_ids = {
        sample["id"]
        for sample in black_samples
    }

    print("\nChecking image correspondence...")

    missing_black = normal_ids - black_ids
    missing_normal = black_ids - normal_ids

    if missing_black:

        print(
            f"\nERROR: {len(missing_black)} Normal images "
            "have no corresponding Black image."
        )

        print("\nExamples:")

        for image_id in sorted(missing_black)[:10]:
            print(" ", image_id)

        raise RuntimeError(
            "Normal/Black datasets do not correspond."
        )

    if missing_normal:

        print(
            f"\nERROR: {len(missing_normal)} Black images "
            "have no corresponding Normal image."
        )

        print("\nExamples:")

        for image_id in sorted(missing_normal)[:10]:
            print(" ", image_id)

        raise RuntimeError(
            "Normal/Black datasets do not correspond."
        )

    print("✓ All image IDs correspond.")

    print("\nCreating stratified master split...")

    (
        train_files,
        train_labels,
        val_files,
        val_labels,
        test_files,
        test_labels
    ) = criar_splits(
        NORMAL_DIR
    )

    black_index = create_black_index(
        black_samples
    )

    black_train_files = convert_to_black_paths(
        train_files,
        train_labels,
        black_index
    )

    black_val_files = convert_to_black_paths(
        val_files,
        val_labels,
        black_index
    )

    black_test_files = convert_to_black_paths(
        test_files,
        test_labels,
        black_index
    )

    train_files = convert_normal_split_to_hf_paths(
        train_files
    )

    val_files = convert_normal_split_to_hf_paths(
        val_files
    )

    test_files = convert_normal_split_to_hf_paths(
        test_files
    )

    black_train_files = convert_black_split_to_hf_paths(
        black_train_files
    )

    black_val_files = convert_black_split_to_hf_paths(
        black_val_files
    )

    black_test_files = convert_black_split_to_hf_paths(
        black_test_files
    )

    normal_output = (
        OUTPUT_DIR / "normal"
    )

    black_output = (
        OUTPUT_DIR / "segmented_black"
    )

    save_split(
        normal_output,
        train_files,
        train_labels,
        val_files,
        val_labels,
        test_files,
        test_labels
    )

    save_split(
        black_output,
        black_train_files,
        train_labels,
        black_val_files,
        val_labels,
        black_test_files,
        test_labels
    )

    save_metadata(
        normal_output,
        train_files,
        train_labels,
        val_files,
        val_labels,
        test_files,
        test_labels
    )

    print_statistics(
        "NORMAL",
        train_files,
        train_labels,
        val_files,
        val_labels,
        test_files,
        test_labels
    )

    print_statistics(
        "SEGMENTED BLACK",
        black_train_files,
        train_labels,
        black_val_files,
        val_labels,
        black_test_files,
        test_labels
    )


    print("\nExample Normal relative path:")
    print(train_files[0])

    print("\nExample Segmented Black relative path:")
    print(black_train_files[0])

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"\nSplit files saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()