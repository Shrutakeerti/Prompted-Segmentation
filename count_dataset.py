import os

# CHANGE THIS TO YOUR DATASET ROOT
base_path = r"D:\origin"

# dataset folders (edit if needed)
datasets = [
    r"cracks.v1i.yolov8",
    r"Drywall_dataset_fixed"
]

splits = ["train", "val", "test"]
subfolders = ["images", "labels", "bbox_images", "masks"]

def count_files(folder):
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ])

print("\n================ DATASET SUMMARY ================\n")

for dataset in datasets:

    dataset_path = os.path.join(base_path, dataset)

    print(f"\n📁 DATASET: {dataset}")
    print("-" * 50)

    for split in splits:

        print(f"\n🔹 {split.upper()}")

        split_path = os.path.join(dataset_path, split)

        if not os.path.exists(split_path):
            print("   ❌ Split not found")
            continue

        total = 0

        for sub in subfolders:

            sub_path = os.path.join(split_path, sub)

            count = count_files(sub_path)

            print(f"   {sub:15}: {count}")

            total += count

        print(f"   TOTAL FILES    : {total}")

print("\n================================================\n")