import os
import random
import shutil

# Base path of your project
BASE_PATH = r"C:\Users\Molly\project_s.a.f.e"

# Source (current data)
SOURCE_PATH = os.path.join(BASE_PATH, "Testing_data")

# Destination folders
TRAIN_PATH = os.path.join(BASE_PATH, "Training_data")
TEST_PATH = os.path.join(BASE_PATH, "Testing_data_split")

# Split ratio
SPLIT_RATIO = 0.6

# Supported audio formats
EXTENSIONS = ('.wav', '.mp3')


def split_class(class_name):
    src = os.path.join(SOURCE_PATH, class_name)
    train_dest = os.path.join(TRAIN_PATH, class_name)
    test_dest = os.path.join(TEST_PATH, class_name)

    if not os.path.exists(src):
        print(f"Skipping '{class_name}' - folder not found")
        return

    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    files = [f for f in os.listdir(src) if f.lower().endswith(EXTENSIONS)]

    if len(files) == 0:
        print(f"No files found in '{class_name}'")
        return

    random.shuffle(files)

    split_index = int(len(files) * SPLIT_RATIO)

    train_files = files[:split_index]
    test_files = files[split_index:]

    for f in train_files:
        shutil.move(os.path.join(src, f), os.path.join(train_dest, f))

    for f in test_files:
        shutil.move(os.path.join(src, f), os.path.join(test_dest, f))

    print(f"{class_name} -> Train: {len(train_files)}, Test: {len(test_files)}")


def main():
    for cls in ["human", "ai"]:
        split_class(cls)

    print("Split completed successfully.")


if __name__ == "__main__":
    main()