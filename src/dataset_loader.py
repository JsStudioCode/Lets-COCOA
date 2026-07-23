import os
from pathlib import Path
import pandas as pd

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
RGB_PATH = BASE_DIR / "data" / "rgb"
THERMAL_PATH = BASE_DIR / "data" / "thermal"

# Class names
CLASSES = ["CPB", "OR", "R", "UR"]


def verify_dataset():
    print("=" * 60)
    print("COCOA POD DATASET VERIFICATION")
    print("=" * 60)

    print("BASE_DIR      :", BASE_DIR)
    print("RGB_PATH      :", RGB_PATH)
    print("THERMAL_PATH  :", THERMAL_PATH)
    print()

    total_rgb = 0
    total_thermal = 0

    for cls in CLASSES:
        rgb_folder = RGB_PATH / cls
        thermal_folder = THERMAL_PATH / cls

        print(f"Checking class: {cls}")
        print(f"RGB Folder     : {rgb_folder}")
        print(f"Exists         : {rgb_folder.exists()}")

        if rgb_folder.exists():
            rgb_images = list(rgb_folder.glob("*"))
            rgb_count = len(rgb_images)
        else:
            rgb_count = 0

        print(f"Thermal Folder : {thermal_folder}")
        print(f"Exists         : {thermal_folder.exists()}")

        if thermal_folder.exists():
            thermal_images = list(thermal_folder.glob("*"))
            thermal_count = len(thermal_images)
        else:
            thermal_count = 0

        total_rgb += rgb_count
        total_thermal += thermal_count

        print(f"RGB Images     : {rgb_count}")
        print(f"Thermal Images : {thermal_count}")
        print("-" * 60)

    print("\nTOTAL RGB      :", total_rgb)
    print("TOTAL THERMAL  :", total_thermal)


if __name__ == "__main__":
    verify_dataset()