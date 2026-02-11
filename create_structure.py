#!/usr/bin/env python3
# create_structure.py
# Create subfolders and placeholder files for the modular geometric_extraction project

import os

STRUCTURE = {
    "": ["main.py"],
    "io": ["loader.py", "saver.py"],
    "primitives": ["plane_fitting.py", "cylinder_fitting.py", "sphere_fitting.py", "utils.py"],
    "geometry": ["intersections.py", "hulls.py", "deduplication.py", "edges_vertices.py", "filtering.py"],
    "visualization": ["viewer.py", "colors.py"],
    "utils": ["config.py", "logging_utils.py", "timing.py"]
}

def create_structure(structure):
    base = os.getcwd()
    for folder, files in structure.items():
        path = os.path.join(base, folder)
        os.makedirs(path, exist_ok=True)
        for f in files:
            fpath = os.path.join(path, f)
            if not os.path.exists(fpath):
                with open(fpath, "w") as fp:
                    fp.write(f"# {f}\n")
                print(f"Created {os.path.relpath(fpath, base)}")

if __name__ == "__main__":
    create_structure(STRUCTURE)
    print("\n✅ Subfolder structure created inside current directory.")
