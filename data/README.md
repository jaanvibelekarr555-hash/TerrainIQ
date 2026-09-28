# Dataset

This project uses SemanticKITTI / KITTI LiDAR data.

The dataset itself is not stored in GitHub.

Expected local structure:

data/
└── semantic_kitti/
    └── sequences/
        └── 00/
            ├── velodyne/
            │   └── 000000.bin
            └── labels/
                └── 000000.label

Important:
- Do not commit dataset files to Git.
- The data/ directory is already ignored by .gitignore.
- The .bin and .label files must have matching frame numbers.