from pathlib import Path
import tqdm
import numpy as np
import shutil
import tqdm
import os

# Split 20% of files into validation
TRAIN_SPLIT=0.8

# Get the images we have, and shuffle them deterministically
rng = np.random.default_rng(seed=42)


unordered = Path("unordered/")
train     = Path("train/")
val       = Path("val/")

if (train.exists() or val.exists()):
    raise IOError(f"Train and val must be empty. {train} exists {train.exists()}, {val} exists {val.exists()}")
else:
    train.mkdir()
    val.mkdir()

images    = os.listdir(unordered)
# Sort for deterministic, it actualy makes it much faster on maxwell too? Might just be caching
images    = sorted(images)
img_count = len(images)
rng.shuffle(images)


for i, image in enumerate(tqdm.tqdm(images)):
    # If it's in the range of validation then copy to validation
    if ((i / img_count) > TRAIN_SPLIT):
        shutil.copyfile( unordered/image, val / image )
    else:
        shutil.copyfile( unordered/image, train / image )


