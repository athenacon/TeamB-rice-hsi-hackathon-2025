from pathlib import Path
import tqdm
import numpy as np
import shutil
import tqdm
import os

# Split 20% of files into validation
SPLIT=0.2

# Get the images we have, and shuffle them deterministically
rng = np.random.default_rng(seed=42)


unordered = Path("unordered")
train     = Path("train")
val       = Path("val")

images    = os.listdir(unordered)
# Sort for deterministic, it actualy makes it 250% faster on maxwell too?
images    = sorted(images)
img_count = len(images)
rng.shuffle(images)


for i, image in enumerate(tqdm.tqdm(images)):
    # If it's in the range of validation then copy to validation
    if ((i / img_count) > SPLIT):
        shutil.copyfile( unordered/image, val / image )
    else:
        shutil.copyfile( unordered/image, train / image )


