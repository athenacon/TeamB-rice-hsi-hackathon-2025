import matplotlib.pyplot as plt
import pandas as pd
from helpers import *
import os
import h5py

import numpy as np
from scipy.ndimage import  center_of_mass
from skimage import filters, morphology, measure, color


df = HSIDataSetDataFrame(pd.read_csv(DATA_DIR / "index.csv"))

unordered_path = Path("hyperrice/unordered")


# Make a place for this data to go
if (not os.path.isdir("hyperrice/train")):
    os.makedirs("hyperrice/train")
    os.makedirs("hyperrice/val")
    os.makedirs()
    # os.makedirs("hyperrice/testing")


for i in range(len(df)):
    print(f"Processing image {i}")
    
    img_num = i
    threshold = 0.08
    min_area = 300

    image = df.images[img_num].hsi_calibrated[:, :, 110]
    gray_image = df.images[img_num].hsi_calibrated[:, :, 110]

    binary_mask = image[:580] > threshold
    dialated = morphology.dilation(binary_mask, morphology.disk(3))
    eroded = morphology.erosion(dialated, morphology.disk(3))

    # Perform morphological closing to join nearby regions
    closed = morphology.closing(
        eroded, morphology.disk(3)
    )  # Adjust disk size as needed

    # Remove small objects (regions with area < 100 pixels)
    filtered = morphology.remove_small_objects(closed, min_size=min_area)

    # use regioprops to get the area of each region
    regions = measure.regionprops(measure.label(filtered))
    areas = [region.area for region in regions]
    aspect_ratios = [
        region.major_axis_length / region.minor_axis_length for region in regions
    ]

    # Label connected components
    labeled_regions = measure.label(filtered)

    mask = np.zeros_like(labeled_regions, dtype=bool)
    # Loop through each region and create a mask for regions with aspect ratio <= 5
    for region in regions:
        aspect_ratio = region.major_axis_length / region.minor_axis_length
        if aspect_ratio > 5:  # Remove regions with aspect ratio > 5
            labeled_regions[labeled_regions == region.label] = 0
    


    def sort_regions_grid(labeled_regions, y_threshold=5):
        region_labels = np.unique(labeled_regions)
        region_labels = region_labels[region_labels > 0]  # Exclude background (0)

        # Compute centroids
        centroids = np.array(
            center_of_mass(labeled_regions, labels=labeled_regions, index=region_labels)
        )

        # Sort centroids by Y-coordinates
        sorted_y_indices = np.argsort(centroids[:, 0])
        centroids = centroids[sorted_y_indices]
        region_labels = region_labels[sorted_y_indices]

        # Cluster regions into rows based on Y proximity
        rows = []
        current_row = [region_labels[0]]

        for i in range(1, len(region_labels)):
            if centroids[i, 0] - centroids[i - 1, 0] > y_threshold:
                rows.append(current_row)
                current_row = []
            current_row.append(region_labels[i])

        rows.append(current_row)  # Append last row

        # Sort each row by X-coordinates
        sorted_labels = []
        for row in rows:
            row_centroids = centroids[np.isin(region_labels, row)]
            sorted_x_indices = np.argsort(row_centroids[:, 1])
            sorted_labels.extend(np.array(row)[sorted_x_indices])

        # Renumber labels
        new_labels = {old: new + 1 for new, old in enumerate(sorted_labels)}
        renumbered_regions = np.copy(labeled_regions)
        for old_label, new_label in new_labels.items():
            renumbered_regions[labeled_regions == old_label] = new_label

        return renumbered_regions


    # Usage
    renumbered_regions = sort_regions_grid(labeled_regions, y_threshold=25)

    def extract_masks(labeled_regions):
        region_labels = np.unique(labeled_regions)
        region_labels = region_labels[region_labels > 0]  # Exclude background (0)

        masks = {
            label: (labeled_regions == label).astype(np.uint8) for label in region_labels
        }
        return masks  # Dictionary {label: mask}


    masks = extract_masks(renumbered_regions)

    def extract_bounded_seed(specimg, masks, renumbered_regions, region_id, bbox_size = 30):
        res = specimg * masks[region_id][:, :, np.newaxis]
        centroid = center_of_mass(renumbered_regions == region_id)

        # Get a bounding box
        bbox = (np.add(centroid, -bbox_size).astype(int), np.add(centroid, bbox_size).astype(int))
        # print(bbox)
        # print(centroids[1])
        # Crop it
        res = res[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :]

        return res


    ##
    ## Our part
    ##

    regions = np.max(renumbered_regions)

    # Over every region save it and the annotation
    for r in range(regions):
        # Create a new h5py for this bbox
        with h5py.File(unordered_path / f"img_{i}_region_{r}.h5", "w") as f:
            res = extract_bounded_seed(img, masks, renumbered_regions, region_id=r)
            f.create_dataset("image", compression="lzf", data=res)
            f.create_dataset("metadata", compression="lzf", data=df.images[img_num].metadata)
            f.create_dataset("short_name", compression="lzf", data=df.images[img_num].metadata["Species Short Name"])
