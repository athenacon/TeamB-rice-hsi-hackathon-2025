import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path 
from helpers import * 
import os

#DATA_DIR = Path("data") / "RGB and VIS-NIR HSI Data for 90 Rice Seed Varieties"
DATA_DIR = Path("/uoa/scratch/shared/2025_hackathon/RGB_and_VIS-NIR_HSI_data_for_90_rice_seed_varieties/RGB_and_VIS-NIR_HSI_data_for_90_rice_seed_varieties")
df = HSIDataSetDataFrame(pd.read_csv(DATA_DIR / "index.csv"))

# Create output directory relative to our current script location
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = SCRIPT_DIR / "output"

# Create directory if it doesnt exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

img_num = 13
channel_num = 115
plt.imshow(df.images[img_num].hsi[:, :, channel_num])
plt.title("HSI Raw Image and \nlocation of the spectralon tile")

# draw a red line on row 625 and 675 to show the region of spectralon tile
first, second = df.images[img_num].white_rows
plt.axhline(first, color="red")
plt.axhline(second, color="red")
print(f"White refernece is take from the spectralon tile rows: {first} to {second}")
plt.savefig(OUTPUT_DIR / 'spectralon_plot.png')  # Save the figure


# add figure with 2 subplots
fig, axs = plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.imshow(df.images[img_num].hsi[:, :, channel_num])
plt.title("Raw HSI")
plt.savefig(OUTPUT_DIR / "Raw HSI.png")  

plt.subplot(1, 2, 2)
plt.imshow(df.images[img_num].hsi_calibrated[:, :, channel_num])
plt.title("Calibrated HSI")
plt.savefig(OUTPUT_DIR / "Calibrated HSI.png")  

plt.figure()
plt.imshow(df.images[img_num].rgb)
plt.title("RGB Image")
plt.savefig(OUTPUT_DIR / "RGB Image.png")  

plt.figure(figsize=(10, 10))
plt.imshow(df.images[img_num].hsi_calibrated[:, :, channel_num])
plt.title("Calibrated HSI")
plt.savefig(OUTPUT_DIR / "Calibrated HSI.png")  

# put a star mark on location
row, col = 400, 95
plt.axhline(row, color="red")
plt.axvline(col, color="red")
plt.scatter(
    col, row, marker="*", color="red", s=100
)  # note coordinates is swapped on images (y, x)

plt.figure()
plt.plot(df.images[img_num].wavelengths, df.images[img_num].hsi_calibrated[row, col, :])
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title(f"Reflectance Spectrum at location ({row},{col})")
plt.grid(True)
plt.show()
plt.savefig(OUTPUT_DIR / f"Reflectance Spectrum at location ({row},{col}).png")  
