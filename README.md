# Rice Seed Varieties Hackathon 2025 

## Challenge Overview

Welcome to the 2025 Rice Seed Varieties Hackathon! 

This is an open-ended challenge where teams will develop innovative models and analyses using hyperspectral imaging data of rice seed varieties.

## Dataset

We are using the **RGB and VIS/NIR Hyperspectral Imaging Data for 90 Rice Seed Varieties** dataset:
- **Dataset URL**: [https://zenodo.org/records/3241923](https://zenodo.org/records/3241923)
- **Description**: This comprehensive dataset contains RGB and hyperspectral imaging data for 90 different rice seed varieties, enabling advanced analysis and classification.

A Small version of the dataset to test on can be found here:

- https://tinyurl.com/hsi-partial

> **Important Note**: When running any version of the script (Pyhon or Jupyter), the data directory must **ALSO** be updated in **'helpers.py'**. The default path is set up for Maxwell and will need swapping when running locally.

## Accessing the Dataset

The dataset has been pre-downloaded onto Maxwell and is available in a shared folder:

```
/uoa/scratch/shared/2025_hackathon/RGB_and_VIS-NIR_HSI_data_for_90_rice_seed_varieties/RGB_and_VIS-NIR_HSI_data_for_90_rice_seed_varieties
```

## Computing Resources

You will have access to Maxwell's GPU nodes for your model development and analysis.

### Example SLURM Script
Attached is an example slurm script "runScript.sh"

Running this will then run the hsipy.py file which will store its ouputs inthe "output" folder.

The job should take a few seconds to run.

> **Important Note**: The `--nodelist=agpu004` line will run the job on gpu4. Change the number to use different GPU cards and avoid queueing for the same resource.

## Getting Started

1. Copy the example script and modify it for your needs
2. The script will run `hsipy.py` and store outputs in the "output" folder
3. Expected runtime: A few seconds for the example script

## Challenge Goals

Your team's objective is to:

1. **Explore the Dataset**: Investigate the hyperspectral data to identify interesting patterns
2. **Develop a Model**: Create an innovative approach to analyze or classify the rice seed varieties
3. **Visualize Results**: Generate some visualizations that demonstrate your findings
4. **Prepare a Presentation**: Document your methodology and results for Friday's presentation

## Submission Requirements

Your final submission should include:

1. **Code Repository**: Well-documented code for your analysis and models
2. **Results Summary**: Key findings and visualizations
3. **Presentation**: A 15-minute presentation explaining your approach and discoveries
4. **Technical Documentation**: Methods, challenges, and potential applications

## Evaluation Criteria

Teams will be evaluated on:

- **Innovation**: Originality of approach and techniques
- **Technical Merit**: Effectiveness and sophistication of models/algorithms
- **Insights**: Quality and relevance of discoveries from the data
- **Presentation**: Clarity and engagement of the final presentation

## Good Luck!

We look forward to seeing your creative approaches and innovative solutions! If you have any questions, please reach out to the hackathon Christos or Matt.

---
