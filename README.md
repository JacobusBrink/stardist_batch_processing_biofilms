# StardistOpp Batch Segmentation Pipeline for .czi Files

Takes a directory containing .czi files or .tif files and processes them through StarDist OPP post processing for segmentation. Outputs can be used for BiofilmQ single cell segmentation.


## Notes
StarDist OPP is a post-processing tool that facilitates SarDist segmentation on non-star-convex objects. Details can be found on the [github repository](https://github.com/gatoniel/merge-stardist-masks) and in their [paper](https://doi.org/10.1111/mmi.15064), which demonstrates the perfomance of the tool.


Naming Conventions, set by BiofilmQ must be followed for segmented results to work with BiofilmQ. They are detailed [here](https://drescherlab.org/data/biofilmQ/docs/usage/segmentation_import.html). Underscores `_` dilineate parameters like time, channel and position, so image names must not contain underscores.


## Installing the Python Environment (Windows):
1. Open anaconda prompt terminal and navigate to the directory location containing the .yml environment file:
```bash
cd C:\Users\Username\Documents\GitHub\stardistopp_batch_processing_biofilms
```
2. Create a new conda environment using the .yml file provided.

    If there is an NVIDIA GPU installed on your machine, install the gpu based envrionment.

    Otherwise, install the cpu based environment.

```bash
conda env create -f stardist-opp-gpu.yml
```

```bash
conda env create -f stardist-opp-cpu.yml
```



## Running the Code:

When the code is run, two fields will show in the terminal.

1. First choose whether the image files will be .czi or .tif by entering `1` or `2` and press the `Enter` key
```bash
============================================
 StarDist OPP → BiofilmQ Segmentation Tool
============================================

Select input file type:
  [1] CZI  (.czi  — Zeiss microscopy files)
  [2] TIF  (.tif / .tiff — TIFF stacks, including variable-size crops)
Enter 1 or 2:

```
2. Enter the directory folder containing the .czi image files and press the `Enter` key.
```bash
============================================
 StarDist OPP → BiofilmQ Segmentation Tool
============================================

Select input file type:
  [1] CZI  (.czi  — Zeiss microscopy files)
  [2] TIF  (.tif / .tiff — TIFF stacks, including variable-size crops)
Enter 1 or 2: 1

Enter folder path containing .tif files:
```

3. Enter the directory path of the stardist opp model and press the `Enter` key.
```bash
============================================
 StarDist OPP → BiofilmQ Segmentation Tool
============================================

Select input file type:
  [1] CZI  (.czi  — Zeiss microscopy files)
  [2] TIF  (.tif / .tiff — TIFF stacks, including variable-size crops)
Enter 1 or 2: 1

Enter folder path containing .tif files: "C:\Users\user\image_files"
Enter StarDist model folder path:
```
3. The code will now analyse each image.
## Output:
Once run, the code will create a folder for each .czi image analysed. Two images will be output here:

 ch1 - this is a tif of the original .czi

 ch2 - this is the stardistopp segmented label image.

