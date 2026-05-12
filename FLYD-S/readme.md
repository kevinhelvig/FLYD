# FLYD-S: Semantic Segmentation dataset for flying-spot laser thermography

This sub-directory contains the **annotations for pixel-level crack segmentation on metallic parts through flying-spot laser infrared thermography**. This dataset provides pixel-wise ground truth to identify the exact shape and morphology of cracks, going beyond bounding box localization.

This dataset enables the training of semantic segmentation architectures, such as [**U-Net**](https://arxiv.org/abs/1505.04597) and [**DeepLabV3+**](https://arxiv.org/abs/1802.02611). The thermal images are manually annotated using the [**CVAT**](https://www.cvat.ai/) software. 

The images and segmentation masks are contained in the following archive:

- **FLYD-S.zip**: Archive containing 701 pairs of thermal images and their corresponding binary masks.
    - **Format**: Standard image/mask pairs (JPG/PNG).
    - **Resolution**: 120x120 pixels.
    - **Organization**: The archive follows a standard deep learning structure with **train**, **val**, and **test** splits. Each split contains an `images/` folder and a `masks/` folder with matching filenames for seamless integration with PyTorch dataloaders.
