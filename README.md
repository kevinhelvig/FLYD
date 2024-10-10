# FLYD
This repository provides access to the **FLYing-spot laser thermography Dataset (FLYD)** and its instances. This work was presented in the [**Quality Control by Artificial Vision (QCAV 2023)**](https://qcav2023.sciencesconf.org/) conference, under the title [**"Laser flying-spot thermography: an open-access dataset for machine learning and deep learning"**](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12749/127491A/Laser-flying-spot-thermography--an-open-access-dataset-for/10.1117/12.3000481.short). The dataset consists in reconstructed thermal images from the recordings of laser thermography examinations. The thermal scans are performed parallel to the crack, in order to follow the crack length on the material surface. Parts examined are several metallic fatigue test specimens with various crack opening and length. More details about the experimental and recording settings are given in the QCAV 2023 proceeding.

- **FLYD-C instance: binary classification task dataset**, between crack and uncrack images. The dataset contains 891 reconstructed thermal images for training, and 286 for evaluation.
- **FLYD-D instance: crack detection+localization task.**	
- **FLYD-S instance: crack segmentation task. In construction** :building_construction:	

<!-- Align images to the center -->
<figure>
<p align="center">
  <img src="illustrations\example_scan1.gif" alt="Alt Text 1" width="250" height="250">
  <img src="illustrations\example_scan2.gif" alt="Alt Text 2" width="250" height="250">
  <img src="illustrations\example_scan3.gif" alt="Alt Text 3" width="250" height="250">
</p>
<figcaption style="text-align: center; font-style: italic;"> <p> <i> Examples of thermal recording. These recordings are then converted in reconstructed thermal images. </i> </p> </figcaption>
</figure>

# Road-Map :construction:

- __[2024.10.10]__ Add a sub-repo for the mono-spectrum experiments presented at the [__SPIE Thermosense: Thermal Infrared Applications XLVI__](https://doi.org/10.1117/12.3013537) conference. See [this folder](https://github.com/kevinhelvig/FLYD/tree/main/Thermosense_experiments). :thermometer:
-  We firstly provide **the original dataset for classification, FLYD-C**. A training code is added.
-  **[2023.10.16] Annotations for crack localization added**: . See FLYD-D repository for annotations. 
-  **[2023.11.20] Download links for the subsampled thermal recordings added** :robot:: See section **Subsampled thermal recordings**.
-  We plan to **add more samples in both datasets in the longer run**.

# Subsampled thermal recordings :movie_camera:
You can download the subsampled thermal recordings using the following links. Provided thermal recordings are subsampled from 100 Hz to 50 Hz, in order to reduce the weight of the whole dataset. Files are stored in .avi file format. 
- [Download the parallel scan recordings](https://zenodo.org/records/10160129/files/FLYD_Movies.zip?download=1) : scans used for the presented datasets and deep neural nets trainings.
- [Several conventional scan recordings](https://zenodo.org/records/10160129/files/FLYD-perp.zip?download=1) are provided : these scans pass accross the examined defect, following the most conventional flying-spot approach (difference between a forward and a backward scan passing through the defect).

# Benchmarking some architectures (classification task) :memo:
The following results correspond to the classification scores presented during the QCAV 2023 conference. A large panel of architectures are compared (both convolution and attention based architectures). **The different metrics are evaluated on the test-set of FLYD-C**.

- Convolution-based architectures:

| Model           | Initialization | Accuracy |  F1-Score | Precision | Recall |
|-----------------|----------------|----------|-----------|-----------|--------|
| VGG13           | Random         | 0.839    | 0.840     | 0.910     | 0.781  |
|                 | Pre-trained    | 0.902    | 0.902     | 0.985     | 0.832  |
| VGG16           | Random         | 0.755    | 0.713     | 0.977     | 0.561  |
|                 | Pre-trained    | 0.811    | 0.795     | 0.977     | 0.677  |
| ConvNext        | Random         | -        | -         | -         | -      |
|                 | Pre-trained    | 0.989    | 0.990     | 0.981     | 0.999  |


- Attention-based architectures: 

| Model           | Initialization | Accuracy |  F1-Score | Precision | Recall |
|-----------------|----------------|----------|-----------|-----------|--------|
| ViT-B           | Random         | 0.867    | 0.881     | 0.850     | 0.916  |
|                 | Pre-trained    | 0.986    | 0.987     | 0.975     | 0.999  |
| ViT-L           | Random         | 0.843    | 0.862     | 0.819     | 0.910  |
|                 | Pre-trained    | 0.990    | 0.990     | 0.981     | 1.00   |
| Swin            | Random         | -        | -         | -         | -      |
|                 | Pre-trained    | 0.989     | 0.990    | 0.987     | 0.993  |
| CaiT            | Random         | -        | -         | -         | -      |
|                 | Pre-trained    | 0.989     | 0.990    | 0.981     | 0.999  |

# Train a classifier with the proposed code :rocket:	
Requirements: **pytorch, scikit-learn, timm library (pytorch image model)** (latest versions install through conda or pip should work).
- Follow the instructions on [Pytorch's official website](https://pytorch.org/) to properly install PyTorch based on your specifications.
- Pip commands to install [scikit-learn](https://scikit-learn.org/stable/) and [timm](https://timm.fast.ai/):
  ```
  pip install scikit-learn, timm 
  ```

You can directly clone this github repository and launch the python script <strong> train.py </strong> on the command line, with specific arguments. You should decompress the archive containing the dataset before (FLYD-C), into this main directory in order to run the provided code.

Here's a brief explanation of the command-line arguments
- --training_rep: path to the training dataset.
- --test_rep: path to the test dataset.
- --model: name of the timm model to use (default: vgg13)
- --pretrained: load a pretrained model (better performance, thanks to transfer learning) or not (less performance and/or longer training duration). (default: True)  
- --num_epochs: number of epochs to train the model. (default: 10, adapted for a pre-trained model) 
- --batch_size: number of input images feeding the model at each iteration, during one epoch. You can increase it, depending on your hardware specifications. (default: 16) 
- --learning_rate: learning rate for the optimizer (default: 1e-4).
- --output_dir: path to the directory to save the log and model state dict (default: None).

You can customize the script by changing the values of the command-line arguments as needed. More details about the training (augmentations, metrics evaluated...) are given in the proceeding from the conference. 

## Cite
If the dataset is used for academic purpose, please consider citing our work: 

```bibtex
@incollection{Helvig2023Jul,
	author = {Helvig, K. and Trouve-Peloux, P. and Gaverina, L. and Roche, J.-M. and Abeloos, B. and Pradere, C.},
	title = {{Laser flying-spot thermography: an open-access dataset for machine learning and deep learning}},
	booktitle = {{Proceedings Volume 12749, Sixteenth International Conference on Quality Control by Artificial Vision}},
	journal = {Sixteenth International Conference on Quality Control by Artificial Vision},
	volume = {12749},
	pages = {334--340},
	year = {2023},
	month = jul,
	publisher = {SPIE},
	doi = {10.1117/12.3000481}
}
```
