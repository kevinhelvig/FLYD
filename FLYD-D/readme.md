# FLYD-D: Detection and localization pretraining dataset for Flying-spot thermography
<!-- Align images to the center -->
<figure>
<p align="center">
  <img src=".\realtime_FLYD2.gif" alt="Alt Text 1" width="250" height="250">
  <img src=".\realtime_FLYD3.gif" alt="Alt Text 2" width="250" height="250">
  <img src=".\realtime_FLYD4.gif" alt="Alt Text 2" width="250" height="250">
</p>
<figcaption style="text-align: center; font-style: italic;"> <p> <i> Examples of thermal recording with frame per frame crack localization. The model is trained on reconstructed thermal images from FLYD. </i> </p> </figcaption>
</figure>

This sub-directory contains the **annotations for crack detection/localization on metallic parts through flying-spot laser infrared thermography**. This dataset enables the training of object localization architectures, such as [**Faster RCNN**](https://arxiv.org/abs/1506.01497) and [**DETR**](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf). The reconstructed thermal images annotated here are identical to those of FLYD-C. All the images are manually annotated using the [**Labelstudio**](https://labelstud.io/) software. 
The images and annotations are contained in the following archives:
- **FLYD-D_coco.zip**: Archive containing annotations and images in **MS-COCO 2017 format**. This format is typically directly usable by a Coco dataloader (via Pytorch/TensorFlow).
- **FLYD-D_yolo.zip**: Archive containing annotations and images in **YOLO format**. Allows for the training of architectures from the YOLO family, via the [**Ultralytics framework**](https://docs.ultralytics.com/models/).
