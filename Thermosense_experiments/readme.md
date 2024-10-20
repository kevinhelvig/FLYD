# Thermosense conference experiments (2024) 🌡️
This folder gives minimal codes and data for the work presented at the __SPIE Defense+Commercial Sensing, Thermosense: Thermal Infrared Applications XLVI__ (April 2024) . We provide weights from Stable Diffusion fine-tuning, data amounts used for both diffusers training and crack detection using laser Flying-Spot thermography ("frame-per-frame" detection case). 

The paper : [Synthetic visible-IR images pairs generation for multi-spectral NDT using flying spot thermography and deep learning [Helvig et al]](https://doi.org/10.1117/12.3013537) 

⚠️ This folder releases __only mono-spectrum (IR) experimental data and diffusion weights for image synthesis__.


# Roadmap :building_construction:

- __[10/10/2024]__ : Folder created. Adding the different links and explanations is on the way ! :biking_man:

# Stable Diffusion: Weights and minimal script usage 🏋️🖌️

The main focus of the paper is mono (and multi) spectrum image synthesis using Stable Diffusion for data augmentation. A minimal script for sampling is added: the script can generate both cracked or sane thermal frame, depending on the model chosen, and the sentence used for the semantic guidance.
The model used for image synthesis is [__Stable Diffusion v1.5__](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Instead of the image synthesis using [Denoising Diffusion Probabilistic Models](https://github.com/lucidrains/denoising-diffusion-pytorch) [[Ho et al., 2020](https://arxiv.org/abs/2006.11239)] as in [one our previous works](https://doi.org/10.1080/17686733.2023.2266176), Stable Diffusion adds semantic guidance. This model can also be identified as a __foundation model__ which is __domain "agnostic"__ and/or can be __fine-tuned for a new and specific data domain with limited computation cost__ (such as thermal data ? 🙂). The canonical Dreambooth script for fine-tuning has been used for the training (available [here](https://huggingface.co/docs/diffusers/en/training/dreambooth)). <br>
* Link to the negative generator's weights 🏋️‍♂️ : [[Zenodo]()] <br>
* Link to the positive generator's weights 🏋️‍♀️: [[Zenodo]()] <br> 

Dependencies: __Hugging Face 🤗__ [Diffusers](https://huggingface.co/docs/diffusers/index) and [Transformers](https://huggingface.co/docs/transformers/en/index) 🤖 essentially. 

```bash
pip install --upgrade diffusers accelerate transformers
```

A typical example of command to run the Dreambooth : 
```bash
CUDA_VISIBLE_DEVICES=3 python ./diffusers/examples/dreambooth/train_dreambooth.py   --instance_data_dir=./positive   --output_dir=./out  --instance_prompt="An infrared thermal frame from the laser scan recording of a metallic part, with a surface crack."   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --max_train_steps=10000   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --train_text_encoder
```

__Example of sentences for semantic guidance__ 📘 : 
```python
# negative sentence : "An infrared thermal frame from the laser scan recording of a metallic part, without surface crack."
# positive sentence : "An infrared thermal frame from the laser scan recording of a metallic part, presenting a surface crack."
```
You can also play with the guidance scale to test various syntheses. This setting selects how the denoising process is influenced by the text information. A guidance between 7 and 10 gives generally accurate synthesis quality for the infrared images. 

⚠️ If __a large majority of the GPUs should run on inference__ for each diffusion models easily (i.e. generating new images from the weights provided). __The training procedure as described in the paper can be high-memory intensive__ (consider using more than 12-24 Go of VRAM if available).

# Links to the FLYD-Frames dataset 🎥
Thermal frames produced for the mono-spectrum experiments described in the paper are subsampled from the thermal recordings measured for the legacy FLYD dataset. A strategy of using various crop sizes centered on the maximum of observed temperature is applied in order to tend to maximize the generalization and the detection robustness for the models using these images on learning step. This strategy may also influence the thermal image synthesis by the diffusion models (more diversity in crack synthesis ?). 
We provide links to download the corresponding thermal images [[Classes]()] [[YOLO]()] [[COCO]()] 

- Classes : crack and sane images separated. For diffusion models training.
- YOLO : labels formated for the crack detection using [YOLO family models](https://github.com/ultralytics/ultralytics).
- COCO : labels formated for the crack detection using [DEtection TRansformers](https://github.com/facebookresearch/dino).
  
# Cite 🔖
If this work is used for academic purpose, please consider citing our paper: 

```bibtex
@inproceedings{helvig2024synthetic,
  title={Synthetic visible-IR images pairs generation for multi-spectral NDT using flying spot thermography and deep learning},
  author={Helvig, Kevin and Trouv{\'e}-Peloux, Pauline and Gav{\'e}rina, Ludovic and Roche, Jean-Michel and Abeloos, Baptiste},
  booktitle={Thermosense: Thermal Infrared Applications XLVI},
  volume={13047},
  pages={14--26},
  year={2024},
  organization={SPIE},
  doi = {10.1117/12.3013537}
}
```
