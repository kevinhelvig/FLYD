# Thermosense conference experiments (2024)
This folder gives minimal codes and data for the work presented at the __SPIE Defense+Commercial Sensing, Thermosense: Thermal Infrared Applications XLVI__ (April 2024) . We provide weights from Stable Diffusion fine-tuning, data amounts used for training and crack detection using Flying-Spot thermography ("frame-per-frame" detection). 

The paper : [Synthetic visible-IR images pairs generation for multi-spectral NDT using flying spot thermography and deep learning [Helvig et al]](https://doi.org/10.1117/12.3013537) 

⚠️ This folder releases __only mono-spectrum (IR) experimental data and diffusion weights for image synthesis__.


# Roadmap :building_construction:
- __[10/10/2024]__ : Folder created. Adding the different links and explanations is on the way ! :biking_man:

# Stable Diffusion: Weights and minimal script usage 
-- TO DO 
A minimal script for sampling is added: the script can generate both cracked or sane thermal frame, depending on the model chosen, and the sentence used for the semantic guidance.
The model used for image synthesis is Stable Diffusion v1.5. The canonical Dreambooth script for fine-tuning has been used for the training (available [here]()). <br>
Link to the negative generator : [[Zenodo]()] <br>
Link to the positive generator : [[Zenodo]()] <br> 

Dependencies: __Hugging Face 🤗__ [Diffusers](https://huggingface.co/docs/diffusers/index) and [Transformers](https://huggingface.co/docs/transformers/en/index) 🤖 essentially. 

```bash
pip install --upgrade diffusers accelerate transformers
```

Sentences for semantic guidance : 
```python
# positive sentence : " "
# negative sentence : " "
```
You can also play with the guidance scale to test various syntheses. This setting selects how the denoising process is influenced by the text information. A guidance between 7 and 10 gives generally accurate synthesis quality for the infrared images.

# Links to the FLYD-Frames dataset 
-- TO DO 

