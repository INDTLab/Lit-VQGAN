This is the official PyTorch implementation of the paper 

#### [Terrain Scene Generation Using A Lightweight Vector Quantized Generative Adversarial Network](https://ieeexplore.ieee.org/abstract/document/10858453)

### Preparation

- Install required packages: `pip install -r requirements.txt`
- Install [Taming Transformers](https://github.com/CompVis/taming-transformers):
  - Download the [repo](https://github.com/CompVis/taming-transformers)
  - Run `pip install -e .`

### Natural Terrain Scene Data Set (NTSD)

Download [NTSD](https://drive.google.com/drive/folders/1fEGIvaNngXxSVOGn6fWOLiAo1FC9ApVR?usp=drive_link)

### Training Lit-VQGAN

```
python main.py --base logs/c2fmn13_400/configs/project.yaml
```

### Training Transformer

```
python main.py --base logs/c400_sam200/configs/project.yaml
```

Note: Training with multiple GPUs is supported. Simply specify the GPU ids with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,...`



### Paper

[Terrain Scene Generation Using A Lightweight Vector Quantized Generative Adversarial Network](https://ieeexplore.ieee.org/abstract/document/10858453)

Please cite the following paper if you find it useful for your research:

```
@ARTICLE{10858453,
  author={Wang, Yan and Zhou, Huiyu and Dong, Xinghui},
  journal={IEEE Transactions on Big Data}, 
  title={Terrain Scene Generation Using A Lightweight Vector Quantized Generative Adversarial Network}, 
  year={2025},
  volume={},
  number={},
  pages={1-13},
  keywords={Vectors;Image synthesis;Transformers;Decoding;Superresolution;Generative adversarial networks;Feature extraction;Computational modeling;Training;Diffusion models;Terrain scenes;natural terrains;image generation;super-resolution;lightweight networks},
  doi={10.1109/TBDATA.2025.3536926}}
```

