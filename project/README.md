# 👾 Group Project

| Name and surname    |  Matric. Nr. | GitHub username  |   e-mail address   |
|:--------------------|:-------------|:-----------------|:-------------------|
| Matteo Malvestiti | 4731243| Matteo-Malve | matteo.malvestiti@stud.uni-heidelberg.de|
| Sandra Friebolin | 3175035 | friebolin | sandra_friebolin@proton.me |
| Yusuf Berber | 4736316 | yberber | yusuf.berber@stud.uni-heidelberg.de |

### Advisor
Robin Khanna (R.Khanna@stud.uni-heidelberg.de)


# Installation
### 1) Missing modules

In order to let the user run the code, we chose a very simple approach: all required packages are inside requirements.txt \
Hence, the first thing to do is to install the missing packages in your python environment with:

    # If you run on M1/M2 Macs
    pip install -r requirements-mps.txt

    # For machines with support for Cuda
    pip install -r requirements-cuda.txt


NOTE: We all ran and tested on a Miniconda environment for ARM machines, since we all had M1/M2 MacBooks. To reproduce the same exact environment you can build on top of this [configuration file](https://github.com/jeffheaton/app_deep_learning/blob/main/install/torch.yml). It should be however sufficient to install all the requirement modules in your own environment.

### 2) Embedding and LLM
The Embedding model will be automatically downloaded from Huggingface's hub. It's public, thus it should not require any authentification token.

Concernig the LLM (_Mistral instruct_), our code is flexible. You can either:
- do nothing and it will be automatically downloaded from Huggingface's hub
- [Suggested for Mac users] download it manually with GPT4ALL and specify your local path in 
    >chatbot/cfg.yaml
