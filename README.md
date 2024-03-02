# 👾 Group Project

| Name and surname    |  Matric. Nr. | GitHub username  |   e-mail address   |
|:--------------------|:-------------|:-----------------|:-------------------|
| Matteo Malvestiti | 4731243| Matteo-Malve | matteo.malvestiti@stud.uni-heidelberg.de|
| Sandra Friebolin | 3175035 | friebolin | sandra_friebolin@proton.me |
| Yusuf Berber | 4736316 | yberber | yusuf.berber@stud.uni-heidelberg.de |

### Advisor
Robin Khanna (R.Khanna@stud.uni-heidelberg.de)


# 🛠️ Installation

### 1) Missing Modules

In order to let the user run the code, we chose a very simple approach: all required packages are inside requirements.txt \
Hence, the first thing to do is to install the missing packages in your python environment with:

First of all, move inside the requirements folder: `cd requirements`. Then:

➡️ If you run on M1/M2 Macs: `pip install -r requirements-mps.txt` (see [`requirements-mps.txt`](project/requirements/requirements-mps.txt))

➡️ For machines with support for Cuda: `pip install -r requirements-cuda.txt` (see [`requirements-cuda.txt`](project/requirements/requirements-cuda.txt))

⚠️ We all ran and tested on a Miniconda environment for ARM machines, since we all had M1/M2 MacBooks. To reproduce the same exact environment you can build on top of this [configuration file](https://github.com/jeffheaton/app_deep_learning/blob/main/install/torch.yml). It should be however sufficient to install all the requirement modules in your own environment.

### 2) Embedding and LLM Models

The embedding model will be automatically downloaded from [Huggingface's Hub](https://huggingface.co/thenlper/gte-base). It is public and thus should not require any authentification token.

Concernig the LLM _Mistral Instruct_, our code is flexible. You can either:
- Do nothing and it will be automatically downloaded from [Huggingface's Hub](https://huggingface.co/mistralai/Mistral-7B-v0.1/discussions/104)
- 🍎 Suggested for Mac users: download it manually with GPT4ALL and specify your local path in the [`cfg.yaml`](project/chatbot/app/cfg.yaml) configuration file.

# 🚀 Launch the chatbot app

Open the jupyter notebook [app_launcher.ipynb](./chatbot/frontend/app_launcher.ipynb) inside the folder `./chatbot/frontend/`

➡️ If you run on a Mac it's easier then ever! \
Just execute the first and only cell inside the relative section!

    !streamlit run app.py
A new tab in your default browser should authomatically open on the local server and you will be ready to go.
If this doesn't happen, copy and paste the outputed url in your browser.

➡️ If you run on Colab: a little less friendly option. \
Start by running the first cell and installing all the required modules. Secondly, mount your google drive and move to the correct folder.
Then run the launch command provided in the last cell:

    !wget -q -O - ipv4.icanhazip.com
    !streamlit run app.py & npx localtunnel --port 8501
Being Colab a closed environment, a localtunnel is required.
You will have to click on the provided url.
A gate webpage will block you before actually getting to the app. Copy and paste the IP address printed by the cell, paste it in the box and cofirm.


