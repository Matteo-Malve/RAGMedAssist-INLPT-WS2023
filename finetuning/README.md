# finetuning

- 🗂️ [`contrastive_learning`](finetuning/contrastive_learning): This folder contains files for contrastive learning in order to fine tune the embedding model.

    - 💻 [`create_contrastive_learning_data.py`](finetuning/contrastive_learning/create_contrastive_learning_data.py): This code generates fine tuning data. Currently, only the positive pairs for fine tuning are generated (via paraphrasing). Generating negative pairs will be future work.
    - 💽 [`finetuning_positive_pairs.txt`](finetuning/contrastive_learning/finetuning_positive_pairs.txt): Contains positive pairs: an original document and its paraphrased version.

- 🗂️ [`TSDAE`](finetuning/TSDAE): This folder contains files for the "Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning".

    - 👾 [`gte-base-fine-tune`](finetuning/TSDAE/gte-base-fine-tune): This folder contains the finetuned embedding model.
    - 💻 [`TSDAE.py`](finetuning/TSDAE/TSDAE.py): This code fine tunes the model via TSADE.
    