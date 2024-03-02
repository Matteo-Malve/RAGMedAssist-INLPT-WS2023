# 👾 chatbot

- 🗂️ [`app`](app): This folder contains the main code for our chatbot application.

    - ⚙️ [`cfg.yaml`](app/cfg.yaml): Configuration file for the chatbot.

    - 💻 [`custom_chatbot.py`](app/custom_chatbot.py): Main code for chatbot.

    - 💻 [`custom_retriever.py`](app/custom_chatbot.py): Overwrites LangChain's `EnsembleRetriever` to perform rank fusion of documents retrieved by multiple retrievers.

- 🗂️ [`frontend`](frontend): This folder contains the code for our frontend web application.

    - 💻 [`app_launcher.ipynb`](frontend/app_launcher.ipynb): Launches the web application.

    - 💻 [`app.py`](frontend/app.py): Defines web application layout.