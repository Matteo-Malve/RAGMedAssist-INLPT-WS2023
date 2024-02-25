"""
This script performs TSDAE (Transformer -based Sequential Denoising Auto-Encode for Unsupervised Sentence Embedding Learning). 
For more information see the paper (2021) by Kexin Wang, Nils Reimers & Iryna Gurevych: https://arxiv.org/abs/2104.06979

The key idea is to reconstruct the original sentence from a corrupted sentence.
"""

from sentence_transformers import SentenceTransformer, models
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers import evaluation
from transformers import AutoModel
import torch
from torch.utils.data import DataLoader
import re
import unicodedata
import pandas as pd
import random
import time
import nltk
nltk.download('punkt')


# Specify paths
data_1 = "pubmed_data_part1.csv"
data_2 = "pubmed_data_part2.csv"
model_name = "thenlper/gte-base"
saving_path = "gte-base-fine-tune"


# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Load data, read in the two CSV files
df_part1 = pd.read_csv(data_1, usecols=['Abstract'])
df_part2 = pd.read_csv(data_2, usecols=['Abstract'])

# Concatenate the two DataFrames
df = pd.concat([df_part1, df_part2], ignore_index=True)

# Clean the data
df = df.dropna(subset=['Abstract'])                                                 #remove results with missing abstracts
df = df[df['Abstract'].apply(lambda x: len(x) >= 100)]                              #remove abstracts shorter than 100 characters
df['Abstract'] = df['Abstract'].str.replace('\xa0', ' ')                            #remove non-breaking spaces
df['Abstract'] = df['Abstract'].str.strip().str.replace('\s+', ' ', regex=True)     #remove trailing whitespaces

# Normalize diacritics and accents
def normalize_characters(text):
    without_diacritics = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return without_diacritics

df['Abstract'] = df['Abstract'].apply(normalize_characters)
#print(df.head(2))

# Extract abstracts
data = df['Abstract'].to_list()

# Select 250 docs randomly for evaluation
eval_docs = random.sample(data, 250)

# Extract sentences
sentences = []
def generate_sentence(sentence):
    spliter = re.compile(r'(?<!\d)\.\s?(?!\d)')                            #splits at "." fowllowed by whitespace or "\n" (keeps periods for decimal numbers like 0.5)
    list_of_sentences = spliter.split(sentence)
    for sent in list_of_sentences:
        if len(sent) > 30:                                                 #and len(sentences) < 100_000:
            sent = re.sub(r'[^\w\s]', " ", sent)
            sentences.append(sent)

for sent in eval_docs:
    generate_sentence(sent)

print(f"Number of sentences: {len(sentences)}")


# Add noise to training data via DenoisingAutoEncoderDataset 
train_data = DenoisingAutoEncoderDataset(sentences)            #returns InputExamples in the format: texts=[noise_fn(sentence), sentence] 
loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Define embedding model
embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(embedding_model.get_word_embedding_dimension(), 'cls')     #aggregate embeddings into single vector using 'CLS' token
model = SentenceTransformer(modules = [embedding_model, pooling_model])
model = model.to(device)

# Define loss function
loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

start_time = time.time()  # Start timing

model.fit(
    train_objectives=[(loader, loss)],
    epochs=1,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True
)
model.save(saving_path)

execution_time = time.time() - start_time  # Calculate execution time

print(f"Execution took {round(execution_time, 2)} seconds on {device}")
 