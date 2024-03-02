"""
This script creates data to use for fine-tuning an embedding model on data where ground truth labels are unavailable,
making use of contrastive learning:

    - 
"""


from sentence_transformers import SentenceTransformer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import re
import unicodedata
import pandas as pd
import random
import time
from tqdm import tqdm 



# Specify paths
data_1 = "../data/pubmed_data_part1.csv"
data_2 = "../data/pubmed_data_part2.csv"
embedding_model = "thenlper/gte-base"
paraphrase_model = "tuner007/pegasus_paraphrase"
pos_saving_path = "../data/finetuning_positive_pairs.txt"
neg_saving_path = "../data/finetuning_negative_pairs.txt"


# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Load data, read in the two CSV files
df_part1 = pd.read_csv(data_1, usecols=['Abstract'])
df_part2 = pd.read_csv(data_2, usecols=['Abstract'])

# Concatenate the two DataFrames
df = pd.concat([df_part1, df_part2], ignore_index=True)
#print(df.head())

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


# ------------------------------------------------------------------------------------------------------------------
# Positive Pairs
# ------------------------------------------------------------------------------------------------------------------

# Select 250 docs randomly for positive pairs
selected_documents = random.sample(data, 250)


# Specify model parameters
tokenizer = PegasusTokenizer.from_pretrained(paraphrase_model)
model = PegasusForConditionalGeneration.from_pretrained(paraphrase_model).to(device)

# Create paraphrases
def generate_paraphrase(input_text, num_return_sequences=1, num_beams=10):
    batch = tokenizer(
        [input_text],
        truncation=True,
        padding='longest',
        max_length=model.config.max_position_embeddings, 
        return_tensors="pt").to(device)
    paraphrased = model.generate(
        **batch,
        max_length=model.config.max_position_embeddings,
        num_beams=num_beams, 
        num_return_sequences=num_return_sequences,
        do_sample=True, 
        temperature=2.5)
    tgt_text = tokenizer.batch_decode(paraphrased, skip_special_tokens=True)
    return tgt_text


paraphrases = []

for doc in selected_documents:
    splitter = re.compile(r'(?<!\d)\.\s?(?!\d)')                            #splits at "." fowllowed by whitespace or "\n" (keeps periods for decimal numbers like 0.5)
    sentences = splitter.split(doc)
    paraphrased_sents = []
    for sent in sentences[:len(sentences)-1]:                               #discard last empty string (due to splitting at period+whitespace/newline)
        paraphrased_sents.append(generate_paraphrase(sent))
    flat_paraphrases = ' '.join([item[0] for item in paraphrased_sents if item])
    paraphrases.append(flat_paraphrases)


# Save pos pairs
with open(pos_saving_path, "w") as file:    
    for original, paraphrase in zip(selected_documents, paraphrases):
        file.write(f"{[original, paraphrase]}\n\n")



# ------------------------------------------------------------------------------------------------------------------
# Negative Pairs
# ------------------------------------------------------------------------------------------------------------------

model_kwargs = {'device': device,}
embed_model = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs=model_kwargs)

# # Generate embeddings for similarity calculation
# all_embeddings = np.array([embed_model.embed_documents(doc) for doc in data])
# selected_embeddings = np.array([embed_model.embed_documents(doc) for doc in selected_documents])
def batch_generator(data, batch_size=32):
    """Yield consecutive batches of the specified size from the input list."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def generate_embeddings_in_batches(embed_model, data, batch_size=32):
    """Generate embeddings for a list of documents in batches."""
    batched_embeddings = []
    for batch in tqdm(batch_generator(data, batch_size), desc="Generating Embeddings"):
        embeddings = [embed_model.embed_documents(doc) for doc in batch]
        batched_embeddings.extend(embeddings)
    return np.array(batched_embeddings)

# Adjust the batch_size based on your GPU's memory capacity
batch_size = 32 
all_embeddings = generate_embeddings_in_batches(embed_model, data, batch_size)
selected_embeddings = generate_embeddings_in_batches(embed_model, selected_documents, batch_size)


# Calculate cosine similarity between selected docs and all docs
similarity_matrix = cosine_similarity(selected_embeddings, all_embeddings)

# Since a document is most similar to itself, we set these values to 1 (as we're looking for min values later)
np.fill_diagonal(similarity_matrix, 1)

negative_indices = np.argmin(similarity_matrix, axis=1)
negative_pairs = [(selected_documents[i], data[negative_indices[i]]) for i in range(len(selected_documents))]


# Save neg pairs
with open(neg_saving_path, "w") as file:    
    for pos_sample, neg_sample in negative_pairs:
        file.write(f"{[pos_sample, negative_pairs]}\n\n")