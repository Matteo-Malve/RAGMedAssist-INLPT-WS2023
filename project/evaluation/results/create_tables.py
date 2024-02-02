import pandas as pd

# Example file paths - replace these with your actual file paths
accuracy_paths = [
    './text_files/retriever_accuracy_dmis-lab_biobert-base-cased-v1.1.txt',
    './text_files/retriever_accuracy_all-MiniLM-L6-v2.txt',
    './text_files/retriever_accuracy_BAAI_bge-base-en-v1.5.txt', 
    './text_files/retriever_accuracy_llmrails_ember-v1.txt', 
    './text_files/retriever_accuracy_jamesgpt1_sf_model_e5.txt', 
    './text_files/retriever_accuracy_thenlper_gte-base.txt', 
    './text_files/retriever_accuracy_intfloat_e5-base-v2.txt', 
]

model_names = [
    'dmis-lab_biobert-base-cased-v1.1',
    'all-MiniLM-L6-v2',
    'BAAI_bge-base-en-v1.5',
    'llmrails_ember-v1',
    'jamesgpt1_sf_model_e5',
    'thenlper_gte-base',
    'intfloat_e5-base-v2'
]

# Initialize an empty DataFrame
df_accuracies = pd.DataFrame()

# Read each file and add the results to the DataFrame
for file_path, model_name in zip(accuracy_paths, model_names):

    with open(file_path, 'r') as file:
        accuracies = [round(float(line.strip()), 3) for line in file]
    
    # Add to DataFrame
    df_accuracies[model_name] = accuracies

# Transpose DataFrame so each row represents a model
df_accuracies = df_accuracies.T
df_accuracies.columns = [f'k={i+1}' for i in range(20)]  # Rename columns to reflect k values


# Function to convert DataFrame to markdown table
def df_to_markdown(df):
    markdown = df.to_markdown()
    return markdown

# Convert to markdown
markdown_table = df_to_markdown(df_accuracies)

# Print or save the markdown table
print(markdown_table)
