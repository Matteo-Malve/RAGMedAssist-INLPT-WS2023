"""
This script is used to evaluate our chatbot.
"""
import sys
# Add parent directory of 'app' to Python path
sys.path.append('../../..')
from chatbot.app.custom_chatbot import MedicalChatbot
import yaml
import time



with open("../../../chatbot/app/cfg.yaml", "r") as file:
    cfg = yaml.safe_load(file)

# Define different testing queries
queries = [
    "Is the association between intelligence and lifespan mostly genetic?",
    "Is regular breakfast consumption associated with increased IQ in kindergarten children?",
    "Explore the relationship between genetic factors and intelligence variations in monozygotic twins.",
    "Does tick-borne encephalitis carry a high risk of incomplete recovery in children?",
    "Is there a correlation between working memory and fluid intelligence across children with developmental disorders?",
    "How does the Theory of Multiple Intelligences differentiate cognitive abilities?",
    "Based on the discovery that mitochondrial dysfunction plays an important role in Alzheimer's disease, how could interventions aimed at improving mitochondrial function affect the treatment or management of Alzheimer's disease?",
    "Should artificial intelligence surpass human intelligence in diagnostic accuracy, what ethical considerations would arise in relying on AI for making critical healthcare decisions?",
    "List the cognitive domains affected by intelligence quotient (IQ) variations in children with specific learning disabilities.",
    "What are the key components of emotional intelligence as it relates to patient care in nursing?"
]

name = "Different Question Types"

# Call chatbot
chatbot = MedicalChatbot(cfg)

# Initialize list to store query-response pairs
query_response_pairs = []

#Open a Markdown file for writing the results
with open("question_types_1_77.md", "w") as md_file:
    md_file.write(f"# Testing of Differnt Question Types\n")
    md_file.write(f"LLM parameters: temp={cfg['llm_model']['temperature']}, topp={cfg['llm_model']['top_p']}, score_threshold={cfg['retrievers']['faiss']['score_threshold']}\n\n")
    prompt = chatbot.get_prompt()
    md_file.write(f"## Custom Prompt Template:\n```python\n{prompt}\n```\n\n")
    md_file.write(f"### {name}\n")
    for query in queries:
        start_time = time.time()  # Start timing
        result = chatbot.generate_response(query)
        execution_time = time.time() - start_time  # Calculate execution time
        # Store query-response pair in list
        query_response_pairs.append([query, result])
        # Write the query, execution time, and result to the Markdown file
        md_file.write(f"## Query:\n*{query}*\n\n")
        md_file.write(f"**Execution Time:**\n{round(execution_time, 2)} seconds on {chatbot.device} using {cfg['llm_model']['name']}.\n\n")
        md_file.write(f"### Response:\n{result}\n\n")
        # Add a horizontal rule for separation between entries
        md_file.write("---\n\n")

# Save query-response pairs in .txt file for easier extraction
with open("question_types_1_77.txt", "w") as txt_file:
    # Convert list of pairs to a string representation
    txt_content = str(query_response_pairs)
    txt_file.write(txt_content)

print("Results have been saved.")
