# -*- coding: utf-8 -*-
"""
Improved version of untitled25.py that runs with real models (not mock)
and ensures visualizations are generated correctly
"""

from google.colab import drive
drive.mount('/content/drive')

# Install required packages - make sure to include all dependencies
!pip install transformers nltk sentence-transformers tqdm rouge-score matplotlib pandas torch

# Download nltk data
!python -m nltk.downloader punkt

# Copy the hybrid model script and create necessary directories
!cp "/content/drive/MyDrive/HybridModels/multi_model_hybrid.py" /content/
!mkdir -p data model_evaluations/multi_model_hybrid visualizations

import re
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Read the original file
with open("/content/drive/MyDrive/HybridModels/Pride_and_prejudice.txt", 'r', encoding='utf-8', errors='ignore') as file:
    content = file.read()

# Split by paragraphs
paragraphs = re.split(r'\n\s*\n', content)
paragraphs = [p.strip() for p in paragraphs if p.strip()]
print(f"Found {len(paragraphs)} paragraphs")

# Create a modified version with clear chapter markers every 20 paragraphs
modified_content = ""
for i in range(0, len(paragraphs), 20):
    chapter_num = i // 20 + 1
    modified_content += f"Chapter {chapter_num}\n\n"

    # Add paragraphs for this chapter
    chapter_paragraphs = paragraphs[i:i+20]
    modified_content += "\n\n".join(chapter_paragraphs)
    modified_content += "\n\n"

# Save the preprocessed content for the model to use
with open('/content/pride_and_prejudice.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(modified_content)

print(f"Created preprocessed version with chapter markers")

# Define functions for knowledge base creation and context pairs
def create_knowledge_base(text_file, chunk_size=500, overlap=100):
    """Create knowledge base chunks from a text file"""
    with open(text_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Split into paragraphs and then into chunks
    paragraphs = [p for p in full_text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = ""
    chunk_id = 0

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'text': current_chunk.strip()
            })
            # Include some overlap from the previous chunk
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            chunk_id += 1

        current_chunk += paragraph + "\n\n"

    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append({
            'id': f'chunk_{chunk_id}',
            'text': current_chunk.strip()
        })

    return chunks

def create_context_pairs(kb_chunks, num_pairs=30):
    """Create context-query-answer triples for evaluation"""
    import random

    pairs = []

    for _ in range(num_pairs):
        # Select a random chunk as context
        context_chunk = random.choice(kb_chunks)
        context = context_chunk['text']

        # Split the context into sentences
        sentences = nltk.sent_tokenize(context)

        if len(sentences) < 3:
            continue

        # Select a random sentence to use as the expected answer
        answer_idx = random.randint(1, len(sentences) - 1)
        expected_answer = sentences[answer_idx]

        # Create a question about the selected sentence
        prev_sentence = sentences[answer_idx - 1]
        query = f"What happens after: '{prev_sentence}'?"

        pairs.append({
            'context': context,
            'query': query,
            'expected_answer': expected_answer
        })

    return pairs[:num_pairs]  # Ensure we have exactly num_pairs

# Import the functions from multi_model_hybrid.py
from multi_model_hybrid import run_multi_model_evaluation, evaluate_multi_model_results, visualize_multimodel_results

# Create knowledge base from the preprocessed text
print("Creating knowledge base...")
kb_chunks = create_knowledge_base('/content/pride_and_prejudice.txt')
with open('data/knowledge_base.json', 'w') as f:
    json.dump(kb_chunks, f, indent=2)
print(f"Created knowledge base with {len(kb_chunks)} chunks")

# Create context pairs for evaluation (using fewer pairs for real model testing)
print("Creating context pairs...")
context_pairs = create_context_pairs(kb_chunks, num_pairs=10)  # Reduced number for real models
with open('data/context_pairs.json', 'w') as f:
    json.dump(context_pairs, f, indent=2)
print(f"Created {len(context_pairs)} context pairs")

# Generate embeddings
print("Generating embeddings for knowledge base...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
kb_texts = [chunk['text'] for chunk in kb_chunks]
kb_embeddings = encoder.encode(kb_texts, show_progress_bar=True)

# Run the evaluation with REAL models (use_mock=False)
print("Running evaluation with REAL models...")
results = run_multi_model_evaluation(context_pairs, kb_chunks, kb_embeddings, use_mock=False)

# Evaluate results
print("Evaluating results...")
evaluation = evaluate_multi_model_results(results)

# Create visualizations
print("Creating visualizations...")
visualize_multimodel_results(evaluation)

# Display evaluation results
print("\nEvaluation Results Summary:")
for model_name, metrics in evaluation.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# Check that visualizations were created
visualization_files = os.listdir("visualizations")
print("\nVisualization files created:")
for file in visualization_files:
    print(f"- {file}")

# Save the evaluation results
with open('model_evaluations/evaluation_summary.json', 'w') as f:
    # Convert numpy values to Python native types for JSON serialization
    serializable_evaluation = {}
    for model, metrics in evaluation.items():
        serializable_evaluation[model] = {
            k: float(v) for k, v in metrics.items()
        }
    json.dump(serializable_evaluation, f, indent=2)

print("\nEvaluation complete! Check the visualizations directory for outputs.")
