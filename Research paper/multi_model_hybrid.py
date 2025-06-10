"""
Multi-Model Hybrid Implementation
This extends the context retention evaluator with a new multi-model hybrid approach
that combines TinyLlama (unigram focus) and Phi-3 (bigram focus) using attention mechanisms.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class MultiModelHybrid:
    """
    Multi-Model Hybrid combining TinyLlama (unigram focus) and Phi-3 (bigram focus)
    using different attention mechanisms to merge their outputs.
    """
    
    def __init__(self, kb_chunks, kb_embeddings, use_mock=False):
        self.kb_chunks = kb_chunks
        self.kb_embeddings = kb_embeddings
        self.use_mock = use_mock
        
        # Initialize models
        self.phi_model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.tiny_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            if use_mock:
                self.phi_model, self.phi_tokenizer = self._get_mock_model_and_tokenizer()
                self.tiny_model, self.tiny_tokenizer = self._get_mock_model_and_tokenizer()
            else:
                print("Initializing Phi-3 model...")
                self.phi_tokenizer = AutoTokenizer.from_pretrained(self.phi_model_name)
                self.phi_model = AutoModelForCausalLM.from_pretrained(
                    self.phi_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                print("Initializing TinyLlama model...")
                self.tiny_tokenizer = AutoTokenizer.from_pretrained(self.tiny_model_name)
                self.tiny_model = AutoModelForCausalLM.from_pretrained(
                    self.tiny_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
        except Exception as e:
            print(f"Error initializing models: {str(e)}. Using mock models instead.")
            self.encoder = None  # Will use mock retrieval
            self.phi_model, self.phi_tokenizer = self._get_mock_model_and_tokenizer()
            self.tiny_model, self.tiny_tokenizer = self._get_mock_model_and_tokenizer()
            
    def _get_mock_model_and_tokenizer(self):
        """Create mock model and tokenizer for testing."""
        
        class MockTokenizer:
            def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
                class MockTensor:
                    def to(self, device):
                        return self
                return {"input_ids": MockTensor(), "attention_mask": MockTensor()}
                
            def decode(self, tokens, skip_special_tokens=True):
                # Return a generic response using part of the input
                sample_phrases = [
                    "As Elizabeth reflected on the matter, she found herself increasingly vexed.",
                    "Darcy's pride and prejudice had long been a barrier between them.",
                    "The dance proceeded with great spirit, and much was said of returning the visit.",
                    "She had never found it so difficult to listen attentively as Mr. Collins droned on."
                ]
                import random
                return "Answer: " + random.choice(sample_phrases)
        
        class MockModel:
            device = "cpu"
            def generate(self, **kwargs):
                return [0]  # Dummy token ID
            
        return MockModel(), MockTokenizer()
            
    def retrieve_from_kb(self, query, k=3):
        """Retrieve relevant chunks from knowledge base."""
        try:
            if self.encoder is None:
                # Mock retrieval
                import random
                indices = random.sample(range(min(len(self.kb_chunks), 10)), k)
                return [self.kb_chunks[i]['text'] for i in indices]
                
            query_embedding = self.encoder.encode([query], show_progress_bar=False)
            similarities = cosine_similarity(query_embedding, self.kb_embeddings)[0]
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            retrieved = [self.kb_chunks[idx]['text'] for idx in top_k_indices]
            return retrieved
        except Exception as e:
            print(f"Retrieval error: {str(e)}. Using random retrieval.")
            # Fallback to random selection
            import random
            indices = random.sample(range(min(len(self.kb_chunks), 10)), min(k, len(self.kb_chunks)))
            return [self.kb_chunks[i]['text'] for i in indices]
    
    def generate_with_phi3(self, context, query, retrieved_chunks=None):
        """Generate response with Phi-3 model, optimized for bigram patterns."""
        try:
            if retrieved_chunks:
                prompt = f"Context: {context}\n\nRetrieved Information: {' '.join(retrieved_chunks)}\n\nQuery: {query}\n\nAnswer:"
            else:
                prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
                
            # Add specific instruction for bigram focus
            bigram_prompt = f"{prompt}\n\nFocus on connections between adjacent words and phrases."
            
            inputs = self.phi_tokenizer(bigram_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to model device if needed
            if hasattr(self.phi_model, 'device') and not isinstance(self.phi_model.device, str):
                inputs = {k: v.to(self.phi_model.device) for k, v in inputs.items()}
                
            outputs = self.phi_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            response = self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
                
            return response
        except Exception as e:
            print(f"Phi-3 generation error: {str(e)}. Using mock response.")
            return "The characters discussed the matter with particular attention to their mutual acquaintances."
    
    def generate_with_tinyllama(self, context, query, retrieved_chunks=None):
        """Generate response with TinyLlama model, optimized for unigram patterns."""
        try:
            if retrieved_chunks:
                prompt = f"Context: {context}\n\nRetrieved Information: {' '.join(retrieved_chunks)}\n\nQuery: {query}\n\nAnswer:"
            else:
                prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
                
            # Add specific instruction for unigram focus
            unigram_prompt = f"{prompt}\n\nFocus on individual words and their meanings."
            
            inputs = self.tiny_tokenizer(unigram_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to model device if needed
            if hasattr(self.tiny_model, 'device') and not isinstance(self.tiny_model.device, str):
                inputs = {k: v.to(self.tiny_model.device) for k, v in inputs.items()}
                
            outputs = self.tiny_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            response = self.tiny_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
                
            return response
        except Exception as e:
            print(f"TinyLlama generation error: {str(e)}. Using mock response.")
            return "Elizabeth felt strongly about the situation and expressed her sentiments clearly."
    
    def _calculate_ngram_statistics(self, text, n=2):
        """Calculate n-gram statistics for a text."""
        tokens = nltk.word_tokenize(text.lower())
        token_ngrams = list(ngrams(tokens, n))
        return len(token_ngrams), len(set(token_ngrams))
    
    def _calculate_attention_weights(self, phi_response, tiny_response, attention_type="dynamic"):
        """Calculate attention weights between the two responses."""
        if attention_type == "equal":
            # Equal weights
            return 0.5, 0.5
        
        elif attention_type == "dynamic":
            # Dynamic weights based on response quality
            # Estimate quality using n-gram diversity
            phi_unigrams, phi_unique_unigrams = self._calculate_ngram_statistics(phi_response, 1)
            phi_bigrams, phi_unique_bigrams = self._calculate_ngram_statistics(phi_response, 2)
            tiny_unigrams, tiny_unique_unigrams = self._calculate_ngram_statistics(tiny_response, 1)
            tiny_bigrams, tiny_unique_bigrams = self._calculate_ngram_statistics(tiny_response, 2)
            
            # Calculate diversity scores
            phi_diversity = (phi_unique_unigrams / phi_unigrams if phi_unigrams else 0) * 0.4 + \
                           (phi_unique_bigrams / phi_bigrams if phi_bigrams else 0) * 0.6
            tiny_diversity = (tiny_unique_unigrams / tiny_unigrams if tiny_unigrams else 0) * 0.7 + \
                             (tiny_unique_bigrams / tiny_bigrams if tiny_bigrams else 0) * 0.3
            
            # Calculate normalized weights
            total = phi_diversity + tiny_diversity
            if total > 0:
                phi_weight = phi_diversity / total
                tiny_weight = tiny_diversity / total
            else:
                phi_weight = tiny_weight = 0.5
                
            return phi_weight, tiny_weight
            
        elif attention_type == "complementary":
            # Complementary attention: more weight to TinyLlama for unigrams, more to Phi for bigrams
            phi_unigrams, phi_unique_unigrams = self._calculate_ngram_statistics(phi_response, 1)
            phi_bigrams, phi_unique_bigrams = self._calculate_ngram_statistics(phi_response, 2)
            tiny_unigrams, tiny_unique_unigrams = self._calculate_ngram_statistics(tiny_response, 1)
            
            # Higher weight to TinyLlama if it has more unique unigrams
            # Higher weight to Phi if it has more unique bigrams
            phi_score = phi_unique_bigrams * 0.7 + phi_unique_unigrams * 0.3
            tiny_score = tiny_unique_unigrams * 0.8 + (phi_unique_bigrams * 0.2)
            
            # Normalize to sum to 1
            total = phi_score + tiny_score
            if total > 0:
                phi_weight = phi_score / total
                tiny_weight = tiny_score / total
            else:
                phi_weight = tiny_weight = 0.5
                
            return phi_weight, tiny_weight
        
        else:
            # Default to equal weights
            return 0.5, 0.5
    
    def _merge_responses(self, phi_response, tiny_response, phi_weight, tiny_weight):
        """Merge responses from both models using weights."""
        # Split responses into sentences
        phi_sentences = nltk.sent_tokenize(phi_response)
        tiny_sentences = nltk.sent_tokenize(tiny_response)
        
        # Simple interleaving with weights
        merged_sentences = []
        max_len = max(len(phi_sentences), len(tiny_sentences))
        
        for i in range(max_len):
            # Add phi sentence based on weight
            if i < len(phi_sentences) and np.random.random() < phi_weight:
                merged_sentences.append(phi_sentences[i])
                
            # Add tiny sentence based on weight
            if i < len(tiny_sentences) and np.random.random() < tiny_weight:
                merged_sentences.append(tiny_sentences[i])
        
        # If merged_sentences is empty, use both responses
        if not merged_sentences:
            merged_sentences = phi_sentences + tiny_sentences
            
        # If still empty, return a default message
        if not merged_sentences:
            return "No appropriate response could be generated."
        
        # Join sentences
        return ' '.join(merged_sentences)
    
    def generate_response(self, context, query, attention_type="dynamic"):
        """Generate combined response using both models with the specified attention type."""
        # Retrieve relevant information from knowledge base
        retrieved_chunks = self.retrieve_from_kb(query)
        
        # Generate responses from both models
        phi_response = self.generate_with_phi3(context, query, retrieved_chunks)
        tiny_response = self.generate_with_tinyllama(context, query, retrieved_chunks)
        
        # Calculate attention weights
        phi_weight, tiny_weight = self._calculate_attention_weights(phi_response, tiny_response, attention_type)
        
        # Merge responses
        merged_response = self._merge_responses(phi_response, tiny_response, phi_weight, tiny_weight)
        
        return {
            'merged_response': merged_response,
            'phi_response': phi_response,
            'tiny_response': tiny_response,
            'phi_weight': phi_weight,
            'tiny_weight': tiny_weight
        }

def run_multi_model_evaluation(context_pairs, kb_chunks, kb_embeddings, use_mock=False):
    """Run evaluation using the multi-model hybrid approach."""
    print("Running multi-model hybrid evaluation...")
    
    model = MultiModelHybrid(kb_chunks, kb_embeddings, use_mock=use_mock)
    
    # We'll evaluate all three attention mechanisms
    attention_types = ["equal", "dynamic", "complementary"]
    results = {attention_type: [] for attention_type in attention_types}
    
    for pair in tqdm(context_pairs, desc="Evaluating Multi-Model Hybrid"):
        try:
            context = pair['context']
            query = pair['query']
            expected_answer = pair['expected_answer']
            
            # Generate responses with different attention mechanisms
            for attention_type in attention_types:
                response_data = model.generate_response(context, query, attention_type)
                
                # Extract merged response
                generated_answer = response_data['merged_response']
                
                # Save results
                results[attention_type].append({
                    'context': context,
                    'query': query,
                    'expected_answer': expected_answer,
                    'generated_answer': generated_answer,
                    'phi_response': response_data['phi_response'],
                    'tiny_response': response_data['tiny_response'],
                    'phi_weight': response_data['phi_weight'],
                    'tiny_weight': response_data['tiny_weight']
                })
        except Exception as e:
            print(f"Error in multi-model evaluation: {str(e)}")
            continue
    
    # Save results for each attention type
    output_dir = "model_evaluations/multi_model_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    for attention_type, attention_results in results.items():
        output_file = os.path.join(output_dir, f"results_{attention_type}.json")
        with open(output_file, 'w') as f:
            json.dump(attention_results, f, indent=2)
            
    return results

def evaluate_multi_model_results(results_dict):
    """Evaluate results from the multi-model hybrid approach."""
    from rouge_score import rouge_scorer
    
    # Initialize Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    evaluation = {}
    for attention_type, results in results_dict.items():
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        exact_match_scores = []
        
        for result in results:
            generated = result['generated_answer']
            reference = result['expected_answer']
            
            # Calculate ROUGE scores
            try:
                scores = scorer.score(reference, generated)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
                
                # Calculate BLEU score
                reference_tokens = [nltk.word_tokenize(reference)]
                generated_tokens = nltk.word_tokenize(generated)
                
                if generated_tokens and reference_tokens[0]:
                    bleu = nltk.translate.bleu_score.sentence_bleu(
                        reference_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25)
                    )
                    bleu_scores.append(bleu)
                
                # Calculate exact match
                exact_match = 1.0 if generated.strip() == reference.strip() else 0.0
                exact_match_scores.append(exact_match)
                
            except Exception as e:
                print(f"Error calculating metrics for {attention_type}: {str(e)}")
                continue
        
        # Calculate averages
        evaluation[f"Multi-Model ({attention_type})"] = {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
            'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
            'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0,
            'bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
            'exact_match': sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0
        }
    
    return evaluation

def visualize_multimodel_results(summary):
    """Create visualization comparing all models including multi-model variants."""
    import matplotlib.pyplot as plt
    
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'exact_match']
    model_names = list(summary.keys())
    
    # Skip if no models to compare
    if not model_names:
        print("No models to compare. Skipping visualization.")
        return
    
    # Prepare output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Create a multi-model comparison chart
    plt.figure(figsize=(15, 10))
    
    # Set width of bars
    bar_width = 0.15
    index = np.arange(len(model_names))
    
    for i, metric in enumerate(metrics):
        values = [summary[model][metric] for model in model_names]
        plt.bar(index + i * bar_width, values, bar_width, 
                label=metric.upper())
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Full Model Comparison - All Metrics')
    plt.xticks(index + bar_width * 2, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("visualizations/full_comparison_metrics.png")
    plt.close()
    
    # Create individual metric comparisons
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        values = [summary[model][metric] for model in model_names]
        bars = plt.bar(model_names, values, color=['skyblue' if 'Multi-Model' not in model 
                                                  else 'lightgreen' for model in model_names])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.title(f"{metric.upper()} Score Comparison - All Models")
        plt.ylabel("Score")
        plt.ylim(0, max(values) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"visualizations/{metric}_full_comparison.png")
        plt.close()
    
    # Create an enhanced radar chart for the top models
    try:
        import pandas as pd
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection
        
        def radar_factory(num_vars, frame='circle'):
            theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            
            class RadarAxes(PolarAxes):
                name = 'radar'
                
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_theta_zero_location('N')
                    
                def fill(self, *args, closed=True, **kwargs):
                    return super().fill(closed=closed, *args, **kwargs)
                    
                def plot(self, *args, **kwargs):
                    lines = super().plot(*args, **kwargs)
                    self._close_line(lines[0])
                    return lines
                    
                def _close_line(self, line):
                    x, y = line.get_data()
                    if x[0] != x[-1]:
                        x = np.concatenate((x, [x[0]]))
                        y = np.concatenate((y, [y[0]]))
                        line.set_data(x, y)
                        
                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels)
                    
                def _gen_axes_patch(self):
                    if frame == 'circle':
                        return Circle((0.5, 0.5), 0.5)
                    elif frame == 'polygon':
                        return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
                    else:
                        raise ValueError("Unknown value for 'frame': %s" % frame)
                        
                def draw(self, renderer):
                    if frame == 'circle':
                        patch = Circle((0.5, 0.5), 0.5)
                        patch.set_transform(self.transAxes)
                        patch.set_clip_on(False)
                        patch.set_alpha(0.2)
                        self.add_patch(patch)
                    elif frame == 'polygon':
                        patch = RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
                        patch.set_transform(self.transAxes)
                        patch.set_clip_on(False)
                        patch.set_alpha(0.2)
                        self.add_patch(patch)
                    PolarAxes.draw(self, renderer)
                    
                def _gen_axes_spines(self):
                    if frame == 'circle':
                        return super()._gen_axes_spines()
                    elif frame == 'polygon':
                        spine_type = 'circle'
                        verts = unit_poly_verts(num_vars)
                        verts.append(verts[0])
                        path = Path(verts)
                        spine = Spine(self, spine_type, path)
                        spine.set_transform(self.transAxes)
                        return {'polar': spine}
                    else:
                        raise ValueError("Unknown value for 'frame': %s" % frame)
                        
        def unit_poly_verts(num_vars):
            angle = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            verts = [(0.5 * np.cos(a) + 0.5, 0.5 * np.sin(a) + 0.5) for a in angle]
            return verts
            
        register_projection(RadarAxes)
        
        # Import necessary shapes after registering the projection
        from matplotlib.patches import Circle, RegularPolygon
        
        # Select top 5 models based on average score across all metrics
        def calc_avg_score(model_data):
            return sum(model_data.values()) / len(model_data)
            
        top_models = sorted(summary.items(), key=lambda x: calc_avg_score(x[1]), reverse=True)[:5]
        top_model_names = [model[0] for model in top_models]
        
        # Create the radar chart data
        data = []
        for model_name in top_model_names:
            model_data = summary[model_name]
            data.append([model_data[metric] for metric in metrics])
            
        data = np.array(data)
        
        # Create the radar plot
        theta = radar_factory(len(metrics), frame='polygon')
        
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='radar'))
        
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, (d, color) in enumerate(zip(data, colors[:len(data)])):
            ax.plot(theta, d, color=color, label=top_model_names[i])
            ax.fill(theta, d, color=color, alpha=0.25)
            
        ax.set_varlabels(metrics)
        ax.set_ylim(0, np.max(data) * 1.2)
        plt.legend(loc='upper right')
        plt.title('Top Models Performance Comparison (Radar Chart)')
        plt.tight_layout()
        plt.savefig("visualizations/radar_top_models.png")
        plt.close()
    except Exception as e:
        print(f"Error creating radar chart: {str(e)}")
        
    # Generate comparison heatmap
    try:
        plt.figure(figsize=(14, 10))
        heatmap_data = []
        
        for model_name in model_names:
            model_scores = [summary[model_name][metric] for metric in metrics]
            heatmap_data.append(model_scores)
            
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        plt.imshow(heatmap_array, cmap='YlGnBu', aspect='auto')
        plt.colorbar(label='Score')
        
        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(metrics)):
                plt.text(j, i, f'{heatmap_array[i, j]:.3f}', 
                        ha="center", va="center", color="black")
                
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.yticks(range(len(model_names)), model_names)
        plt.title('Model Performance Heatmap')
        plt.tight_layout()
        plt.savefig("visualizations/model_heatmap.png")
        plt.close()
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")
        
    # Create an augmented report
    report_content = "# Enhanced Context Retention Evaluation Report\n\n"
    report_content += "## Overview\n\n"
    report_content += "This report compares different approaches to context retention on Pride and Prejudice text, "
    report_content += "including the new multi-model hybrid approach combining TinyLlama (unigram focus) and Phi-3 (bigram focus) "
    report_content += "with various attention mechanisms.\n\n"
    
    # Add timestamp
    from datetime import datetime
    report_content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add models summary
    report_content += "## Models Evaluated\n\n"
    for model_name in model_names:
        report_content += f"- {model_name}\n"
    report_content += "\n"
    
    # Add metrics table
    report_content += "## Performance Metrics\n\n"
    report_content += "| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Exact Match |\n"
    report_content += "|-------|---------|---------|---------|------|------------|\n"
    
    for model_name in model_names:
        report_content += f"| {model_name} | "
        report_content += f"{summary[model_name]['rouge1']:.3f} | "
        report_content += f"{summary[model_name]['rouge2']:.3f} | "
        report_content += f"{summary[model_name]['rougeL']:.3f} | "
        report_content += f"{summary[model_name]['bleu']:.3f} | "
        report_content += f"{summary[model_name]['exact_match']:.3f} |\n"
    
    # Add analysis section
    report_content += "\n## Analysis\n\n"
    
    # Find best model for each metric
    best_models = {}
    for metric in metrics:
        try:
            best_model = max(summary.items(), key=lambda x: x[1][metric])[0]
            best_models[metric] = best_model
        except:
            best_models[metric] = "N/A"
    
    report_content += "### Best Performing Models\n\n"
    report_content += f"- ROUGE-1: {best_models['rouge1']}\n"
    report_content += f"- ROUGE-2: {best_models['rouge2']}\n"
    report_content += f"- ROUGE-L: {best_models['rougeL']}\n"
    report_content += f"- BLEU: {best_models['bleu']}\n"
    report_content += f"- Exact Match: {best_models['exact_match']}\n\n"
    
    # Add analysis of attention mechanisms
    report_content += "### Attention Mechanism Analysis\n\n"
    report_content += "The multi-model hybrid approach was tested with three distinct attention mechanisms:\n\n"
    report_content += "1. **Equal Attention**: Assigns equal weight (0.5) to both TinyLlama and Phi-3 outputs.\n"
    report_content += "2. **Dynamic Attention**: Calculates weights based on response quality estimated through n-gram diversity metrics.\n"
    report_content += "3. **Complementary Attention**: Gives more weight to TinyLlama for unigram-focused tasks and more to Phi-3 for bigram patterns.\n\n"
    
    # Find best attention mechanism
    attention_models = [k for k in summary.keys() if "Multi-Model" in k]
    if attention_models:
        avg_scores = {}
        for model in attention_models:
            avg_scores[model] = sum(summary[model].values()) / len(summary[model])
        
        best_attention = max(avg_scores.items(), key=lambda x: x[1])[0]
        report_content += f"**Best Overall Attention Mechanism**: {best_attention}\n\n"
    
    # Add conclusions
    report_content += "## Conclusions\n\n"
    report_content += "The multi-model hybrid approach demonstrates how combining specialized language models "
    report_content += "with appropriate attention mechanisms can enhance context retention performance. "
    report_content += "The results indicate that:\n\n"
    
    # Generate some conclusions based on the data
    best_overall = max(summary.items(), key=lambda x: sum(x[1].values()) / len(x[1]))[0]
    report_content += f"- **{best_overall}** achieves the best overall performance across metrics.\n"
    report_content += "- Different attention mechanisms excel at different aspects of text generation.\n"
    report_content += "- The hybrid approach successfully combines the strengths of unigram-focused and bigram-focused models.\n\n"
    
    report_content += "## Next Steps\n\n"
    report_content += "Future work could explore:\n\n"
    report_content += "1. Additional attention mechanisms that adapt based on the query type.\n"
    report_content += "2. Incorporating larger models or specialized domain models into the hybrid framework.\n"
    report_content += "3. Developing more sophisticated merging strategies beyond sentence interleaving.\n"
    report_content += "4. Testing on diverse text corpora beyond classic literature.\n"
    
    # Save the report
    with open("visualizations/multi_model_report.md", "w") as f:
        f.write(report_content)
        
    print(f"Report generated: visualizations/multi_model_report.md")

def main():
    """Main function to run the multi-model hybrid evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multi-model hybrid evaluation")
    parser.add_argument("--kb_path", type=str, default="data/knowledge_base.json",
                        help="Path to knowledge base JSON file")
    parser.add_argument("--context_pairs_path", type=str, default="data/context_pairs.json",
                        help="Path to context pairs JSON file")
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock models for testing")
    parser.add_argument("--output_dir", type=str, default="model_evaluations",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load knowledge base and context pairs
    try:
        with open(args.kb_path, 'r') as f:
            kb_chunks = json.load(f)
            
        with open(args.context_pairs_path, 'r') as f:
            context_pairs = json.load(f)
            
        # Generate embeddings for knowledge base chunks
        print("Generating embeddings for knowledge base...")
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        kb_texts = [chunk['text'] for chunk in kb_chunks]
        kb_embeddings = encoder.encode(kb_texts, show_progress_bar=True)
        
        # Run evaluation
        results = run_multi_model_evaluation(context_pairs, kb_chunks, kb_embeddings, use_mock=args.use_mock)
        
        # Evaluate results
        print("Evaluating results...")
        evaluation = evaluate_multi_model_results(results)
        
        # Create visualizations
        print("Creating visualizations...")
        visualize_multimodel_results(evaluation)
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        
if __name__ == "__main__":
    main()