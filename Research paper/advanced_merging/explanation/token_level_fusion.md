# Token-level Fusion

## Detailed Approach

Token-level fusion is a granular approach to merging model outputs that operates at the token (word) level rather than the sentence level. This fine-grained fusion enables more coherent integration of information from multiple language models.

### Theoretical Foundation

Token-level fusion draws inspiration from:
- **Machine Translation**: Word alignment techniques used in machine translation
- **Ensemble Methods**: Token-level voting in ensemble models
- **Text Interpolation**: Methods for smooth blending of text from multiple sources

### Implementation Steps

#### 1. Tokenization and Alignment

First, both model outputs are tokenized and aligned to identify corresponding segments:

```python
def align_tokens(response1, response2):
    """
    Align tokens between two model responses using dynamic programming.
    """
    # Tokenize both responses
    tokens1 = nltk.word_tokenize(response1)
    tokens2 = nltk.word_tokenize(response2)
    
    # Create embedding representations for tokens
    embeddings1 = embedding_model.encode([token for token in tokens1], convert_to_tensor=True)
    embeddings2 = embedding_model.encode([token for token in tokens2], convert_to_tensor=True)
    
    # Calculate similarity matrix between all tokens
    similarity_matrix = torch.matmul(embeddings1, embeddings2.transpose(0, 1))
    
    # Use dynamic programming to find optimal alignment
    # This is a simplified version - actual implementation would be more complex
    alignments = []
    for i, token1 in enumerate(tokens1):
        # Find most similar token in response2
        best_match = similarity_matrix[i].argmax().item()
        alignments.append((i, best_match))
    
    return tokens1, tokens2, alignments
```

#### 2. Token Selection Strategy

Next, for each aligned segment, decide which model's tokens to select:

```python
def select_tokens(tokens1, tokens2, alignments, weights, confidence_scores=None):
    """
    Select tokens from either model based on weights and confidence scores.
    """
    selected_tokens = []
    covered_indices2 = set()
    
    for i, (idx1, idx2) in enumerate(alignments):
        # Default to weights if no confidence scores provided
        if confidence_scores is None:
            score1 = weights[0]
            score2 = weights[1]
        else:
            score1 = weights[0] * confidence_scores[0][idx1]
            score2 = weights[1] * confidence_scores[1][idx2]
        
        # Select token with higher score
        if score1 > score2 and tokens1[idx1] not in selected_tokens:
            selected_tokens.append(tokens1[idx1])
        elif tokens2[idx2] not in selected_tokens and idx2 not in covered_indices2:
            selected_tokens.append(tokens2[idx2])
            covered_indices2.add(idx2)
    
    # Add any unaligned tokens from model 2 that might be important
    for i, token in enumerate(tokens2):
        if i not in covered_indices2 and token not in selected_tokens:
            # Only add if it's a meaningful token (not punctuation or stopword)
            if is_meaningful_token(token):
                selected_tokens.append(token)
    
    return selected_tokens
```

#### 3. Grammatical Correction and Coherence Enhancement

After selecting tokens, ensure grammatical correctness and coherence:

```python
def enhance_coherence(tokens):
    """
    Apply post-processing to ensure grammatical correctness and coherence.
    """
    # 1. Check for missing determiners, prepositions, etc.
    tokens = add_missing_function_words(tokens)
    
    # 2. Ensure proper capitalization
    tokens = fix_capitalization(tokens)
    
    # 3. Correct verb tense consistency
    tokens = ensure_tense_consistency(tokens)
    
    # 4. Add appropriate punctuation
    tokens = add_punctuation(tokens)
    
    # 5. Use a language model to fill in any gaps or smooth transitions
    tokens = smooth_transitions(tokens)
    
    return tokens
```

#### 4. Confidence Scoring

Add a confidence scoring mechanism to improve token selection:

```python
def generate_confidence_scores(model, tokens, original_response):
    """
    Generate confidence scores for each token in the response.
    """
    # Method 1: Use model logits/probabilities if available
    if hasattr(model, 'get_token_probabilities'):
        return model.get_token_probabilities(original_response)
    
    # Method 2: Use perplexity as a proxy for confidence
    scores = []
    for i in range(len(tokens)):
        # Create a context without current token
        context = ' '.join(tokens[:i]) if i > 0 else ""
        next_token = tokens[i]
        target_context = context + " " + next_token
        
        # Calculate perplexity
        perplexity = calculate_perplexity(model, target_context)
        # Convert to a confidence score (lower perplexity = higher confidence)
        confidence = 1.0 / (1.0 + perplexity)
        scores.append(confidence)
    
    return scores
```

### Complete Token-level Fusion Algorithm

Putting it all together, the complete token-level fusion process is:

```python
def token_level_fusion(response1, response2, weights=(0.5, 0.5), models=None):
    """
    Fuse two model responses at the token level.
    
    Args:
        response1: Text response from first model
        response2: Text response from second model
        weights: Importance weights for each model (default: equal weighting)
        models: Optional, the original models for confidence scoring
        
    Returns:
        Fused response text
    """
    # 1. Tokenize and align responses
    tokens1, tokens2, alignments = align_tokens(response1, response2)
    
    # 2. Generate confidence scores if models are provided
    confidence_scores = None
    if models is not None:
        confidence_scores = [
            generate_confidence_scores(models[0], tokens1, response1),
            generate_confidence_scores(models[1], tokens2, response2)
        ]
    
    # 3. Select tokens based on weights and confidence
    selected_tokens = select_tokens(tokens1, tokens2, alignments, weights, confidence_scores)
    
    # 4. Enhance coherence and grammaticality
    enhanced_tokens = enhance_coherence(selected_tokens)
    
    # 5. Join tokens to form final response
    fused_response = ' '.join(enhanced_tokens)
    
    # 6. Final cleanup (remove double spaces, fix punctuation spacing)
    fused_response = cleanup_text(fused_response)
    
    return fused_response
```

## Advantages Over Sentence-level Merging

1. **Greater Precision**: Can select the best information at word level rather than entire sentences
2. **Improved Flow**: Avoids abrupt transitions between different writing styles
3. **Reduced Redundancy**: Can eliminate duplicate information more effectively
4. **Contradiction Resolution**: Can resolve contradictory statements at a fine-grained level
5. **Better Utilization of Model Strengths**: Can leverage each model's strengths for specific words or phrases

## Challenges and Solutions

### Challenge 1: Computational Complexity
- **Challenge**: Token alignment is computationally expensive, especially for long responses
- **Solution**: Use efficient alignment algorithms and parallel processing for longer texts

### Challenge 2: Grammatical Correctness
- **Challenge**: Selecting tokens from different sources may break grammatical structure
- **Solution**: Apply grammatical correction post-processing and use language models to ensure fluency

### Challenge 3: Semantic Preservation
- **Challenge**: Token-level merging may disrupt semantic meaning
- **Solution**: Use contextual embeddings to ensure semantic coherence and apply consistency checks

## Evaluation Approach

To evaluate token-level fusion against sentence-level merging:

1. **Automated Metrics**:
   - Standard metrics (ROUGE, BLEU)
   - Grammaticality metrics
   - Perplexity using a separate language model

2. **Human Evaluation**:
   - Fluency rating (1-5 scale)
   - Coherence rating (1-5 scale)
   - Information accuracy (compared to ground truth)
   - A/B testing against sentence-level merged responses
