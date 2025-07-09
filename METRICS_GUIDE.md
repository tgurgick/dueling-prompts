# Prompt Duel Metrics Guide

This guide explains all available metrics in the Prompt Duel tool, their applications, what good scores look like, how to interpret results, and their limitations.

## Table of Contents

1. [Comparative Metrics](#comparative-metrics)
2. [Similarity Metrics](#similarity-metrics)
3. [NLP Evaluation Metrics](#nlp-evaluation-metrics)
4. [LLM Judge Metrics](#llm-judge-metrics)
5. [Safety Metrics](#safety-metrics)
6. [Metric Selection Guide](#metric-selection-guide)
7. [Interpreting Results](#interpreting-results)

---

## Comparative Metrics

### Exact Match
**What it measures**: Whether the response exactly matches the expected output (case-insensitive).

**Application**: 
- Factual accuracy testing
- Code generation validation
- Structured data extraction
- Compliance checking

**Good score**: 1.0 (perfect match) or 0.0 (no match)

**Interpretation**:
- **1.0**: Response exactly matches expected output
- **0.0**: Response differs from expected output
- Binary metric - no partial credit

**Limitations**:
- Very strict - minor differences result in 0.0
- Doesn't account for equivalent but differently worded responses
- May not be suitable for creative or open-ended tasks

**Example**:
```
Expected: "The capital of France is Paris"
Response: "The capital of France is Paris" → 1.0
Response: "Paris is the capital of France" → 0.0
```

### Contains Check
**What it measures**: Whether the expected text is contained within the response.

**Application**:
- Key information verification
- Fact checking
- Content validation
- Safety content detection

**Good score**: 1.0 (contains) or 0.0 (doesn't contain)

**Interpretation**:
- **1.0**: Expected text is found in response
- **0.0**: Expected text is not found in response
- Binary metric - no partial credit

**Limitations**:
- Doesn't measure quality of surrounding context
- May miss paraphrased or reworded content
- Sensitive to exact phrasing

**Example**:
```
Expected: "machine learning"
Response: "Machine learning is a subset of AI" → 1.0
Response: "AI includes various techniques" → 0.0
```

---

## Similarity Metrics

### Relevance
**What it measures**: Semantic relevance between response and expected output using sentence transformers.

**Application**:
- Content quality assessment
- Topic alignment verification
- Response appropriateness
- General text evaluation

**Good score**: 0.7-1.0 (high relevance), 0.4-0.7 (moderate), 0.0-0.4 (low)

**Interpretation**:
- **0.8-1.0**: Highly relevant responses
- **0.6-0.8**: Good relevance
- **0.4-0.6**: Moderate relevance
- **0.0-0.4**: Low relevance

**Limitations**:
- Depends on the underlying sentence transformer model
- May not capture domain-specific nuances
- Sensitive to model quality and training data

**Model**: `all-MiniLM-L6-v2` (default)

### Semantic Similarity
**What it measures**: Cosine similarity between response and expected output embeddings.

**Application**:
- Meaning preservation verification
- Semantic equivalence testing
- Content similarity assessment
- Quality control

**Good score**: 0.7-1.0 (high similarity), 0.4-0.7 (moderate), 0.0-0.4 (low)

**Interpretation**:
- **0.8-1.0**: Very similar meanings
- **0.6-0.8**: Similar meanings
- **0.4-0.6**: Somewhat similar
- **0.0-0.4**: Different meanings

**Limitations**:
- May not distinguish between different valid interpretations
- Sensitive to embedding model quality
- Doesn't account for factual accuracy

**Model**: `all-MiniLM-L6-v2` (default)

---

## NLP Evaluation Metrics

### BLEU (Bilingual Evaluation Understudy)
**What it measures**: N-gram overlap between response and reference text.

**Application**:
- Translation quality assessment
- Text generation evaluation
- Summarization quality
- Content similarity

**Good score**: 0.3-1.0 (high), 0.1-0.3 (moderate), 0.0-0.1 (low)

**Interpretation**:
- **0.4+**: High n-gram overlap
- **0.2-0.4**: Moderate overlap
- **0.0-0.2**: Low overlap
- **0.0**: No overlap

**Limitations**:
- Doesn't measure semantic meaning
- May penalize valid paraphrases
- Sensitive to reference quality
- Doesn't account for fluency

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
**What it measures**: Word overlap between response and reference text.

**Application**:
- Summarization evaluation
- Text generation assessment
- Content extraction verification
- Information retrieval

**Good score**: 0.3-1.0 (high), 0.1-0.3 (moderate), 0.0-0.1 (low)

**Interpretation**:
- **0.4+**: High word overlap
- **0.2-0.4**: Moderate overlap
- **0.0-0.2**: Low overlap
- **0.0**: No overlap

**Limitations**:
- Focuses on word overlap, not meaning
- May miss semantic equivalence
- Sensitive to reference quality
- Doesn't measure coherence

**Types**: `rouge1` (unigrams), `rouge2` (bigrams), `rougeL` (longest common subsequence)

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
**What it measures**: Synonym-aware text similarity with explicit word matching.

**Application**:
- Translation evaluation
- Paraphrase detection
- Content similarity assessment
- Cross-lingual evaluation

**Good score**: 0.5-1.0 (high), 0.3-0.5 (moderate), 0.0-0.3 (low)

**Interpretation**:
- **0.6+**: High similarity with synonym awareness
- **0.4-0.6**: Moderate similarity
- **0.0-0.4**: Low similarity

**Limitations**:
- Requires synonym resources
- May not work well for all languages
- Sensitive to reference quality

### BERTScore
**What it measures**: Contextual similarity using BERT embeddings.

**Application**:
- Semantic similarity assessment
- Content quality evaluation
- Meaning preservation verification
- Advanced text comparison

**Good score**: 0.8-1.0 (high), 0.6-0.8 (moderate), 0.0-0.6 (low)

**Interpretation**:
- **0.9+**: Very high semantic similarity
- **0.7-0.9**: High similarity
- **0.5-0.7**: Moderate similarity
- **0.0-0.5**: Low similarity

**Limitations**:
- Computationally expensive
- Sensitive to BERT model quality
- May not capture domain-specific nuances

**Model**: `microsoft/deberta-xlarge-mnli` (default)

### BLEURT (BLEU + BERT)
**What it measures**: Combined n-gram overlap and BERT-based semantic similarity.

**Application**:
- Comprehensive text evaluation
- Quality assessment
- Content similarity verification
- Advanced comparison

**Good score**: 0.6-1.0 (high), 0.4-0.6 (moderate), 0.0-0.4 (low)

**Interpretation**:
- **0.7+**: High quality with good overlap and semantics
- **0.5-0.7**: Moderate quality
- **0.0-0.5**: Low quality

**Limitations**:
- Requires BLEURT model
- Computationally intensive
- May not be suitable for all domains

### TER (Translation Edit Rate)
**What it measures**: Edit distance between response and reference text.

**Application**:
- Translation quality assessment
- Text similarity evaluation
- Error analysis
- Content comparison

**Good score**: 0.0-0.3 (low edit rate = good), 0.3-0.6 (moderate), 0.6-1.0 (high edit rate = poor)

**Interpretation**:
- **0.0-0.2**: Very similar (few edits needed)
- **0.2-0.4**: Somewhat similar
- **0.4+**: Different (many edits needed)

**Limitations**:
- Doesn't account for semantic meaning
- May penalize valid rephrasing
- Sensitive to word order

---

## LLM Judge Metrics

### LLM Judge
**What it measures**: AI-powered comparative evaluation between two responses.

**Application**:
- Quality comparison
- Preference testing
- Subjective evaluation
- Human-like assessment

**Good score**: 1.0 (wins), 0.5 (tie), 0.0 (loses)

**Interpretation**:
- **1.0**: Response A wins
- **0.5**: Tie between responses
- **0.0**: Response B wins

**Limitations**:
- Subjective and model-dependent
- May be inconsistent across runs
- Sensitive to prompt quality
- Requires API calls

**Model**: `gpt-4o` (default)

---

## Safety Metrics

### Safety Judge
**What it measures**: Safety evaluation of responses for potential harm, bias, or inappropriate content.

**Application**:
- Content moderation
- Safety filtering
- Bias detection
- Harmful content identification

**Good score**: 0.8-1.0 (safe), 0.5-0.8 (moderate), 0.0-0.5 (unsafe)

**Interpretation**:
- **0.9+**: Very safe content
- **0.7-0.9**: Safe content
- **0.5-0.7**: Moderate safety concerns
- **0.0-0.5**: Unsafe content

**Limitations**:
- Subjective safety definitions
- May have cultural biases
- Sensitive to prompt quality
- Requires API calls

**Model**: `gpt-4o-mini` (default)

---

## Metric Selection Guide

### For Different Use Cases

**Content Generation**:
- Primary: Relevance, Semantic Similarity
- Secondary: BLEU, ROUGE, BERTScore
- Safety: Safety Judge

**Translation/Summarization**:
- Primary: BLEU, ROUGE, METEOR
- Secondary: BERTScore, Semantic Similarity
- Quality: LLM Judge

**Factual Accuracy**:
- Primary: Exact Match, Contains Check
- Secondary: Relevance, Semantic Similarity
- Verification: LLM Judge

**Creative Writing**:
- Primary: LLM Judge, Relevance
- Secondary: Semantic Similarity
- Safety: Safety Judge

**Code Generation**:
- Primary: Exact Match, Contains Check
- Secondary: LLM Judge
- Quality: Relevance

### Metric Combinations

**Comprehensive Evaluation**:
```
- Relevance (semantic quality)
- Semantic Similarity (meaning preservation)
- BLEU (n-gram overlap)
- LLM Judge (overall quality)
- Safety Judge (safety check)
```

**Quality-Focused**:
```
- LLM Judge (primary quality)
- Relevance (semantic appropriateness)
- Semantic Similarity (meaning alignment)
```

**Safety-Focused**:
```
- Safety Judge (primary safety)
- Relevance (appropriate content)
- LLM Judge (overall assessment)
```

---

## Interpreting Results

### Win Thresholds

The Prompt Duel tool uses **metric-specific win thresholds** to determine when one prompt truly outperforms another:

**Binary Metrics** (0.001 threshold):
- **Exact Match**: Very strict - only exact matches win
- **Contains Check**: Very strict - only exact containment wins

**Similarity Metrics** (0.05 threshold):
- **Relevance**: Moderate sensitivity - 0.05 difference needed
- **Semantic Similarity**: Moderate sensitivity - 0.05 difference needed

**NLP Metrics** (0.01-0.03 threshold):
- **BLEU**: 0.02 threshold - sensitive to small differences
- **ROUGE**: 0.01 threshold - very sensitive due to typical low scores (0.1-0.4 range)
- **METEOR**: 0.03 threshold - moderate sensitivity
- **BERTScore**: 0.02 threshold - sensitive to small differences
- **BLEURT**: 0.02 threshold - sensitive to small differences
- **TER**: 0.05 threshold - lower scores are better

**LLM Judge Metrics** (0.1 threshold):
- **LLM Judge**: 0.1 threshold - moderate sensitivity
- **Safety Judge**: 0.1 threshold - moderate sensitivity

### Score Ranges

**Excellent (0.8-1.0)**:
- High-quality responses
- Strong performance across metrics
- Minimal room for improvement

**Good (0.6-0.8)**:
- Solid performance
- Some areas for improvement
- Generally acceptable quality

**Moderate (0.4-0.6)**:
- Mixed performance
- Clear areas for improvement
- May need prompt refinement

**Poor (0.0-0.4)**:
- Significant issues
- Major prompt redesign needed
- May indicate fundamental problems

### Cross-Metric Analysis

**Consistent High Scores**:
- Indicates robust prompt design
- Good across different evaluation criteria
- Reliable performance

**Inconsistent Scores**:
- May indicate prompt weaknesses
- Different metrics revealing different aspects
- Need for targeted improvements

**All Low Scores**:
- Fundamental prompt issues
- May need complete redesign
- Consider different approach

### Control Prompt Analysis

**Good Prompt vs Random**:
- Should consistently beat random
- If not, experiment may be flawed
- Random serves as baseline validation

**Random Performance**:
- Provides context for other scores
- Helps calibrate expectations
- Validates experiment design

### Statistical Insights

**Wins vs Average Scores**:
- Wins show relative performance
- Average scores show absolute performance
- Both important for complete picture

**Metric Sensitivity**:
- Some metrics may be too strict/lenient
- Consider adjusting thresholds
- May need metric-specific analysis

---

## Best Practices

### Metric Selection
1. **Start with relevant metrics** for your use case
2. **Include a control prompt** (automatic in Prompt Duel)
3. **Use multiple metrics** for comprehensive evaluation
4. **Consider your specific requirements**

### Interpretation
1. **Look at both wins and scores**
2. **Compare against control prompt**
3. **Consider metric limitations**
4. **Focus on actionable insights**

### Experiment Design
1. **Use diverse test cases**
2. **Include edge cases**
3. **Test with different inputs**
4. **Validate with human evaluation**

### Continuous Improvement
1. **Iterate based on results**
2. **Refine prompts systematically**
3. **Track improvements over time**
4. **Consider domain-specific metrics**

---

## Troubleshooting

### Common Issues

**All Ties (0.5 scores)**:
- Metrics may not be sensitive enough
- Consider adjusting thresholds
- May need different metrics

**Consistent Low Scores**:
- Prompt may need fundamental redesign
- Test cases may be too challenging
- Consider different approach

**Inconsistent Results**:
- May indicate prompt instability
- Consider temperature settings
- May need more test cases

**High Variance**:
- May need more test cases
- Consider prompt stability
- May indicate random behavior

### Metric-Specific Issues

**Exact Match Always 0**:
- May be too strict for your use case
- Consider Contains Check instead
- May need more flexible evaluation

**LLM Judge Inconsistency**:
- May need better judge prompts
- Consider multiple judge calls
- May need different judge model

**Safety Judge Concerns**:
- May be too strict/lenient
- Consider domain-specific safety
- May need custom safety criteria

**All Ties in NLP Metrics**:
- May indicate threshold too high
- Consider metric-specific sensitivity
- May need different evaluation approach

**High Variance in Similarity Metrics**:
- May indicate threshold too low
- Consider increasing threshold
- May need more test cases

---

This guide should help you understand and effectively use all available metrics in the Prompt Duel tool. Remember that the best metric combination depends on your specific use case and requirements. 