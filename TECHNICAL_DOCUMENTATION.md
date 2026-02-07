# Hallucination Hunter: Technical Documentation
## AI-Powered Fact-Checking & Attribution System

---

## Executive Summary

The Hallucination Hunter is an AI auditing system designed to verify LLM-generated content against trusted source documents. It addresses the critical challenge of hallucinations in enterprise AI applications by providing automated fact-checking, citation generation, and explainable verification results.

**Key Metrics:**
- Detection Accuracy: F1-Score optimized for high recall (avoiding false negatives)
- Citation Precision: Direct evidence linking with relevance scoring
- Processing Latency: <5 seconds for standard documents
- Trust Score: Normalized confidence metric (0-100%)

---

## System Architecture

### Pipeline Overview

```
┌─────────────────────┐
│  Source Documents   │
│   (Ground Truth)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌──────────────────┐
│  LLM-Generated      │────▶ │ Claim Extraction │
│     Summary         │      │    (Atomic)      │
└─────────────────────┘      └────────┬─────────┘
                                      │
                                      ▼
                          ┌─────────────────────┐
                          │  RAG Retrieval      │
                          │ (Semantic Search)   │
                          └────────┬────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │  NLI Verification   │
                          │ (BERT/RoBERTa)      │
                          └────────┬────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │  Classification     │
                          │  - Supported        │
                          │  - Contradicted     │
                          │  - Unverifiable     │
                          └────────┬────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │  Annotation &       │
                          │  Explainability     │
                          └─────────────────────┘
```

---

## Component Details

### 1. Claim Extraction Module

**Purpose:** Decompose LLM output into atomic, verifiable factual claims.

**Implementation:**
```python
def extract_claims(text):
    """
    Extract atomic claims from generated text.
    
    Strategy:
    1. Sentence tokenization (using spaCy or NLTK)
    2. Dependency parsing to identify sub-claims
    3. Compound sentence splitting on conjunctions
    4. Filter out non-factual statements (opinions, questions)
    
    Returns: List of Claim objects with position indices
    """
    sentences = sentence_tokenize(text)
    claims = []
    
    for sent in sentences:
        # Parse dependencies
        doc = nlp(sent)
        
        # Split on conjunctions (and, or, but)
        sub_claims = split_on_conjunctions(doc)
        
        for sub in sub_claims:
            if is_factual(sub):  # Filter opinions/questions
                claims.append({
                    'text': sub.text,
                    'start': sub.start_char,
                    'end': sub.end_char,
                    'type': classify_claim_type(sub)  # numerical, entity, event
                })
    
    return claims
```

**Key Considerations:**
- Atomic claims should be independently verifiable
- Preserve context for claims that depend on previous sentences
- Handle co-reference resolution (pronouns → entities)

---

### 2. RAG (Retrieval-Augmented Generation) Module

**Purpose:** Find relevant passages in source documents for each claim.

**Implementation:**
```python
import faiss
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self, source_documents):
        # Use lightweight embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Chunk source documents
        self.chunks = self.chunk_documents(source_documents)
        
        # Create FAISS index
        embeddings = self.encoder.encode(self.chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
    
    def retrieve(self, claim, k=3):
        """
        Retrieve top-k most relevant source passages.
        
        Returns: List of (chunk, score) tuples
        """
        query_embedding = self.encoder.encode([claim])
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'text': self.chunks[idx],
                'relevance': 1 / (1 + dist),  # Convert distance to similarity
                'source_id': self.chunk_metadata[idx]['source'],
                'line_number': self.chunk_metadata[idx]['line']
            })
        
        return results
```

**Optimization:**
- Chunk size: 256-512 tokens (balance context vs. precision)
- Overlap: 50 tokens between chunks to avoid boundary issues
- Model: `all-MiniLM-L6-v2` (384-dim, fast inference)

---

### 3. NLI (Natural Language Inference) Verification

**Purpose:** Classify the relationship between claim and evidence.

**Implementation:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class NLIVerifier:
    def __init__(self):
        # Load lightweight NLI model
        self.model_name = "microsoft/deberta-v3-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        
    def verify(self, claim, evidence):
        """
        Classify claim-evidence relationship.
        
        Returns: {
            'label': 'entailment' | 'contradiction' | 'neutral',
            'confidence': float (0-1),
            'logits': tensor
        }
        """
        # Format as premise-hypothesis pair
        inputs = self.tokenizer(
            evidence,  # premise
            claim,     # hypothesis
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Map to our categories
        label_map = {
            'entailment': 'supported',
            'contradiction': 'contradicted',
            'neutral': 'unverifiable'
        }
        
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0, pred_idx].item()
        
        return {
            'label': label_map[self.model.config.id2label[pred_idx]],
            'confidence': confidence
        }
```

**Model Selection:**
- **Primary:** DeBERTa-v3-small (Fine-tuned on MNLI)
- **Alternative:** RoBERTa-large-MNLI
- **Lightweight:** DistilBERT-base-uncased-MNLI (for speed)

**Fine-Tuning Strategy:**
```python
# Fine-tune on domain-specific data
def fine_tune_nli(base_model, domain_data):
    """
    domain_data: HaluEval or custom annotated examples
    Format: (claim, evidence, label) tuples
    """
    trainer = Trainer(
        model=base_model,
        args=TrainingArguments(
            per_device_train_batch_size=16,
            num_train_epochs=3,
            learning_rate=2e-5,
            evaluation_strategy="epoch"
        ),
        train_dataset=domain_data['train'],
        eval_dataset=domain_data['validation']
    )
    
    trainer.train()
    return trainer.model
```

---

### 4. Classification & Scoring

**Categories:**

1. **SUPPORTED (Green)**
   - NLI: Entailment
   - Confidence threshold: ≥0.8
   - Evidence: Direct textual match or paraphrase

2. **CONTRADICTED (Red)**
   - NLI: Contradiction
   - Confidence threshold: ≥0.75
   - Evidence: Explicit conflict with source

3. **UNVERIFIABLE (Yellow)**
   - NLI: Neutral OR low confidence
   - No sufficient evidence in source documents

**Trust Score Calculation:**
```python
def calculate_trust_score(claims):
    """
    Weighted trust score considering claim importance.
    
    Formula:
    TrustScore = (Σ supported_confidence - Σ contradicted_confidence) / n_claims
    Normalized to 0-100 scale.
    """
    total_score = 0
    
    for claim in claims:
        if claim.status == 'supported':
            total_score += claim.confidence
        elif claim.status == 'contradicted':
            total_score -= claim.confidence * 1.5  # Penalize hallucinations more
        else:  # unverifiable
            total_score += 0  # Neutral
    
    # Normalize to [0, 100]
    normalized = ((total_score / len(claims)) + 1) / 2 * 100
    return max(0, min(100, normalized))
```

---

### 5. Explainability Engine

**Components:**

1. **Evidence Extraction**
   ```python
   def extract_evidence(source_text, claim):
       """
       Extract minimal supporting passage.
       """
       # Use named entity recognition to find key terms
       claim_entities = extract_entities(claim)
       
       # Find sentences containing these entities
       evidence_sentences = []
       for sent in source_sentences:
           if has_entity_overlap(sent, claim_entities):
               evidence_sentences.append(sent)
       
       # Rank by semantic similarity
       ranked = rank_by_similarity(evidence_sentences, claim)
       
       # Return top match with context window
       return get_context_window(ranked[0], window=2)
   ```

2. **Explanation Generation**
   ```python
   def generate_explanation(claim, evidence, status):
       """
       Template-based explanation with specifics filled in.
       """
       templates = {
           'supported': "The claim '{claim}' is supported by the source: \"{evidence}\"",
           'contradicted': "The claim states '{claim_fact}' but the source indicates '{source_fact}'",
           'unverifiable': "No information found in source documents to verify '{claim}'"
       }
       
       # Extract key facts for contradiction case
       if status == 'contradicted':
           claim_fact = extract_key_fact(claim)
           source_fact = extract_key_fact(evidence)
           return templates[status].format(
               claim_fact=claim_fact,
               source_fact=source_fact
           )
       
       return templates[status].format(claim=claim, evidence=evidence)
   ```

3. **Correction Suggestion**
   ```python
   def suggest_correction(claim, evidence):
       """
       Generate corrected version using evidence.
       """
       # Extract the correct fact from evidence
       correct_fact = extract_fact(evidence)
       
       # Replace hallucinated portion in claim
       corrected = claim.replace(
           find_hallucinated_span(claim, evidence),
           correct_fact
       )
       
       return corrected
   ```

---

## Evaluation Metrics

### 1. Detection Accuracy

**F1-Score with emphasis on Recall:**
```python
def evaluate_detection(predictions, ground_truth):
    """
    Prioritize avoiding False Negatives (missed hallucinations).
    """
    TP = sum((p == 'contradicted' and g == 'contradicted') 
             for p, g in zip(predictions, ground_truth))
    FP = sum((p == 'contradicted' and g != 'contradicted') 
             for p, g in zip(predictions, ground_truth))
    FN = sum((p != 'contradicted' and g == 'contradicted') 
             for p, g in zip(predictions, ground_truth))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Weighted F1 favoring recall
    f1_weighted = 0.3 * precision + 0.7 * recall
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_weighted': f1_weighted
    }
```

### 2. Citation Precision

```python
def evaluate_citations(predicted_citations, gold_citations):
    """
    Measure citation quality using ROUGE-L and exact match.
    """
    from rouge import Rouge
    rouge = Rouge()
    
    scores = []
    for pred, gold in zip(predicted_citations, gold_citations):
        # Check if citation is in correct source section
        exact_match = (pred.source_id == gold.source_id)
        
        # Measure text overlap
        rouge_score = rouge.get_scores(pred.text, gold.text)[0]['rouge-l']['f']
        
        scores.append({
            'exact_match': exact_match,
            'rouge_l': rouge_score,
            'combined': 0.5 * exact_match + 0.5 * rouge_score
        })
    
    return {
        'exact_match_rate': sum(s['exact_match'] for s in scores) / len(scores),
        'average_rouge_l': sum(s['rouge_l'] for s in scores) / len(scores),
        'combined_score': sum(s['combined'] for s in scores) / len(scores)
    }
```

### 3. Explanation Quality

**Human Evaluation Rubric:**
- Clarity (1-5): Is the explanation easy to understand?
- Completeness (1-5): Does it explain why the claim is flagged?
- Actionability (1-5): Can a reviewer act on this information?

**Automated Proxy:**
```python
def evaluate_explanation_quality(explanation):
    """
    Heuristic quality check.
    """
    checks = {
        'has_specific_fact': bool(re.search(r'\d+|[A-Z][a-z]+', explanation)),
        'cites_source': 'source' in explanation.lower() or 'document' in explanation.lower(),
        'provides_alternative': 'but' in explanation or 'instead' in explanation,
        'appropriate_length': 20 < len(explanation.split()) < 100
    }
    
    return sum(checks.values()) / len(checks)
```

### 4. Latency Benchmarks

**Target Performance:**
- Claim extraction: <500ms
- RAG retrieval per claim: <200ms
- NLI verification per claim: <100ms
- Total for 1-page document (~10 claims): <3 seconds

**Optimization Techniques:**
- Batch processing for NLI inference
- FAISS GPU acceleration for large document sets
- Model quantization (INT8) for faster inference

---

## Datasets

### Training Data

1. **HaluEval** (Primary)
   - 5,000+ examples across domains
   - Format: (source, generation, label, explanation)
   - Domains: QA, dialogue, summarization

2. **FEVER** (Fact Extraction and VERification)
   - 185,000+ claims with Wikipedia evidence
   - Three-class labels: Supported, Refuted, NotEnoughInfo
   - Use for NLI fine-tuning

3. **SummEval**
   - Human annotations for summary factuality
   - Good for medical/legal domain transfer learning

### Synthetic Data Generation

```python
def generate_synthetic_hallucinations(source_doc, model):
    """
    Create training examples by deliberately introducing errors.
    """
    # Extract facts from source
    facts = extract_facts(source_doc)
    
    synthetic_examples = []
    
    for fact in facts:
        # Generate contradictory versions
        contradicted = [
            flip_boolean(fact),      # "Type 1" → "Type 2"
            modify_number(fact),     # "20 units" → "25 units"
            swap_entity(fact),       # "Dr. Smith" → "Dr. Jones"
            negate_statement(fact)   # "has allergy" → "no allergy"
        ]
        
        for hallucination in contradicted:
            synthetic_examples.append({
                'source': source_doc,
                'claim': hallucination,
                'label': 'contradicted',
                'original_fact': fact
            })
    
    return synthetic_examples
```

---

## User Interface Requirements

### Interactive Components

1. **Split-Screen View**
   ```javascript
   // Click handler for claim-to-source linking
   function handleClaimClick(claim) {
       // Highlight claim in LLM output
       highlightText(claim.id);
       
       // Auto-scroll source document to evidence
       const evidenceElement = document.getElementById(claim.evidenceId);
       evidenceElement.scrollIntoView({ behavior: 'smooth' });
       
       // Show explanation panel
       showExplanationPanel(claim.explanation, claim.confidence);
   }
   ```

2. **Confidence Meter**
   ```javascript
   // Visual trust score display
   <div className="trust-meter">
       <div className="trust-fill" style={{ width: `${trustScore}%` }}>
           {trustScore.toFixed(1)}%
       </div>
   </div>
   ```

3. **Color-Coded Annotations**
   - Green (#D1FAE5): Supported claims
   - Red (#FEE2E2): Hallucinations
   - Yellow (#FEF3C7): Unverifiable claims

### Accessibility

- Keyboard navigation for claim browsing
- Screen reader support with ARIA labels
- Color-blind safe palette with icons (not just color)

---

## Deployment Considerations

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- GPU: Optional (3x speedup with CUDA)

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA T4 or better
- Storage: 50 GB (for models and indices)

### API Design

```python
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/verify")
async def verify_document(
    source_docs: List[UploadFile],
    llm_output: str,
    threshold: float = 0.8
):
    """
    Main verification endpoint.
    
    Returns: {
        'claims': [...],
        'trust_score': float,
        'summary': {...}
    }
    """
    # Load source documents
    sources = [parse_document(doc) for doc in source_docs]
    
    # Initialize pipeline
    extractor = ClaimExtractor()
    retriever = RAGRetriever(sources)
    verifier = NLIVerifier()
    
    # Process
    claims = extractor.extract(llm_output)
    results = []
    
    for claim in claims:
        evidence = retriever.retrieve(claim.text)
        verification = verifier.verify(claim.text, evidence[0]['text'])
        
        results.append({
            'claim': claim.text,
            'status': verification['label'],
            'confidence': verification['confidence'],
            'evidence': evidence[0],
            'explanation': generate_explanation(claim, evidence[0], verification['label'])
        })
    
    return {
        'claims': results,
        'trust_score': calculate_trust_score(results),
        'summary': generate_summary(results)
    }
```

### Scalability

**For Large Document Sets:**
1. Implement caching for frequently accessed sources
2. Use distributed RAG with Elasticsearch/Pinecone
3. Load balance NLI inference across multiple GPUs
4. Implement streaming for long documents

---

## Future Enhancements

1. **Multi-Modal Support**
   - Verify claims about images/charts in source PDFs
   - OCR for scanned documents

2. **Cross-Document Verification**
   - Check consistency across multiple source versions
   - Temporal fact-checking (detect outdated information)

3. **Active Learning**
   - User feedback loop to improve model
   - Confidence calibration based on corrections

4. **Domain Adaptation**
   - Medical: UMLS terminology integration
   - Legal: Citation format standardization (Bluebook)
   - Finance: Numerical reasoning enhancement

---

## References & Resources

**Papers:**
- "HaluEval: A Large-Scale Hallucination Evaluation Benchmark" (Li et al., 2023)
- "FEVER: a large-scale dataset for Fact Extraction and VERification" (Thorne et al., 2018)
- "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection" (Laban et al., 2022)

**Models:**
- Sentence-Transformers: https://www.sbert.net/
- DeBERTa-v3: https://huggingface.co/microsoft/deberta-v3-base
- RoBERTa-MNLI: https://huggingface.co/roberta-large-mnli

**Libraries:**
- FAISS: https://github.com/facebookresearch/faiss
- Transformers: https://huggingface.co/docs/transformers
- spaCy: https://spacy.io/

---

## Contact & Support

For implementation questions or dataset access:
- GitHub: [repository-link]
- Documentation: [docs-link]
- Issues: [issues-link]

---

**Last Updated:** February 2026
**Version:** 1.0
**License:** MIT