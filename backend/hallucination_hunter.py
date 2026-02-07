import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

class ClaimStatus(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


@dataclass
class Claim:
    id: str
    text: str
    start: int
    end: int
    full_sentence: str
    claim_type: str = "general"
    key_entities: List[str] = field(default_factory=list)
    key_values: List[str] = field(default_factory=list)


@dataclass
class Evidence:
    text: str
    source_id: str
    line_number: int
    relevance_score: float
    context_before: str = ""
    context_after: str = ""
    matched_entities: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    claim: Claim
    status: ClaimStatus
    confidence: float
    evidence: List[Evidence]
    explanation: str
    correction: Optional[str] = None
    citation: Optional[str] = None  

class ClaimExtractor:
    def __init__(self):
        self.state_verbs = r'\b(is|are|was|were|has|have|had|diagnosed|takes|receiving)\b'
        
        self.split_conjunctions = ['and', 'but', 'while', 'whereas']
        
        self.non_factual = [
            r'^\s*if\b',  # Conditionals
            r'^\s*may\b',  
            r'^\s*might\b',
            r'\?$'  # Questions
        ]
    
    def extract(self, text: str) -> List[Claim]:
        """Extract atomic claims from text"""
        claims = []
        
        # Split into sentences more carefully
        sentences = self._split_sentences(text)
        
        cursor = 0
        claim_id = 0
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            
            # Skip non-factual sentences
            if self._is_non_factual(sent):
                continue
            
            # Check if sentence contains factual content
            if not self._has_factual_content(sent):
                continue
            
            # Try to split compound sentences
            sub_claims = self._split_compound(sent)
            
            for sub in sub_claims:
                # Find position in original text
                start = text.find(sub, cursor)
                if start == -1:
                    # Fallback: approximate position
                    start = cursor
                
                # Extract entities and values
                entities = self._extract_entities(sub)
                values = self._extract_values(sub)
                
                claims.append(Claim(
                    id=f"claim_{claim_id}",
                    text=sub.strip(),
                    start=start,
                    end=start + len(sub),
                    full_sentence=sent,
                    claim_type=self._classify_claim_type(sub),
                    key_entities=entities,
                    key_values=values
                ))
                
                claim_id += 1
                cursor = start + len(sub)
        
        return claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Improved sentence splitting"""
        # Handle common abbreviations
        text = re.sub(r'\bDr\.', 'Dr', text)
        text = re.sub(r'\bMr\.', 'Mr', text)
        text = re.sub(r'\bMrs\.', 'Mrs', text)
        
        # Split on sentence terminators
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_non_factual(self, sentence: str) -> bool:
        """Check if sentence is non-factual"""
        for pattern in self.non_factual:
            if re.search(pattern, sentence, re.I):
                return True
        return False
    
    def _has_factual_content(self, sentence: str) -> bool:
        """Check if sentence contains factual claims"""
        # Must have state verbs or specific entities
        if re.search(self.state_verbs, sentence, re.I):
            return True
        
        # Or contain specific medical/factual terms
        if re.search(r'\b(Type\s\d+|diagnosed|medication|units|mg|ml)\b', sentence, re.I):
            return True
        
        return False
    
    def _split_compound(self, sentence: str) -> List[str]:
        """Split compound sentences on conjunctions"""
        parts = [sentence]
        
        for conj in self.split_conjunctions:
            new_parts = []
            for part in parts:
                # Only split if both sides are substantial
                splits = re.split(rf'\s+{conj}\s+', part, maxsplit=1)
                if len(splits) == 2 and len(splits[0]) > 10 and len(splits[1]) > 10:
                    new_parts.extend(splits)
                else:
                    new_parts.append(part)
            parts = new_parts
        
        return [p.strip() for p in parts if len(p.strip()) > 5]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities (names, medical terms)"""
        entities = []
        
        # Proper names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(names)
        
        # Medical/key terms (expanded)
        medical = re.findall(
            r'\b(Diabetes|Insulin|Metformin|Penicillin|Type\s*\d+|'
            r'allerg(?:y|ies)|blood\s+pressure|normal|elevated|'
            r'follow-?up|month|week)\b', 
            text, re.I
        )
        entities.extend(medical)
        
        return list(set(entities))
    
    def _extract_values(self, text: str) -> List[str]:
        """Extract numerical values and measurements"""
        values = []
        
        # Numbers with units
        measurements = re.findall(r'\b\d+\.?\d*\s*(?:units?|mg|ml|mmHg|%|kg|lb)\b', text, re.I)
        values.extend(measurements)
        
        # Standalone numbers in context
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        values.extend(numbers)
        
        return list(set(values))
    
    def _classify_claim_type(self, text: str) -> str:
        """Classify claim type for better verification"""
        text_lower = text.lower()
        
        if re.search(r'\b\d+\.?\d*\s*(?:units?|mg|ml|mmHg|%)\b', text):
            return "numerical_measurement"
        
        if re.search(r'\btype\s\d+\b', text_lower):
            return "classification"
        
        if re.search(r'\b(diagnosed|diagnosis)\b', text_lower):
            return "diagnosis"
        
        if re.search(r'\b(medication|drug|allergy)\b', text_lower):
            return "medication"
        
        if re.search(r'\b\d+\.?\d*\b', text):
            return "numerical"
        
        return "general"


# ============================================================================
# RAG RETRIEVER - ENHANCED
# ============================================================================

class RAGRetriever:
    """
    Enhanced retriever with better chunking and context preservation.
    """
    
    def __init__(self, source_documents: List[str], chunk_overlap: int = 1):
        """
        Initialize with source documents.
        
        Args:
            source_documents: List of source texts
            chunk_overlap: Number of lines to overlap between chunks
        """
        self.chunks = []
        self.chunk_overlap = chunk_overlap
        
        # Process each source document
        for doc_id, doc in enumerate(source_documents):
            self._process_document(doc, doc_id)
    
    def _process_document(self, document: str, doc_id: int):
        """Process document into searchable chunks"""
        lines = document.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Create chunk with context
            context_before = ""
            context_after = ""
            
            if i > 0:
                context_before = lines[i-1].strip()
            
            if i < len(lines) - 1:
                context_after = lines[i+1].strip()
            
            # Extract entities from chunk for better matching
            entities = self._extract_chunk_entities(line)
            
            self.chunks.append({
                'text': line,
                'source_id': str(doc_id),
                'line': i + 1,
                'context_before': context_before,
                'context_after': context_after,
                'entities': entities
            })
    
    def _extract_chunk_entities(self, text: str) -> List[str]:
        """Extract key entities from chunk"""
        entities = []
        
        # Numbers with context
        nums = re.findall(r'\b\d+\.?\d*\s*(?:units?|mg|ml|mmHg|%|kg|months?|weeks?)?\b', text, re.I)
        entities.extend(nums)
        
        # Medical/key terms (expanded)
        terms = re.findall(
            r'\b(Type\s*\d+|Diabetes|Insulin|diagnosed|normal|elevated|'
            r'Penicillin|allerg(?:y|ies)|blood\s+pressure|follow-?up)\b', 
            text, re.I
        )
        entities.extend(terms)
        
        return list(set(e.strip() for e in entities))
    
    def retrieve(self, claim: Claim, k: int = 3) -> List[Evidence]:
        """
        Retrieve most relevant evidence for a claim.
        
        Uses multi-factor scoring:
        - Token overlap (Jaccard)
        - Entity matching
        - Value matching
        """
        scores = []
        
        for chunk in self.chunks:
            score = self._calculate_relevance(claim, chunk)
            scores.append(score)
        
        # Get top-k indices
        top_k_idx = np.argsort(scores)[-k:][::-1]
        
        evidence_list = []
        for idx in top_k_idx:
            if scores[idx] > 0.05:  # Minimum threshold
                chunk = self.chunks[idx]
                
                # Find matched entities
                matched = self._find_matched_entities(claim, chunk)
                
                evidence_list.append(Evidence(
                    text=chunk['text'],
                    source_id=chunk['source_id'],
                    line_number=chunk['line'],
                    relevance_score=scores[idx],
                    context_before=chunk['context_before'],
                    context_after=chunk['context_after'],
                    matched_entities=matched
                ))
        
        return evidence_list
    
    def _calculate_relevance(self, claim: Claim, chunk: Dict) -> float:
        """Multi-factor relevance scoring"""
        # 1. Token overlap (Jaccard)
        token_score = self._jaccard_similarity(claim.text, chunk['text'])
        
        # 2. Entity matching
        claim_entities_lower = [e.lower() for e in claim.key_entities]
        chunk_entities_lower = [e.lower() for e in chunk['entities']]
        
        entity_matches = sum(1 for e in claim_entities_lower if e in chunk_entities_lower)
        entity_score = entity_matches / max(len(claim.key_entities), 1)
        
        # 3. Value matching (exact)
        value_matches = sum(1 for v in claim.key_values if v in chunk['text'])
        value_score = value_matches / max(len(claim.key_values), 1)
        
        # Special handling for classification claims (Type X)
        if claim.claim_type == 'classification':
            # For Type claims, prioritize chunks with "Type" keyword
            if 'type' in chunk['text'].lower():
                # Strong boost for Type matches
                return 0.3 * token_score + 0.6 * entity_score + 0.1 * value_score
            else:
                # Penalize chunks without Type keyword
                return 0.3 * token_score + 0.3 * entity_score + 0.1 * value_score
        
        # Weighted combination for other claim types
        # Prioritize exact value matches for numerical claims
        if claim.claim_type.startswith('numerical'):
            return 0.2 * token_score + 0.3 * entity_score + 0.5 * value_score
        else:
            return 0.4 * token_score + 0.4 * entity_score + 0.2 * value_score
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity between two texts"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_matched_entities(self, claim: Claim, chunk: Dict) -> List[str]:
        """Find which entities matched between claim and chunk"""
        matched = []
        chunk_text_lower = chunk['text'].lower()
        
        for entity in claim.key_entities:
            if entity.lower() in chunk_text_lower:
                matched.append(entity)
        
        for value in claim.key_values:
            if value in chunk['text']:
                matched.append(value)
        
        return matched


# ============================================================================
# NLI VERIFIER - PRECISION ENHANCED
# ============================================================================

class NLIVerifier:
    """
    Enhanced NLI verifier with precise contradiction detection.
    
    Improvements:
    - Specific pattern matching for common contradictions
    - Confidence calibration
    - Better handling of numerical claims
    """
    
    def __init__(self, threshold_supported: float = 0.7, threshold_contradicted: float = 0.75):
        self.threshold_supported = threshold_supported
        self.threshold_contradicted = threshold_contradicted
        
        # Contradiction patterns (claim_pattern, evidence_pattern, description)
        self.contradiction_rules = [
            # Type mismatches
            (r'Type\s*(\d+)', r'Type\s*(\d+)', 'diabetes_type'),
            
            # Numerical mismatches (units, dosages)
            (r'(\d+\.?\d*)\s*units?', r'(\d+\.?\d*)\s*units?', 'dosage'),
            (r'(\d+\.?\d*)\s*mg', r'(\d+\.?\d*)\s*mg', 'dosage_mg'),
            (r'(\d+\.?\d*)\s*(?:months?|weeks?)', r'(\d+\.?\d*)\s*(?:months?|weeks?)', 'timeframe'),
            
            # Status contradictions
            (r'\bnormal\b', r'\b(elevated|high|low|abnormal|slightly\s+elevated)\b', 'status_normal_vs_abnormal'),
            (r'\belevated\b', r'\bnormal\b', 'status_elevated_vs_normal'),
            (r'\b(slightly\s+)?elevated\b', r'\bnormal\b', 'status_elevated_vs_normal2'),
            
            # Negation contradictions - handled separately below
            # (r'\bno\s+(?:known\s+)?(\w+)', r'\b\1\b', 'negation_presence'),
        ]
    
    def verify(self, claim: Claim, evidence: Evidence) -> Tuple[ClaimStatus, float]:
        """
        Verify claim against evidence with high precision.
        
        Returns: (status, confidence)
        """
        claim_text = claim.text.lower()
        evidence_text = evidence.text.lower()
        
        # Step 1: Check for HARD contradictions
        contradiction_result = self._check_contradictions(claim, evidence)
        if contradiction_result:
            return contradiction_result
        
        # Step 2: Check for strong support
        support_result = self._check_support(claim, evidence)
        if support_result:
            return support_result
        
        # Step 3: Default to unverifiable with low confidence
        return ClaimStatus.UNVERIFIABLE, evidence.relevance_score * 0.5
    
    def _check_contradictions(self, claim: Claim, evidence: Evidence) -> Optional[Tuple[ClaimStatus, float]]:
        """Check for contradictions with high precision"""
        claim_text = claim.text.lower()
        evidence_text = evidence.text.lower()
        
        for claim_pattern, evidence_pattern, rule_type in self.contradiction_rules:
            claim_match = re.search(claim_pattern, claim_text, re.I)
            
            # For Type diabetes, check if evidence has different type anywhere
            if rule_type == 'diabetes_type' and claim_match:
                # Look for any Type X in evidence
                all_evidence_types = re.findall(r'Type\s*(\d+)', evidence_text, re.I)
                if all_evidence_types:
                    claim_type_num = claim_match.group(1)
                    # If ANY evidence type differs, it's a contradiction
                    if claim_type_num not in all_evidence_types:
                        confidence = 0.95
                        return ClaimStatus.CONTRADICTED, confidence
                continue
            
            evidence_match = re.search(evidence_pattern, evidence_text, re.I)
            
            if not (claim_match and evidence_match):
                continue
            
            # Rule-specific contradiction checking
            if rule_type in ['diabetes_type', 'dosage', 'dosage_mg', 'timeframe']:
                # Extract numerical values
                claim_val = claim_match.group(1)
                evidence_val = evidence_match.group(1)
                
                if claim_val != evidence_val:
                    # HARD contradiction detected
                    confidence = 0.95
                    return ClaimStatus.CONTRADICTED, confidence
            
            elif rule_type == 'status_normal_vs_abnormal':
                # Normal vs elevated/high/low/abnormal
                confidence = 0.92
                return ClaimStatus.CONTRADICTED, confidence
            
            elif rule_type == 'status_elevated_vs_normal':
                # Elevated vs normal
                confidence = 0.90
                return ClaimStatus.CONTRADICTED, confidence
            
            elif rule_type == 'status_elevated_vs_normal2':
                # Slightly elevated vs normal
                confidence = 0.88
                return ClaimStatus.CONTRADICTED, confidence
        
        # Special handling for negation contradictions
        # Check if claim says "no X" but evidence mentions "X"
        no_match = re.search(r'\bno\s+(?:known\s+)?(?:\w+\s+)?(\w+ies|allergy)', claim_text, re.I)
        if no_match:
            claimed_absent = no_match.group(1)
            # Check if the thing claimed absent is actually present in evidence
            if re.search(rf'\b{claimed_absent}\b', evidence_text, re.I) and 'no ' not in evidence_text:
                # Found evidence of what's claimed to be absent
                confidence = 0.93
                return ClaimStatus.CONTRADICTED, confidence
        
        # Generic "no X" pattern
        no_match2 = re.search(r'\bno\s+(?:known\s+)?(\w+)', claim_text, re.I)
        if no_match2:
            claimed_absent = no_match2.group(1)
            # Check if the thing claimed absent is actually present in evidence
            if re.search(rf'\b{claimed_absent}\b', evidence_text, re.I) and 'no ' not in evidence_text:
                # Specific entities that indicate contradiction
                if claimed_absent.lower() in ['allerg', 'medication', 'drug', 'condition']:
                    confidence = 0.90
                    return ClaimStatus.CONTRADICTED, confidence
        
        return None
    
    def _check_support(self, claim: Claim, evidence: Evidence) -> Optional[Tuple[ClaimStatus, float]]:
        """Check for strong support"""
        claim_text = claim.text.lower()
        evidence_text = evidence.text.lower()
        
        # Strong support criteria:
        # 1. All key values match
        # 2. High token overlap
        # 3. No contradictory signals
        
        # Check value matching
        value_matches = sum(1 for v in claim.key_values if v in evidence.text)
        value_total = len(claim.key_values)
        
        if value_total > 0 and value_matches == value_total:
            # All values match - strong support
            confidence = 0.90 + (evidence.relevance_score * 0.05)
            return ClaimStatus.SUPPORTED, min(confidence, 0.98)
        
        # Check entity matching
        entity_match_count = sum(1 for e in claim.key_entities if e.lower() in evidence_text)
        entity_total = len(claim.key_entities)
        
        if entity_total > 0 and entity_match_count >= entity_total * 0.8:
            # Most entities match
            confidence = 0.75 + (evidence.relevance_score * 0.1)
            return ClaimStatus.SUPPORTED, min(confidence, 0.95)
        
        # Check high token overlap
        if evidence.relevance_score > self.threshold_supported:
            confidence = evidence.relevance_score * 0.9
            return ClaimStatus.SUPPORTED, confidence
        
        return None


# ============================================================================
# EXPLAINABILITY ENGINE - ENHANCED
# ============================================================================

class ExplainabilityEngine:
    """
    Enhanced explainability with precise explanations and citations.
    """
    
    def generate_explanation(self, claim: Claim, evidence: Evidence, status: ClaimStatus) -> str:
        """Generate human-readable explanation"""
        
        if status == ClaimStatus.SUPPORTED:
            return self._explain_supported(claim, evidence)
        
        elif status == ClaimStatus.CONTRADICTED:
            return self._explain_contradicted(claim, evidence)
        
        else:  # UNVERIFIABLE
            return self._explain_unverifiable(claim, evidence)
    
    def _explain_supported(self, claim: Claim, evidence: Evidence) -> str:
        """Explain why claim is supported"""
        if claim.key_values:
            value = claim.key_values[0]
            return f"The value '{value}' is accurately cited from source document (line {evidence.line_number})."
        
        if claim.key_entities:
            entity = claim.key_entities[0]
            return f"Statement about '{entity}' is confirmed by source document."
        
        return f"This claim is supported by evidence in the source document."
    
    def _explain_contradicted(self, claim: Claim, evidence: Evidence) -> str:
        """Explain why claim is contradicted"""
        claim_text = claim.text
        evidence_text = evidence.text
        
        # Check for Type mismatch
        claim_type = re.search(r'Type\s*(\d+)', claim_text, re.I)
        evidence_type = re.search(r'Type\s*(\d+)', evidence_text, re.I)
        
        if claim_type and evidence_type and claim_type.group(1) != evidence_type.group(1):
            return (
                f"The claim states 'Type {claim_type.group(1)}' but the source document "
                f"clearly indicates 'Type {evidence_type.group(1)}' (line {evidence.line_number})."
            )
        
        # Check for numerical mismatch
        claim_nums = re.findall(r'\b\d+\.?\d*\b', claim_text)
        evidence_nums = re.findall(r'\b\d+\.?\d*\b', evidence_text)
        
        if claim_nums and evidence_nums:
            for c_num, e_num in zip(claim_nums, evidence_nums):
                if c_num != e_num:
                    return (
                        f"The claim states '{c_num}' but the source document "
                        f"specifies '{e_num}' (line {evidence.line_number})."
                    )
        
        # Check for status contradiction
        if 'normal' in claim_text.lower() and re.search(r'\b(elevated|high|abnormal)\b', evidence_text, re.I):
            status_word = re.search(r'\b(elevated|high|abnormal)\b', evidence_text, re.I).group(1)
            return (
                f"The claim describes this as 'normal' but the source indicates "
                f"it is '{status_word}' (line {evidence.line_number})."
            )
        
        # Generic contradiction
        return (
            f"This statement contradicts information in the source document: "
            f"\"{evidence_text}\" (line {evidence.line_number})."
        )
    
    def _explain_unverifiable(self, claim: Claim, evidence: Evidence) -> str:
        """Explain why claim is unverifiable"""
        if not evidence or evidence.relevance_score < 0.1:
            return "No relevant information found in source documents to verify this claim."
        
        return (
            "Insufficient overlap between claim and source document to confidently "
            "verify or contradict this statement."
        )
    
    def generate_correction(self, claim: Claim, evidence: Evidence, status: ClaimStatus) -> Optional[str]:
        """Generate correction suggestion for contradicted claims"""
        if status != ClaimStatus.CONTRADICTED:
            return None
        
        claim_text = claim.text
        evidence_text = evidence.text
        corrected = claim_text
        
        # Fix Type mismatches
        claim_type = re.search(r'Type\s*(\d+)', claim_text, re.I)
        evidence_type = re.search(r'Type\s*(\d+)', evidence_text, re.I)
        
        if claim_type and evidence_type:
            corrected = re.sub(
                rf'Type\s*{claim_type.group(1)}',
                f'Type {evidence_type.group(1)}',
                corrected,
                flags=re.I
            )
        
        # Fix numerical mismatches
        claim_nums = re.findall(r'\b\d+\.?\d*\b', claim_text)
        evidence_nums = re.findall(r'\b\d+\.?\d*\b', evidence_text)
        
        for c_num, e_num in zip(claim_nums, evidence_nums):
            if c_num != e_num:
                corrected = corrected.replace(c_num, e_num, 1)
        
        # Fix status words
        if 'normal' in claim_text.lower():
            status_match = re.search(r'\b(elevated|high|abnormal)\b', evidence_text, re.I)
            if status_match:
                corrected = re.sub(r'\bnormal\b', status_match.group(1), corrected, flags=re.I)
        
        # Only return if actually changed
        return corrected if corrected != claim_text else None
    
    def generate_citation(self, evidence: Evidence) -> str:
        """Generate precise citation"""
        citation = f"Source document, line {evidence.line_number}: \"{evidence.text}\""
        
        if evidence.context_before:
            citation = f"[Context: {evidence.context_before}] " + citation
        
        return citation


# ============================================================================
# MAIN ORCHESTRATOR - ENHANCED
# ============================================================================

class HallucinationHunter:
    """
    Main orchestrator with enhanced precision.
    """
    
    def __init__(self, sources: List[str]):
        """Initialize with source documents"""
        self.extractor = ClaimExtractor()
        self.retriever = RAGRetriever(sources)
        self.verifier = NLIVerifier()
        self.explainer = ExplainabilityEngine()
    
    def verify_document(self, text: str) -> Dict:
        """
        Main verification pipeline.
        
        Returns comprehensive results with high precision.
        """
        # Step 1: Extract atomic claims
        claims = self.extractor.extract(text)
        
        if not claims:
            return {
                'results': [],
                'trust_score': 0.0,
                'summary': {
                    'total': 0,
                    'supported': 0,
                    'contradicted': 0,
                    'unverifiable': 0
                }
            }
        
        # Step 2: Verify each claim
        results = []
        
        for claim in claims:
            # Retrieve evidence
            evidence_list = self.retriever.retrieve(claim, k=3)
            
            if not evidence_list:
                # No evidence found
                results.append(VerificationResult(
                    claim=claim,
                    status=ClaimStatus.UNVERIFIABLE,
                    confidence=0.0,
                    evidence=[],
                    explanation="No relevant evidence found in source documents.",
                    correction=None,
                    citation=None
                ))
                continue
            
            # Find best verification across all evidence
            # Prioritize contradictions for safety
            contradiction_found = None
            best_support = None
            best_unverifiable = None
            
            for evidence in evidence_list:
                status, confidence = self.verifier.verify(claim, evidence)
                
                if status == ClaimStatus.CONTRADICTED:
                    if contradiction_found is None or confidence > contradiction_found[1]:
                        contradiction_found = (status, confidence, evidence)
                
                elif status == ClaimStatus.SUPPORTED:
                    if best_support is None or confidence > best_support[1]:
                        best_support = (status, confidence, evidence)
                
                else:  # UNVERIFIABLE
                    if best_unverifiable is None or confidence > best_unverifiable[1]:
                        best_unverifiable = (status, confidence, evidence)
            
            # Select in priority order: contradiction > support > unverifiable
            # ALWAYS prioritize contradictions for safety (avoid false negatives)
            if contradiction_found:
                # Any contradiction beats support or unverifiable
                best_status, best_confidence, best_evidence = contradiction_found
            elif best_support:
                best_status, best_confidence, best_evidence = best_support
            elif best_unverifiable:
                best_status, best_confidence, best_evidence = best_unverifiable
            else:
                # Fallback
                best_status = ClaimStatus.UNVERIFIABLE
                best_confidence = 0.0
                best_evidence = evidence_list[0]
            
            # Generate explanation and correction
            explanation = self.explainer.generate_explanation(claim, best_evidence, best_status)
            correction = self.explainer.generate_correction(claim, best_evidence, best_status)
            citation = self.explainer.generate_citation(best_evidence) if best_status == ClaimStatus.SUPPORTED else None
            
            results.append(VerificationResult(
                claim=claim,
                status=best_status,
                confidence=best_confidence,
                evidence=evidence_list,
                explanation=explanation,
                correction=correction,
                citation=citation
            ))
        
        # Step 3: Calculate trust score
        trust_score = self._calculate_trust_score(results)
        
        # Step 4: Generate summary
        summary = self._generate_summary(results)
        
        return {
            'results': results,
            'trust_score': trust_score,
            'summary': summary
        }
    
    def _calculate_trust_score(self, results: List[VerificationResult]) -> float:
        """
        Calculate overall trust score (0-100).
        
        Formula: Heavily penalize contradictions, reward supported claims.
        """
        if not results:
            return 0.0
        
        score = 0.0
        
        for result in results:
            if result.status == ClaimStatus.SUPPORTED:
                # Add confidence
                score += result.confidence
            
            elif result.status == ClaimStatus.CONTRADICTED:
                # Heavily penalize (2x the confidence)
                score -= 2.0 * result.confidence
            
            else:  # UNVERIFIABLE
                # Small penalty for uncertainty
                score -= 0.3
        
        # Normalize to 0-100 scale
        # Range: [-2*n, n] where n = number of claims
        n = len(results)
        min_possible = -2 * n
        max_possible = n
        
        normalized = ((score - min_possible) / (max_possible - min_possible)) * 100
        
        return round(max(0, min(100, normalized)), 1)
    
    def _generate_summary(self, results: List[VerificationResult]) -> Dict:
        """Generate summary statistics"""
        return {
            'total': len(results),
            'supported': sum(1 for r in results if r.status == ClaimStatus.SUPPORTED),
            'contradicted': sum(1 for r in results if r.status == ClaimStatus.CONTRADICTED),
            'unverifiable': sum(1 for r in results if r.status == ClaimStatus.UNVERIFIABLE)
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Test case from problem statement
    source_document = """
    Patient: John Doe
    Medical Record: MRN-12345
    Diagnosis: Type 1 Diabetes Mellitus
    Diagnosed: December 10, 2023
    
    Current Medications:
    - Insulin Glargine 20 units daily at bedtime
    - Metformin 500mg twice daily
    
    Allergies: Penicillin (severe rash)
    
    Recent Labs:
    Blood Pressure: 128/82 mmHg (slightly elevated)
    HbA1c: 8.9%
    
    Follow-up: Recommended in 3 months
    """
    
    llm_generated = """
    Patient John Doe has Type 2 Diabetes and takes 25 units of insulin daily.
    He has no known drug allergies.
    His blood pressure is normal.
    Follow-up is scheduled in 6 months.
    """
    
    # Initialize hunter
    hunter = HallucinationHunter([source_document])
    
    # Verify document
    results = hunter.verify_document(llm_generated)
    
    # Display results
    print("=" * 70)
    print(f"TRUST SCORE: {results['trust_score']}%")
    print("=" * 70)
    print(f"\nSUMMARY:")
    print(f"  Total Claims: {results['summary']['total']}")
    print(f"  ✓ Supported: {results['summary']['supported']}")
    print(f"  ✗ Contradicted: {results['summary']['contradicted']}")
    print(f"  ? Unverifiable: {results['summary']['unverifiable']}")
    print("\n" + "=" * 70)
    
    for i, result in enumerate(results['results'], 1):
        print(f"\nCLAIM #{i}: {result.claim.text}")
        print(f"Status: {result.status.value.upper()}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Type: {result.claim.claim_type}")
        print(f"\nExplanation: {result.explanation}")
        
        if result.correction:
            print(f"\n✓ Suggested Correction: {result.correction}")
        
        if result.citation:
            print(f"\n📎 Citation: {result.citation}")
        
        print("-" * 70)