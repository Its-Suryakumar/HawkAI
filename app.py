import json
import faiss
import numpy as np
import requests
import streamlit as st
import time
import random
import os
from typing import List, Tuple, Optional, Dict, Callable
import re
from datetime import datetime
import hashlib
import io
from azure.storage.blob import BlobServiceClient, ContainerClient

# ===============================
# CONFIGURATION
# ===============================

AZURE_EMBED_ENDPOINT = os.environ.get("AZURE_EMBED_ENDPOINT")
AZURE_LLM_ENDPOINT = os.environ.get("AZURE_LLM_ENDPOINT")
API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.environ.get("BLOB_CONTAINER_NAME")


# Database folder mapping in blob storage
DATABASE_BLOB_PATHS = {
    "PRA_Rulebook": "PRA",
    "Basel3_1_Rulebook": "Basel",
    "HKMA_Rulebook": "HKMA"
}

# ===============================
# SCHEMA-SPECIFIC PARSERS
# ===============================

def parse_pra_chunk(chunk: dict) -> str:
    """Parser for PRA Rulebook format (extraction-based structure)"""
    extraction = chunk.get('extraction', {})
    text_parts = []
    
    # Article header
    article_num = extraction.get('article_number', 'N/A')
    article_title = extraction.get('article_title', '')
    if article_num != 'N/A' or article_title:
        text_parts.append(f"Article {article_num}: {article_title}")
    
    # Structural references
    part_ref = extraction.get('part_reference', '')
    section_ref = extraction.get('section_reference', '')
    if part_ref:
        text_parts.append(f"Part: {part_ref}")
    if section_ref:
        text_parts.append(f"Section: {section_ref}")
    
    # Summary
    summary = extraction.get('content_summary', '')
    if summary:
        text_parts.append(f"Summary: {summary}")
    
    # Paragraphs
    paragraphs = extraction.get('paragraphs', [])
    for para in paragraphs:
        if isinstance(para, dict):
            para_text = para.get('text', '').strip()
            if para_text:
                text_parts.append(para_text)
            
            # Sub-paragraphs
            sub_paras = para.get('sub_paragraphs', [])
            for sub_para in sub_paras:
                if isinstance(sub_para, dict):
                    sub_content = sub_para.get('content', '').strip()
                    if sub_content:
                        text_parts.append(f"• {sub_content}")
    
    return '\n'.join(filter(None, text_parts))

def extract_pra_metadata(chunk: dict) -> Dict:
    """Extract metadata for PRA Rulebook"""
    extraction = chunk.get('extraction', {})
    return {
        'article_number': extraction.get('article_number', 'N/A'),
        'article_title': extraction.get('article_title', ''),
        'part_reference': extraction.get('part_reference', ''),
        'section_reference': extraction.get('section_reference', ''),
        'page_number': chunk.get('page_number', 'N/A'),
        'content_length': len(parse_pra_chunk(chunk)),
        'document_type': 'PRA Rulebook'
    }

def parse_basel_chunk(chunk: dict) -> str:
    """Parser for Basel 3.1 Rulebook format (structured_data-based)"""
    structured_data = chunk.get('structured_data', {})
    text_parts = []
    
    # Chapter and topic header
    chapter_num = structured_data.get('chapter_number', 'N/A')
    chapter_title = structured_data.get('chapter_title', '')
    topic = structured_data.get('topic', '')
    
    if chapter_num != 'N/A' or chapter_title:
        text_parts.append(f"Chapter {chapter_num}: {chapter_title}")
    if topic:
        text_parts.append(f"Topic: {topic}")
    
    # Summary
    summary = structured_data.get('content_summary', '')
    if summary:
        text_parts.append(f"Summary: {summary}")
    
    # Paragraphs
    paragraphs = structured_data.get('paragraphs', [])
    for para in paragraphs:
        if isinstance(para, dict):
            para_num = para.get('paragraph_number', '')
            para_text = para.get('text', '').strip()
            
            if para_text:
                para_header = f"Paragraph {para_num}: " if para_num else ""
                text_parts.append(f"{para_header}{para_text}")
            
            # Bullet points
            bullet_points = para.get('bullet_points', [])
            for bullet in bullet_points:
                if bullet.strip():
                    text_parts.append(f"• {bullet}")
    
    # Cross references
    cross_refs = structured_data.get('cross_references', [])
    if cross_refs:
        text_parts.append("Cross References:")
        for ref in cross_refs:
            if isinstance(ref, dict):
                ref_to = ref.get('reference_to', '')
                context = ref.get('context', '')
                if ref_to:
                    text_parts.append(f"• {ref_to}: {context}")
    
    return '\n'.join(filter(None, text_parts))

def extract_basel_metadata(chunk: dict) -> Dict:
    """Extract metadata for Basel 3.1 Rulebook"""
    structured_data = chunk.get('structured_data', {})
    return {
        'chapter_number': structured_data.get('chapter_number', 'N/A'),
        'chapter_title': structured_data.get('chapter_title', ''),
        'topic': structured_data.get('topic', ''),
        'page_number': structured_data.get('page_number', chunk.get('source_page', 'N/A')),
        'content_length': len(parse_basel_chunk(chunk)),
        'document_type': 'Basel 3.1 Rulebook'
    }

def parse_hkma_chunk(chunk: dict) -> str:
    """Parser for HKMA Rulebook format (structured_data with sections)"""
    structured_data = chunk.get('structured_data', {})
    text_parts = []
    
    # Section header
    section_num = structured_data.get('section_number', 'N/A')
    section_title = structured_data.get('section_title', '')
    part_ref = structured_data.get('part_reference', '')
    division_ref = structured_data.get('division_reference', '')
    
    if section_num != 'N/A':
        header = f"Section {section_num}"
        if section_title:
            header += f": {section_title}"
        text_parts.append(header)
    
    if part_ref:
        text_parts.append(f"Part: {part_ref}")
    if division_ref:
        text_parts.append(f"Division: {division_ref}")
    
    # Summary
    summary = structured_data.get('content_summary', '')
    if summary:
        text_parts.append(f"Summary: {summary}")
    
    # Paragraphs
    paragraphs = structured_data.get('paragraphs', [])
    for para in paragraphs:
        if isinstance(para, dict):
            para_num = para.get('paragraph_number', '')
            para_text = para.get('text', '').strip()
            
            if para_text:
                para_header = f"({para_num}) " if para_num else ""
                text_parts.append(f"{para_header}{para_text}")
            
            # Sub-paragraphs with letters and numerals
            sub_paras = para.get('sub_paragraphs', [])
            for sub_para in sub_paras:
                if isinstance(sub_para, dict):
                    letter = sub_para.get('letter', '')
                    content = sub_para.get('content', '').strip()
                    
                    if content:
                        sub_header = f"({letter}) " if letter else "• "
                        text_parts.append(f"{sub_header}{content}")
                    
                    # Sub-items with numerals
                    sub_items = sub_para.get('sub_items', [])
                    for item in sub_items:
                        if isinstance(item, dict):
                            numeral = item.get('numeral', '')
                            item_content = item.get('content', '').strip()
                            if item_content:
                                item_header = f"({numeral}) " if numeral else "  • "
                                text_parts.append(f"{item_header}{item_content}")
    
    # Cross references
    cross_refs = structured_data.get('cross_references', [])
    if cross_refs:
        text_parts.append("Cross References:")
        for ref in cross_refs:
            if isinstance(ref, dict):
                ref_to = ref.get('reference_to', '')
                context = ref.get('context', '')
                if ref_to:
                    text_parts.append(f"• {ref_to}: {context}")
    
    return '\n'.join(filter(None, text_parts))

def extract_hkma_metadata(chunk: dict) -> Dict:
    """Extract metadata for HKMA Rulebook"""
    structured_data = chunk.get('structured_data', {})
    return {
        'section_number': structured_data.get('section_number', 'N/A'),
        'section_title': structured_data.get('section_title', ''),
        'part_reference': structured_data.get('part_reference', ''),
        'division_reference': structured_data.get('division_reference', ''),
        'page_number': structured_data.get('page_number', chunk.get('source_page', 'N/A')),
        'content_length': len(parse_hkma_chunk(chunk)),
        'document_type': 'HKMA Rulebook'
    }

# ===============================
# DATABASE CONFIGURATION
# ===============================
DATABASES = {
    "PRA_Rulebook": {
        "file": "pra_rulebook.jsonl",
        "parser": parse_pra_chunk,
        "metadata_extractor": extract_pra_metadata,
        "citation_format": lambda meta: f"Article {meta.get('article_number', 'N/A')}, Page {meta.get('page_number', 'N/A')}"
    },
    "Basel3_1_Rulebook": {
        "file": "basel3.1_rulebook.jsonl",
        "parser": parse_basel_chunk,
        "metadata_extractor": extract_basel_metadata,
        "citation_format": lambda meta: f"Chapter {meta.get('chapter_number', 'N/A')}, Page {meta.get('page_number', 'N/A')}"
    },
    "HKMA_Rulebook": {
        "file": "hkma_rulebook.jsonl",
        "parser": parse_hkma_chunk,
        "metadata_extractor": extract_hkma_metadata,
        "citation_format": lambda meta: f"Section {meta.get('section_number', 'N/A')}, Page {meta.get('page_number', 'N/A')}"
    }
}

# Enhanced configuration
EMBED_BATCH_SIZE = 16
MAX_RETRIES = 5
BASE_DELAY = 2.0
BACKOFF_FACTOR = 2.0
JITTER_MAX = 1.0

# Dynamic TOP_K based on query complexity
MIN_TOP_K = 3
MAX_TOP_K = 15
DEFAULT_TOP_K = 7

# Similarity thresholds for cosine similarity (0-1 scale)
MIN_SIMILARITY_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.85

# ===============================
# AZURE BLOB STORAGE FUNCTIONS
# ===============================

def get_blob_service_client():
    """Initialize Azure Blob Service Client"""
    if not AZURE_STORAGE_CONNECTION_STRING:
        st.error("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        return None
    try:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    except Exception as e:
        st.error(f"Failed to connect to Azure Blob Storage: {e}")
        return None

def download_blob_to_bytes(container_client: ContainerClient, blob_path: str) -> Optional[bytes]:
    """Download blob content as bytes"""
    try:
        blob_client = container_client.get_blob_client(blob_path)
        return blob_client.download_blob().readall()
    except Exception as e:
        st.error(f"Failed to download blob {blob_path}: {e}")
        return None

def load_index_data_from_blob(db_name: str) -> Tuple[Optional[faiss.Index], Optional[List[dict]], Optional[List[Dict]]]:
    """Load pre-built FAISS index, chunks, and metadata from Azure Blob Storage"""
    
    # Get blob service client
    blob_service_client = get_blob_service_client()
    if not blob_service_client:
        return None, None, None
    
    # Get container client
    try:
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    except Exception as e:
        st.error(f"Failed to access container '{BLOB_CONTAINER_NAME}': {e}")
        return None, None, None
    
    # Get folder path for this database
    folder_path = DATABASE_BLOB_PATHS.get(db_name)
    if not folder_path:
        st.error(f"No blob path configured for {db_name}")
        return None, None, None
    
    # Define blob paths
    index_blob = f"{folder_path}/faiss.index"
    chunks_blob = f"{folder_path}/chunks.json"
    metadata_blob = f"{folder_path}/metadata.json"
    
    try:
        # Download FAISS index
        index_bytes = download_blob_to_bytes(container_client, index_blob)
        if not index_bytes:
            st.error(f"Failed to download index from {index_blob}")
            return None, None, None
        
        # Load FAISS index from bytes
        index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
        
        # Download chunks
        chunks_bytes = download_blob_to_bytes(container_client, chunks_blob)
        if not chunks_bytes:
            st.error(f"Failed to download chunks from {chunks_blob}")
            return None, None, None
        
        chunks = json.loads(chunks_bytes.decode('utf-8'))
        
        # Download metadata
        metadata_bytes = download_blob_to_bytes(container_client, metadata_blob)
        if not metadata_bytes:
            st.error(f"Failed to download metadata from {metadata_blob}")
            return None, None, None
        
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        st.success(f"Successfully loaded {db_name} from Azure Blob Storage ({len(chunks)} sections)")
        return index, chunks, metadata
        
    except Exception as e:
        st.error(f"Error loading data for {db_name} from blob storage: {e}")
        return None, None, None

# ===============================
# UTILITY FUNCTIONS
# ===============================

def get_query_hash(query: str, db_names: List[str]) -> str:
    """Generate hash for query caching"""
    cache_key = f"{query}_{sorted(db_names)}"
    return hashlib.md5(cache_key.encode()).hexdigest()

def load_query_cache(query_hash: str) -> Optional[Dict]:
    """Load cached query results from session state"""
    if 'query_cache' not in st.session_state:
        st.session_state.query_cache = {}
    
    if query_hash in st.session_state.query_cache:
        cache_data = st.session_state.query_cache[query_hash]
        # Check if cache is less than 1 hour old
        if time.time() - cache_data.get('timestamp', 0) < 3600:
            return cache_data
        else:
            # Remove expired cache
            del st.session_state.query_cache[query_hash]
    
    return None

def save_query_cache(query_hash: str, results: Dict):
    """Save query results to session state cache"""
    if 'query_cache' not in st.session_state:
        st.session_state.query_cache = {}
    
    try:
        results['timestamp'] = time.time()
        st.session_state.query_cache[query_hash] = results
        
        # Limit cache size to prevent memory issues (keep last 50 queries)
        if len(st.session_state.query_cache) > 50:
            # Remove oldest entries
            sorted_cache = sorted(
                st.session_state.query_cache.items(),
                key=lambda x: x[1].get('timestamp', 0)
            )
            for old_key, _ in sorted_cache[:-50]:
                del st.session_state.query_cache[old_key]
    except Exception as e:
        st.warning(f"Failed to cache results: {e}")

def determine_top_k(query: str) -> int:
    """Dynamically determine TOP_K based on query complexity"""
    query_length = len(query.split())
    question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
    complex_indicators = ['compare', 'difference', 'versus', 'relationship', 'impact', 'effect']
    
    base_k = DEFAULT_TOP_K
    
    # Adjust based on query length
    if query_length > 15:
        base_k += 3
    elif query_length > 10:
        base_k += 1
    
    # Adjust based on complexity indicators
    query_lower = query.lower()
    if any(indicator in query_lower for indicator in complex_indicators):
        base_k += 4
    
    # Multiple question words suggest complex query
    question_count = sum(1 for word in question_words if word in query_lower)
    if question_count > 1:
        base_k += 2
    
    return min(max(base_k, MIN_TOP_K), MAX_TOP_K)

def preprocess_query(query: str) -> str:
    """Enhanced query preprocessing for legal documents"""
    # Expand common legal abbreviations
    abbreviations = {
        'cet1': 'common equity tier 1',
        't1': 'tier 1',
        't2': 'tier 2',
        'rwa': 'risk weighted assets',
        'lcr': 'liquidity coverage ratio',
        'nsfr': 'net stable funding ratio',
        'crr': 'capital requirements regulation',
        'crd': 'capital requirements directive',
        'pra': 'prudential regulation authority',
        'hkma': 'hong kong monetary authority',
        'basel': 'basel committee banking supervision'
    }
    
    processed_query = query.lower()
    for abbr, full_form in abbreviations.items():
        processed_query = processed_query.replace(abbr, full_form)
    
    return processed_query

def exponential_backoff_with_jitter(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter"""
    delay = BASE_DELAY * (BACKOFF_FACTOR ** attempt)
    jitter = random.uniform(0, JITTER_MAX)
    return min(delay + jitter, 60.0)

def safe_request(func, *args, **kwargs):
    """Wrapper for API requests with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            response = func(*args, **kwargs)
            if response.status_code == 429:
                delay = exponential_backoff_with_jitter(attempt)
                st.warning(f"Rate limited. Waiting {delay:.1f}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(delay)
                continue
            elif response.status_code >= 400:
                st.error(f"API Error: {response.status_code} - {response.text}")
                response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                st.error(f"Final request attempt failed: {e}")
                raise e
            delay = exponential_backoff_with_jitter(attempt)
            st.warning(f"Request failed: {e}. Retrying in {delay:.1f}s ({attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
    return None

# ===============================
# DATA PROCESSING
# ===============================

def chunk_to_text(chunk: dict, db_name: str) -> str:
    """Dynamic chunk-to-text conversion based on database type"""
    db_config = DATABASES[db_name]
    parser = db_config["parser"]
    return parser(chunk)

def extract_metadata(chunk: dict, db_name: str) -> Dict:
    """Extract searchable metadata from chunk based on database type"""
    db_config = DATABASES[db_name]
    metadata_extractor = db_config["metadata_extractor"]
    return metadata_extractor(chunk)

# ===============================
# EMBEDDING AND RETRIEVAL
# ===============================

def embed_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings from Azure with text-embedding-3-large"""
    if not texts or not API_KEY: 
        return []
    
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    try:
        response = safe_request(requests.post, AZURE_EMBED_ENDPOINT, 
                              headers=headers, json={"input": texts}, timeout=60)
        return [item['embedding'] for item in response.json()['data']] if response else None
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return None

def calculate_cosine_similarity(distance: float, is_inner_product: bool = False) -> float:
    """
    Convert distance to cosine similarity score (0-1)
    If using IndexFlatIP (inner product), distance is already similarity
    If using IndexFlatL2 with normalized vectors, need conversion
    """
    if is_inner_product:
        # Inner product with normalized vectors = cosine similarity
        return max(0, min(1, distance))
    else:
        # L2 distance with normalized vectors: similarity = 1 - (distance^2 / 2)
        return max(0, 1 - (distance ** 2 / 2))

def retrieve_chunks_with_scores(query: str, index: faiss.Index, chunks: List[dict], 
                               metadata: List[Dict], top_k: int, db_name: str) -> List[Tuple[dict, float, Dict]]:
    """Retrieve chunks with cosine similarity scores and metadata"""
    processed_query = preprocess_query(query)
    query_embedding = embed_batch([processed_query])
    
    if not query_embedding:
        return []
    
    # Normalize query embedding for cosine similarity
    query_emb_np = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_emb_np)
    
    # Search
    distances, indices = index.search(query_emb_np, min(top_k * 2, len(chunks)))
    
    # Determine if index is inner product based
    is_inner_product = isinstance(index, faiss.IndexFlatIP)
    
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunks):
            similarity = calculate_cosine_similarity(distance, is_inner_product)
            if similarity >= MIN_SIMILARITY_THRESHOLD:
                results.append((chunks[idx], similarity, metadata[idx]))
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def filter_and_rank_chunks(retrieved_chunks: List[Tuple[dict, float, Dict]], 
                          query: str, db_name: str) -> List[Tuple[dict, float, Dict]]:
    """Additional filtering and ranking based on content relevance"""
    query_terms = set(query.lower().split())
    
    scored_chunks = []
    for chunk, similarity, metadata in retrieved_chunks:
        content = chunk_to_text(chunk, db_name).lower()
        
        # Calculate term overlap score
        content_terms = set(content.split())
        term_overlap = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
        
        # Weighted combination
        final_score = similarity * 0.8 + term_overlap * 0.2
        
        # Boost chunks with specific identifiers mentioned in query
        if any(term.isdigit() for term in query_terms):
            for key in ['article_number', 'section_number', 'chapter_number']:
                identifier = metadata.get(key, 'N/A')
                if identifier != 'N/A' and str(identifier) in query:
                    final_score *= 1.15
                    break
        
        scored_chunks.append((chunk, final_score, metadata))
    
    # Re-sort by final score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks

# ===============================
# LLM GENERATION
# ===============================

def generate_enhanced_answer(query: str, retrieved_chunks: List[Tuple[dict, float, Dict]], 
                           db_name: str, is_multi_db: bool = False) -> str:
    """Generate answer with enhanced context and confidence indicators"""
    if not retrieved_chunks:
        return f"No relevant information was found in {db_name} to answer your question."
    if not API_KEY:
        return "API_KEY is not configured. Cannot generate an answer."

    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    db_config = DATABASES[db_name]
    citation_format = db_config["citation_format"]
    
    # Enhanced context with confidence scores
    context = ""
    high_confidence_sources = []
    
    for i, (chunk, score, metadata) in enumerate(retrieved_chunks):
        citation = citation_format(metadata)
        confidence = "High" if score >= HIGH_CONFIDENCE_THRESHOLD else "Medium" if score >= 0.65 else "Low"
        
        if score >= HIGH_CONFIDENCE_THRESHOLD:
            high_confidence_sources.append(citation)
        
        text = chunk_to_text(chunk, db_name)
        context += f"Source {i+1} ({citation}, Confidence: {confidence}):\n{text}\n\n"

    # Enhanced system message
    doc_type = metadata.get('document_type', db_name) if retrieved_chunks else db_name
    
    system_message = f"""You are an expert legal analyst specializing in financial regulations from {doc_type}. Your task is to provide detailed, accurate answers based ONLY on the provided regulatory context.

Guidelines:
1. **Mandatory Citations**: For every piece of information you use, you MUST cite the source using the format provided in the context.

2. **Confidence Indicators**: When using information from high-confidence sources {high_confidence_sources}, emphasize their reliability. For lower-confidence sources, acknowledge uncertainty.

3. **Context Boundaries**: If the answer is not in the provided context, state: 'The provided regulatory context from {doc_type} does not contain sufficient information on this topic.'

4. **Legal Precision**: Use precise legal language appropriate for {doc_type}. Distinguish between requirements, recommendations, and definitions.

5. **Comprehensive Structure**: For complex topics, organize your response with:
   - Direct answer first
   - Supporting regulatory provisions
   - Relevant exceptions or conditions
   - Cross-references to related sections

6. **Document-Specific Language**: 
   - For PRA: Use "Article" references and EU regulation terminology
   - For Basel: Use "Chapter" references and Basel framework terminology  
   - For HKMA: Use "Section" references and Hong Kong banking terminology

7. **Multi-Database Context**: {"This analysis is specific to " + doc_type + "." if is_multi_db else ""}
"""

    user_message = f"""Based ONLY on the following regulatory excerpts from {doc_type}, please answer the question with maximum precision and granularity.

Question: {query}

Regulatory Context from {doc_type}:
{context}

Please ensure your answer is comprehensive, well-structured, and includes all relevant regulatory details from the provided context.
"""

    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 1500,
        "temperature": 0.05,
        "top_p": 0.9
    }
    
    try:
        response = safe_request(requests.post, AZURE_LLM_ENDPOINT, headers=headers, 
                               json=payload, timeout=120)
        if response:
            answer = response.json()['choices'][0]['message']['content'].strip()
            # Add confidence summary
            confidence_summary = f"\n\n**Confidence Assessment**: This answer is based on {len(high_confidence_sources)} high-confidence sources and {len(retrieved_chunks) - len(high_confidence_sources)} additional sources from {doc_type}."
            return answer + confidence_summary
        return "API request failed after multiple retries."
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "An unexpected error occurred while generating the answer."

def generate_comparative_analysis(query: str, all_results: Dict[str, List[Tuple[dict, float, Dict]]]) -> str:
    """Generate comparative analysis across all databases"""
    if not API_KEY:
        return "API_KEY is not configured. Cannot generate comparative analysis."

    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    
    # Build comparative context
    comparative_context = ""
    
    for db_name, retrieved_chunks in all_results.items():
        if retrieved_chunks:
            db_config = DATABASES[db_name]
            citation_format = db_config["citation_format"]
            doc_type = retrieved_chunks[0][2].get('document_type', db_name) if retrieved_chunks else db_name
            
            comparative_context += f"\n=== {doc_type} ===\n"
            
            for i, (chunk, score, metadata) in enumerate(retrieved_chunks[:3]):
                citation = citation_format(metadata)
                text = chunk_to_text(chunk, db_name)
                comparative_context += f"Source {i+1} ({citation}):\n{text}\n\n"
    
    system_message = """You are an expert legal analyst specializing in comparative regulatory analysis across multiple jurisdictions. Your task is to provide a comprehensive comparative analysis of how different regulatory frameworks address the same topic.

Guidelines:
1. **Structured Comparison**: Organize your response by:
   - Common principles across all frameworks
   - Key differences and variations
   - Jurisdictional specificities
   - Practical implications

2. **Clear Attribution**: Always specify which regulatory framework each point comes from.

3. **Analytical Depth**: Don't just list differences - explain WHY they might exist and their practical implications.

4. **Balanced Coverage**: Give appropriate attention to each framework based on the available information.

5. **Regulatory Context**: Consider the different regulatory philosophies and approaches:
   - PRA: EU/UK regulatory approach
   - Basel: International standards framework
   - HKMA: Hong Kong specific implementation

6. **Practical Insights**: Highlight which framework might be most relevant for different scenarios.
"""

    user_message = f"""Based on the regulatory excerpts from multiple frameworks below, please provide a comprehensive comparative analysis addressing the following question:

Question: {query}

Regulatory Context from Multiple Frameworks:
{comparative_context}

Please structure your analysis to show:
1. How each framework approaches this topic
2. Key similarities and differences
3. Practical implications of these differences
4. Which framework provides the most comprehensive coverage for this specific query
"""

    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 2000,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    try:
        response = safe_request(requests.post, AZURE_LLM_ENDPOINT, headers=headers, 
                               json=payload, timeout=120)
        if response:
            return response.json()['choices'][0]['message']['content'].strip()
        return "API request failed after multiple retries."
    except Exception as e:
        st.error(f"Error generating comparative analysis: {e}")
        return "An unexpected error occurred while generating the comparative analysis."

# ===============================
# UI DISPLAY FUNCTIONS
# ===============================

def display_chunk_with_metadata(chunk: dict, score: float, metadata: Dict, key_prefix: str, db_name: str):
    """Display chunk with enhanced metadata and confidence indicators"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.text_area(
            f"Content (Similarity: {score:.3f})", 
            chunk_to_text(chunk, db_name), 
            height=200, 
            key=f"{key_prefix}_content"
        )
    
    with col2:
        st.subheader("Metadata")
        doc_type = metadata.get('document_type', 'Unknown')
        st.write(f"**Source**: {doc_type}")
        
        # Display relevant identifier based on document type
        if 'article_number' in metadata:
            st.write(f"**Article**: {metadata.get('article_number', 'N/A')}")
        elif 'section_number' in metadata:
            st.write(f"**Section**: {metadata.get('section_number', 'N/A')}")
        elif 'chapter_number' in metadata:
            st.write(f"**Chapter**: {metadata.get('chapter_number', 'N/A')}")
        
        st.write(f"**Page**: {metadata.get('page_number', 'N/A')}")
        st.write(f"**Length**: {metadata.get('content_length', 0)} chars")
        
        confidence = "🟢 High" if score >= HIGH_CONFIDENCE_THRESHOLD else "🟡 Medium" if score >= 0.65 else "🔴 Low"
        st.write(f"**Confidence**: {confidence}")

# ===============================
# LOAD INDICES WITH CACHING
# ===============================

@st.cache_resource(show_spinner=False)
def load_all_indices_cached():
    """Load and cache all indices on app startup - cached across all users"""
    indices = {}
    progress_placeholder = st.empty()
    
    for i, db_name in enumerate(DATABASES.keys()):
        progress_placeholder.info(f"Loading {db_name} from Azure Blob Storage... ({i+1}/{len(DATABASES)})")
        
        index, chunks, metadata = load_index_data_from_blob(db_name)
        if index and chunks and metadata:
            indices[db_name] = {
                'index': index,
                'chunks': chunks,
                'metadata': metadata,
                'status': '✅ Ready',
                'source': 'Azure Blob Storage'
            }
        else:
            indices[db_name] = {
                'status': '❌ Not Found',
                'source': 'Azure Blob Storage'
            }
    
    progress_placeholder.empty()
    return indices

# ===============================
# MAIN APPLICATION
# ===============================

def main():
    st.set_page_config(page_title="HawkAI", page_icon="🦅", layout="wide")
    st.title("HawkAI — Your Legal Intelligence, Elevated.")
    st.caption("Your gateway to multi-regulatory legal intelligence at HSBC.")

    if not API_KEY:
        st.error("🚨 AZURE_API_KEY environment variable is not set. The application cannot function.")
        return
    
    if not AZURE_STORAGE_CONNECTION_STRING:
        st.error("🚨 AZURE_STORAGE_CONNECTION_STRING environment variable is not set. Cannot load indices.")
        return

    # Initialize session state for indices (loads once and caches)
    if 'db_data' not in st.session_state:
        with st.spinner("🔄 Loading indices from Azure Blob Storage (first load may take a moment)..."):
            st.session_state.db_data = load_all_indices_cached()

    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Database selection with enhanced options
        db_options = list(DATABASES.keys()) + ["🔍 All Databases (Comparative Analysis)"]
        selected_db = st.selectbox("Choose database(s) to query:", db_options)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            custom_top_k = st.slider("Max results per database", MIN_TOP_K, MAX_TOP_K, DEFAULT_TOP_K)
            show_confidence = st.checkbox("Show confidence scores", True)
            enable_query_cache = st.checkbox("Enable query caching", True)
            show_comparative = st.checkbox("Show comparative analysis (All DBs)", True)
        
        st.markdown("---")
        st.info("💾 Using Azure Blob Storage with cosine similarity")

    # Enhanced database status display
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Database Status")
        for db_name, data in st.session_state.db_data.items():
            status = data.get('status', 'Unknown')
            st.markdown(f"**{db_name}:** {status}")
            if status == '✅ Ready':
                chunk_count = len(data['chunks'])
                avg_length = np.mean([m.get('content_length', 0) for m in data['metadata']])
                doc_type = data['metadata'][0].get('document_type', 'Unknown') if data['metadata'] else 'Unknown'
                st.caption(f"📄 {chunk_count} sections • Avg: {avg_length:.0f} chars")
                st.caption(f"🏛️ {doc_type}")

    # Enhanced main query interface
    st.markdown("### 🔍 Query Interface")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Ask your question about the legal documents:", 
            placeholder="e.g., What are the specific requirements for Common Equity Tier 1 capital calculation?",
            help="Tip: Include specific identifiers (Article numbers, Section numbers, Chapter numbers) for more precise results"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔎 Search", type="primary", width='stretch')

    if search_clicked and query:
        # Determine databases to query
        db_to_query = []
        is_multi_db = False
        
        if "All Databases" in selected_db:
            db_to_query = [name for name, data in st.session_state.db_data.items() 
                          if data.get('status') == '✅ Ready']
            is_multi_db = True
        elif st.session_state.db_data.get(selected_db, {}).get('status') == '✅ Ready':
            db_to_query = [selected_db]
        
        if not db_to_query:
            st.error("❌ Selected database(s) are not ready. Please check the status in the sidebar.")
            return
        
        # Check cache if enabled
        query_hash = get_query_hash(query, db_to_query) if enable_query_cache else None
        cached_results = load_query_cache(query_hash) if query_hash else None
        
        if cached_results:
            st.info("⚡ Loaded results from cache")
            results = cached_results['results']
        else:
            # Dynamic TOP_K determination
            dynamic_top_k = determine_top_k(query)
            if custom_top_k != DEFAULT_TOP_K:
                dynamic_top_k = custom_top_k
            
            st.info(f"🎯 Using {dynamic_top_k} results per database based on query complexity")
            
            results = {}
            
            # Process each database
            for db_name in db_to_query:
                with st.spinner(f"🔍 Searching in {db_name}..."):
                    db_info = st.session_state.db_data[db_name]
                    retrieved_with_scores = retrieve_chunks_with_scores(
                        query, db_info['index'], db_info['chunks'], 
                        db_info['metadata'], dynamic_top_k, db_name
                    )
                    
                    # Apply additional filtering and ranking
                    filtered_chunks = filter_and_rank_chunks(retrieved_with_scores, query, db_name)
                    results[db_name] = filtered_chunks
            
            # Cache results if enabled
            if query_hash and enable_query_cache:
                save_query_cache(query_hash, {'results': results, 'query': query})
        
        # Display results for each database
        for db_name, retrieved_chunks in results.items():
            if len(db_to_query) > 1:
                st.markdown(f"---")
                doc_type = retrieved_chunks[0][2].get('document_type', db_name) if retrieved_chunks else db_name
                st.markdown(f"### 📚 Results from {doc_type}")
            
            if retrieved_chunks:
                # Show retrieved context with confidence scores
                with st.expander(f"📋 Retrieved Context from {db_name} ({len(retrieved_chunks)} sources)", expanded=False):
                    for i, (chunk, score, metadata) in enumerate(retrieved_chunks):
                        with st.container():
                            st.markdown(f"#### Source {i+1}")
                            display_chunk_with_metadata(chunk, score, metadata, f"{db_name}_{i}", db_name)
                            st.markdown("---")
                
                # Generate enhanced answer
                with st.spinner(f"🤖 Generating comprehensive answer from {db_name}..."):
                    answer = generate_enhanced_answer(query, retrieved_chunks, db_name, is_multi_db)
                
                # Display answer with formatting
                st.markdown("#### 📋 Analysis & Answer")
                if show_confidence:
                    high_conf_count = sum(1 for _, score, _ in retrieved_chunks if score >= HIGH_CONFIDENCE_THRESHOLD)
                    total_sources = len(retrieved_chunks)
                    
                    confidence_color = "🟢" if high_conf_count >= 2 else "🟡" if high_conf_count >= 1 else "🔴"
                    st.markdown(f"{confidence_color} **Overall Confidence**: {high_conf_count}/{total_sources} high-confidence sources")
                
                st.markdown(answer)
                
                # Add source summary
                if len(retrieved_chunks) > 0:
                    st.markdown("##### 🔎 Quick Source Reference")
                    db_config = DATABASES[db_name]
                    citation_format = db_config["citation_format"]
                    
                    source_summary = []
                    for i, (_, score, metadata) in enumerate(retrieved_chunks[:3]):
                        citation = citation_format(metadata)
                        conf = "High" if score >= HIGH_CONFIDENCE_THRESHOLD else "Medium" if score >= 0.65 else "Low"
                        source_summary.append(f"• **Source {i+1}**: {citation} ({conf} confidence)")
                    
                    st.markdown('\n'.join(source_summary))
                
            else:
                st.warning(f"⚠️ No relevant sections found in {db_name} for your query.")
                st.info("💡 Try rephrasing your question or using different keywords.")
        
        # Enhanced comparative analysis for multi-database queries
        if is_multi_db and any(results.values()):
            st.markdown("---")
            st.markdown("### 🔍 Cross-Framework Comparative Analysis")
            
            # Create comparison metrics
            comparison_data = []
            for db_name, chunks in results.items():
                if chunks:
                    avg_confidence = np.mean([score for _, score, _ in chunks])
                    high_conf_count = sum(1 for _, score, _ in chunks if score >= HIGH_CONFIDENCE_THRESHOLD)
                    doc_type = chunks[0][2].get('document_type', db_name)
                    
                    comparison_data.append({
                        'Regulatory Framework': doc_type,
                        'Sources Found': len(chunks),
                        'High Confidence': high_conf_count,
                        'Avg Confidence': f"{avg_confidence:.3f}",
                        'Best Source': DATABASES[db_name]["citation_format"](chunks[0][2]) if chunks else "N/A"
                    })
            
            if comparison_data:
                st.dataframe(comparison_data, width='stretch')
                
                # Generate comprehensive comparative analysis
                if show_comparative:
                    with st.spinner("🔍 Generating cross-framework comparative analysis..."):
                        comparative_analysis = generate_comparative_analysis(query, results)
                    
                    st.markdown("#### 🏛️ Regulatory Framework Comparison")
                    st.markdown(comparative_analysis)
                
                # Recommendations
                best_db = max(comparison_data, key=lambda x: (x['High Confidence'], x['Sources Found']))
                st.success(f"🎯 **Primary Recommendation**: {best_db['Regulatory Framework']} provides the most comprehensive coverage for this query")

    elif search_clicked and not query:
        st.warning("⚠️ Please enter a question to search.")

    # Enhanced footer with tips
    st.markdown("---")
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        **Query Optimization:**
        - Include specific identifiers when known:
          - PRA: "Article 25", "Part Three"
          - Basel: "Chapter 4", "Section 2.1" 
          - HKMA: "Section 59", "Division 3"
        - Use framework-specific terminology
        - Ask focused questions rather than broad topics
        
        **Understanding Results:**
        - 🟢 High confidence: Strong semantic match (cosine similarity ≥ 0.85)
        - 🟡 Medium confidence: Relevant but verify (similarity 0.65-0.85)
        - 🔴 Low confidence: Weakly related (similarity 0.5-0.65)
        
        **Multi-Database Analysis:**
        - Use "All Databases" to compare regulatory approaches
        - Review comparative analysis for comprehensive understanding
        - Each framework may have different perspectives on similar topics
        - Consider jurisdictional differences in implementation
        
        **Framework Differences:**
        - **PRA**: EU/UK specific implementation with detailed articles
        - **Basel**: International standards and principles
        - **HKMA**: Hong Kong specific rules and sections
        """)
    
    # System statistics
    if st.checkbox("📊 Show System Statistics"):
        ready_dbs = sum(1 for data in st.session_state.db_data.values() if data.get('status') == '✅ Ready')
        total_chunks = sum(len(data.get('chunks', [])) for data in st.session_state.db_data.values() 
                          if data.get('status') == '✅ Ready')
        
        cache_count = len(st.session_state.get('query_cache', {}))
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ready Databases", ready_dbs)
        col2.metric("Total Sections", total_chunks)
        col3.metric("Cache Entries", cache_count)
        col4.metric("Similarity Method", "Cosine")
        
        # Schema breakdown
        st.markdown("##### Database Schema Types")
        for db_name, data in st.session_state.db_data.items():
            if data.get('status') == '✅ Ready' and data.get('metadata'):
                doc_type = data['metadata'][0].get('document_type', 'Unknown')
                chunk_count = len(data['chunks'])
                st.write(f"**{doc_type}**: {chunk_count} sections")

if __name__ == "__main__":
    main()