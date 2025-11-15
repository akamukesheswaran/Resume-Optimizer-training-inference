"""
Resume-Job Matching Utility Functions
Auto-generated from training notebook
Contains all preprocessing, extraction, and matching functions
"""

# ============================================================================
# IMPORTS
# ============================================================================
import re
import string
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process

# ============================================================================
# GLOBAL INITIALIZATION
# ============================================================================
# NOTE: These models must be loaded in the inference notebook BEFORE importing:
#   - tfidf_vectorizer_combined (from pickle)
#   - svm_combined (from pickle)
#   - label_encoder (from pickle)
#   - bert_model (from HuggingFace: bert-base-uncased)
#   - tokenizer (from HuggingFace: bert-base-uncased)
#   - ml_knowledge_classifier (from HuggingFace: jjzha/jobbert_knowledge_extraction)
#   - skill_similarity_model (from HuggingFace: sentence-transformers/all-MiniLM-L6-v2)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words_set = set(stopwords.words("english"))

# ============================================================================
# CONSTANTS & DICTIONARIES
# ============================================================================

contractions_dict = {
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "cannot",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "oughtn't": "ought not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "let's": "let us",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
}

tech_ngrams = {
    'agile methodology',
    'angular js',
    'back end',
    'big data',
    'block chain',
    'business analyst',
    'business intelligence',
    'ci cd',
    'cloud computing',
    'computer vision',
    'customer service',
    'cyber security',
    'data engineer',
    'data engineering',
    'data science',
    'data scientist',
    'decision tree',
    'decision trees',
    'deep learning',
    'dev ops',
    'distributed systems',
    'elastic search',
    'feature engineering',
    'front end',
    'full stack',
    'functional programming',
    'git hub',
    'human resources',
    'information security',
    'k means',
    'machine learning',
    'micro services',
    'natural language processing',
    'network security',
    'neural network',
    'neural networks',
    'node js',
    'object oriented',
    'operating system',
    'power bi',
    'product management',
    'project management',
    'quality assurance',
    'random forest',
    'react js',
    'react native',
    'reinforcement learning',
    'rest api',
    'software development',
    'software engineer',
    'software engineering',
    'sql server',
    'supply chain',
    'support vector',
    'test driven',
    'time series',
    'version control',
    'vue js',
    'web developer',
    'web development',
}

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def clean_resume_text(text):
    """
    Clean resume text for BERT, TF-IDF, and LinearSVM processing
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Fix common encoding issues
    text = text.replace('â¢', '•')  # Fix bullet points
    text = text.replace('NaÃ¯ve', 'Naive')  # Fix encoding
    text = text.replace('â€™', "'")  # Fix apostrophes
    text = text.replace('â€œ', '"').replace('â€', '"')  # Fix quotes

    # Step 2: Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Step 3: Clean formatting
    text = text.replace('\\r\\n', ' ')  # Remove line breaks
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space

    # Step 4: Remove email addresses (optional - might want to keep for context)
    text = re.sub(r'\S+@\S+', '', text)

    # Step 5: Remove phone numbers
    text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)

    # Step 6: Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)

    # Step 7: Convert to lowercase
    text = text.lower()

    # Step 8: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def expand_contractions(text):
    """
    Expand contractions in text for better TF-IDF processing
    """
    text_lower = text.lower()

    for contraction, expanded in contractions_dict.items():
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(contraction) + r'\b'
        text_lower = re.sub(pattern, expanded, text_lower)

    return text_lower


def preserve_ngrams(text):
    """
    Replace multi-word technical terms with underscore-connected versions
    to preserve them as single tokens during TF-IDF
    """
    text_lower = text.lower()

    # Sort by length (longest first) to avoid partial replacements
    sorted_ngrams = sorted(tech_ngrams, key=len, reverse=True)

    for ngram in sorted_ngrams:
        # Replace the ngram with underscore version
        underscore_version = ngram.replace(' ', '_')
        text_lower = text_lower.replace(ngram, underscore_version)

    return text_lower


def preprocess_for_tfidf_enhanced(text):
    """
    Enhanced TF-IDF preprocessing with all steps from the presentation:
    1. Basic cleaning
    2. Contractions expansion
    3. N-gram preservation
    4. Stop word removal
    5. Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Basic cleaning (from clean_resume_text)
    text = clean_resume_text(text)

    # Step 2: Expand contractions BEFORE n-gram preservation
    text = expand_contractions(text)

    # Step 3: Preserve n-grams (technical terms)
    text = preserve_ngrams(text)

    # Step 4: Tokenize
    words = text.split()

    # Step 5: Remove stop words and short words, apply lemmatization
    processed_words = []
    for word in words:
        if word not in stop_words and len(word) > 2:
            # Only lemmatize if it's not an n-gram (doesn't contain underscore)
            if '_' not in word:
                lemmatized = lemmatizer.lemmatize(word)
                processed_words.append(lemmatized)
            else:
                # Keep n-grams as-is
                processed_words.append(word)

    return ' '.join(processed_words)


def preprocess_for_bert(text, max_tokens=512):
    """
    Lighter preprocessing for BERT - preserves context and structure
    BERT handles stopwords and complexity internally
    """
    if not isinstance(text, str):
        return ""

    # Apply basic cleaning only
    text = clean_resume_text(text)

    # BERT can handle more natural text, so we keep:
    # - More punctuation for context
    # - No aggressive stopword removal
    # - No lemmatization (BERT uses subword tokenization)

    # Truncate to BERT's token limit (roughly)
    # BERT tokenizes to subwords, so we estimate ~1.3 tokens per word
    words = text.split()
    max_words = int(max_tokens / 1.3)  # Conservative estimate

    if len(words) > max_words:
        text = ' '.join(words[:max_words])

    return text


def preprocess_for_svm(text):
    """
    SVM uses same preprocessing as TF-IDF since they work on same vector space
    """
    return preprocess_for_tfidf_simple(text)


def process_new_text(text, model_type='tfidf'):
    """
    Process a new resume or job description using appropriate preprocessing
    """
    if model_type == 'tfidf':
        return preprocess_for_tfidf_enhanced(text)
    elif model_type == 'bert':
        return preprocess_for_bert(text)
    else:
        raise ValueError("model_type must be 'tfidf' or 'bert'")



# ============================================================================
# RULE-BASED SKILL EXTRACTION
# ============================================================================

def extract_skills_and_tools(text):
    """
    Extract skills, tools, and technologies from text
    Uses both regex patterns and the custom tech vocabulary
    """
    text_lower = text.lower()
    found_skills = set()

    # Extract from custom vocabulary (already has tech terms)
    for term in tech_ngrams:
        if term.replace('_', ' ') in text_lower or term.replace('_', '') in text_lower:
            found_skills.add(term.replace('_', ' '))

    # Common programming languages
    prog_langs = ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go',
                  'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'php']
    for lang in prog_langs:
        if re.search(r'\b' + lang + r'\b', text_lower):
            found_skills.add(lang)

    # Frameworks and libraries
    frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring',
                  'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn']
    for framework in frameworks:
        # More flexible matching for frameworks (handles variations)
        if framework.replace('-', '') in text_lower.replace('-', '').replace(' ', ''):
            found_skills.add(framework)

    # Databases
    databases = ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
                'oracle', 'dynamodb', 'elasticsearch']
    for db in databases:
        if re.search(r'\b' + db + r'\b', text_lower):
            found_skills.add(db)

    # Cloud platforms
    clouds = ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services']
    for cloud in clouds:
        if cloud in text_lower:
            found_skills.add(cloud)

    # Tools
    tools = ['docker', 'kubernetes', 'git', 'jenkins', 'terraform', 'ansible',
            'jira', 'confluence', 'tableau', 'power bi']
    for tool in tools:
        if re.search(r'\b' + tool + r'\b', text_lower):
            found_skills.add(tool)

    return list(found_skills)



# ============================================================================
# ML-BASED SKILL EXTRACTION
# ============================================================================

def aggregate_span(results):
    """
    Official aggregation function from the demo
    Merges adjacent tokens into complete skill spans
    """
    if not results or len(results) == 0:
        return []

    new_results = []
    current_result = results[0].copy()

    for result in results[1:]:
        # Check if tokens are adjacent (merge multi-word skills)
        if result["start"] == current_result["end"] + 1:
            current_result["word"] += " " + result["word"]
            current_result["end"] = result["end"]
        else:
            new_results.append(current_result)
            current_result = result.copy()

    new_results.append(current_result)
    return new_results


def is_valid_skill(skill_text):
    """
    VERY STRICT quality filter for extracted skills
    """
    # Remove leading/trailing punctuation
    skill_text = skill_text.strip('.,;:!?\'"()-')

    # Must be at least 2 characters
    if len(skill_text) < 2:
        return False

    # REJECT if starts or ends with dash/hyphen (common in fragments)
    if skill_text.startswith('-') or skill_text.endswith('-'):
        return False

    # REJECT if contains ' - ' (sentence fragments)
    if ' - ' in skill_text:
        return False

    # Skip if mostly punctuation or whitespace
    alpha_chars = sum(c.isalnum() or c.isspace() for c in skill_text)
    if alpha_chars < 2:
        return False

    # Get words
    words = skill_text.lower().split()
    if not words:
        return False

    # REJECT single stop words or common non-skills
    non_skills = stop_words_set | {
        'adult', 'based', 'industry', 'current', 'led', 'within',
        'including', 'following', 'using', 'working', 'related',
        'various', 'multiple', 'several', 'company', 'client',
        'exprience'  # Common typo
    }

    if len(words) == 1 and words[0] in non_skills:
        return False

    # REJECT if starts or ends with stop word
    if words[0] in stop_words_set or words[-1] in stop_words_set:
        return False

    # Reject very long phrases (likely full sentences)
    if len(words) > 4:  # Reduced from 5 to 4
        return False

    # Reject if starts with punctuation
    if skill_text and skill_text[0] in '.,;:!?\'"()-':
        return False

    # Reject if too much punctuation overall
    punct_count = sum(c in '.,;:!?\'"()-' for c in skill_text)
    if punct_count > len(skill_text) * 0.2:  # More than 20% punctuation
        return False

    # REJECT obvious location names (capitalized single words that aren't skills)
    if len(words) == 1 and skill_text[0].isupper():
        # Allow if it's a known tech term
        tech_terms = {'python', 'java', 'aws', 'sql', 'react', 'docker',
                     'kubernetes', 'linux', 'git', 'azure', 'oracle', 'mysql'}
        if skill_text.lower() not in tech_terms:
            return False

    return True


def extract_skills_from_chunk(chunk, confidence_threshold=0.5):
    """
    Extract TECHNICAL skills from a single chunk of text
    ONLY uses Knowledge model (Python, AWS, Docker, etc.)
    """
    if not isinstance(chunk, str) or len(chunk) == 0:
        return []

    try:
        # Extract ONLY Knowledge (technical skills) - NO soft skills
        output_knowledge = ml_knowledge_classifier(chunk)

        # Aggregate multi-token spans
        if len(output_knowledge) > 0:
            output_knowledge = aggregate_span(output_knowledge)

        # Filter by confidence and quality
        chunk_skills = []

        for result in output_knowledge:
            if result.get('score', 0) >= confidence_threshold:
                skill_text = result['word'].strip().lower()

                # QUALITY FILTERS
                if is_valid_skill(skill_text):
                    chunk_skills.append(skill_text)

        return chunk_skills

    except Exception as e:
        print(f"⚠️ Error processing chunk: {str(e)[:100]}")
        return []


def extract_skills_with_ml(text, confidence_threshold=0.6):
    """
    ML-based TECHNICAL skill extraction using SLIDING WINDOW approach
    Processes ENTIRE text by breaking into overlapping chunks

    ONLY extracts hard/technical skills (Python, AWS, Docker, etc.)
    Does NOT extract soft skills (communication, leadership, etc.)

    Args:
        text: Full resume/job text (any length)
        confidence_threshold: Minimum confidence score (0.5 = 50%)

    Returns:
        List of all unique technical skills found across entire document
    """
    if not isinstance(text, str) or len(text) == 0:
        return []

    # Configuration
    MAX_CHARS = 2000  # ~500 tokens per chunk
    OVERLAP_CHARS = 500  # Overlap to catch skills at boundaries

    all_skills = set()  # Use set to auto-deduplicate

    # If text is short enough, process in one pass
    if len(text) <= MAX_CHARS:
        return extract_skills_from_chunk(text, confidence_threshold)

    # Otherwise, use sliding window
    position = 0
    chunk_num = 0

    while position < len(text):
        chunk_num += 1

        # Extract chunk
        end_position = min(position + MAX_CHARS, len(text))
        chunk = text[position:end_position]

        # If not at end, try to break at sentence boundary
        if end_position < len(text):
            # Look for last sentence ending in the last 200 chars of chunk
            sentence_endings = ['.', '!', '?', '\n']
            best_break = -1

            for i in range(len(chunk) - 1, max(0, len(chunk) - 200), -1):
                if chunk[i] in sentence_endings:
                    best_break = i + 1
                    break

            if best_break > 0:
                chunk = chunk[:best_break]
                end_position = position + best_break

        # Extract skills from this chunk
        chunk_skills = extract_skills_from_chunk(chunk, confidence_threshold)
        all_skills.update(chunk_skills)

        # Move window forward
        if end_position >= len(text):
            break  # We've reached the end

        # Move forward with overlap
        position = end_position - OVERLAP_CHARS

        # Safety check to prevent infinite loop
        if position <= chunk_num * 100:
            position = end_position

    return list(all_skills)



# ============================================================================
# BERT SIMILARITY
# ============================================================================

def get_bert_embedding(text, max_length=512):
    """
    Get BERT embedding for a text
    """
    # Tokenize and truncate
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        padding=True,
        truncation=True
    )

    # Get embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Use mean pooling of last hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.squeeze().numpy()


def calculate_bert_similarity(text1, text2):
    """
    Calculate cosine similarity using BERT embeddings
    """
    # Preprocess for BERT
    processed1 = process_new_text(text1, 'bert')
    processed2 = process_new_text(text2, 'bert')

    # Get embeddings
    embed1 = get_bert_embedding(processed1)
    embed2 = get_bert_embedding(processed2)

    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embed1], [embed2])[0][0]

    return similarity



# ============================================================================
# MATCHING FUNCTIONS
# ============================================================================

def fuzzy_match_skills(resume_skills, job_skills, threshold=75):
    """
    Compare two skill lists using fuzzy matching

    Args:
        resume_skills: List of skills from resume
        job_skills: List of skills from job description
        threshold: Minimum similarity score (0-100) to consider a match

    Returns:
        Dictionary with match statistics
    """
    if not resume_skills or not job_skills:
        return {
            'score': 0.0,
            'matched_skills': [],
            'missing_skills': job_skills,
            'match_rate': 0.0
        }

    matched_skills = []
    match_details = []

    for job_skill in job_skills:
        # Find best match in resume skills
        best_match = process.extractOne(
            job_skill,
            resume_skills,
            scorer=fuzz.ratio  # Can also try fuzz.token_set_ratio
        )

        if best_match and best_match[1] >= threshold:
            matched_skills.append({
                'job_requires': job_skill,
                'resume_has': best_match[0],
                'similarity': best_match[1]
            })
            match_details.append(best_match[0])

    # Calculate match rate
    match_rate = len(matched_skills) / len(job_skills) if job_skills else 0.0

    # Identify missing skills
    matched_job_skills = [m['job_requires'] for m in matched_skills]
    missing_skills = [s for s in job_skills if s not in matched_job_skills]

    # Calculate weighted score (considers similarity strength)
    if matched_skills:
        avg_similarity = sum(m['similarity'] for m in matched_skills) / len(matched_skills)
        score = (match_rate * 0.7) + (avg_similarity / 100 * 0.3)
    else:
        score = 0.0

    return {
        'score': score,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'match_rate': match_rate,
        'resume_skill_count': len(resume_skills),
        'job_skill_count': len(job_skills)
    }


def ml_semantic_skill_matching(resume_skills, job_skills, threshold=0.7):
    """
    ML-based semantic skill matching using neural embeddings
    Replaces string-based fuzzy matching with learned representations
    """
    if not resume_skills or not job_skills:
        return {
            'score': 0.0,
            'matched_skills': [],
            'missing_skills': job_skills,
            'match_rate': 0.0
        }

    # Compute embeddings (THIS IS THE ML PART!)
    resume_embeddings = skill_similarity_model.encode(resume_skills, convert_to_tensor=True)
    job_embeddings = skill_similarity_model.encode(job_skills, convert_to_tensor=True)

    matched_skills = []

    for i, job_skill in enumerate(job_skills):
        # Calculate semantic similarity with all resume skills
        similarities = util.cos_sim(job_embeddings[i], resume_embeddings)[0]

        # Find best match
        best_match_idx = similarities.argmax().item()
        best_similarity = similarities[best_match_idx].item()

        if best_similarity >= threshold:
            matched_skills.append({
                'job_requires': job_skill,
                'resume_has': resume_skills[best_match_idx],
                'similarity': best_similarity * 100
            })

    # Calculate metrics
    match_rate = len(matched_skills) / len(job_skills)
    matched_job_skills = [m['job_requires'] for m in matched_skills]
    missing_skills = [s for s in job_skills if s not in matched_job_skills]

    # Weighted score
    if matched_skills:
        avg_similarity = sum(m['similarity'] for m in matched_skills) / len(matched_skills)
        score = (match_rate * 0.7) + (avg_similarity / 100 * 0.3)
    else:
        score = 0.0

    return {
        'score': score,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'match_rate': match_rate,
        'resume_skill_count': len(resume_skills),
        'job_skill_count': len(job_skills)
    }



# ============================================================================
# MAIN MATCHING PIPELINE
# ============================================================================

def match_resume_to_job_with_ml(resume_text, job_description_text):
    """
    Complete matching pipeline with ML skill extraction:
    - BERT (40%): Semantic understanding
    - TF-IDF (25%): Keyword matching
    - Category (15%): Job category alignment
    - ML Skills (20%): Neural extraction + semantic matching
    """
    # Preprocess
    resume_tfidf = preprocess_for_tfidf_enhanced(resume_text)
    job_tfidf = preprocess_for_tfidf_enhanced(job_description_text)

    # 1. BERT Semantic Similarity (40%)
    bert_sim = calculate_bert_similarity(resume_text, job_description_text)

    # 2. TF-IDF Keyword Similarity (25%)
    resume_vec = tfidf_vectorizer_combined.transform([resume_tfidf])
    job_vec = tfidf_vectorizer_combined.transform([job_tfidf])
    tfidf_sim = cosine_similarity(resume_vec, job_vec)[0][0]

    # 3. Category Alignment (15%)
    resume_cat_idx = svm_combined.predict(resume_vec)[0]
    job_cat_idx = svm_combined.predict(job_vec)[0]

    resume_cat = label_encoder.inverse_transform([resume_cat_idx])[0]
    job_cat = label_encoder.inverse_transform([job_cat_idx])[0]

    category_match = 1.0 if resume_cat == job_cat else 0.3

    # 4. ML-Based Skills Matching (20%) - NOW USING WORKING ML!
    # Extract using ML neural models
    resume_skills_ml = extract_skills_with_ml(resume_text, confidence_threshold=0.5)
    job_skills_ml = extract_skills_with_ml(job_description_text, confidence_threshold=0.5)

    # Match using ML semantic similarity (lowered threshold for technical terms)
    ml_result = ml_semantic_skill_matching(resume_skills_ml, job_skills_ml, threshold=0.60)

    # Also get rule-based for comparison
    resume_skills_rule = extract_skills_and_tools(resume_text)
    job_skills_rule = extract_skills_and_tools(job_description_text)
    fuzzy_result = fuzzy_match_skills(resume_skills_rule, job_skills_rule, threshold=75)

    # Calculate final scores for BOTH approaches
    final_score_rule = (
        0.40 * bert_sim +
        0.25 * tfidf_sim +
        0.15 * category_match +
        0.20 * fuzzy_result['score']
    )

    final_score_ml = (
        0.40 * bert_sim +
        0.25 * tfidf_sim +
        0.15 * category_match +
        0.20 * ml_result['score']
    )

    # Recommendations
    def get_recommendation(score):
        if score >= 0.80:
            return 'Strong Match! 🎯'
        elif score >= 0.60:
            return 'Good Match ✅'
        else:
            return 'Needs Enhancement 📝'

    return {
        # Shared components
        'bert_similarity': bert_sim,
        'tfidf_similarity': tfidf_sim,
        'category_match': category_match,
        'resume_category': resume_cat,
        'job_category': job_cat,

        # APPROACH 1: Rule-based
        'approach_1_rule_based': {
            'final_score': final_score_rule,
            'skill_component_score': fuzzy_result['score'],
            'resume_skills': resume_skills_rule,
            'job_skills': job_skills_rule,
            'matched_skills': fuzzy_result['matched_skills'],
            'missing_skills': fuzzy_result['missing_skills'],
            'match_rate': fuzzy_result['match_rate'],
            'recommendation': get_recommendation(final_score_rule)
        },

        # APPROACH 2: ML-based (WORKING!)
        'approach_2_ml_based': {
            'final_score': final_score_ml,
            'skill_component_score': ml_result['score'],
            'resume_skills': resume_skills_ml,
            'job_skills': job_skills_ml,
            'matched_skills': ml_result['matched_skills'],
            'missing_skills': ml_result['missing_skills'],
            'match_rate': ml_result['match_rate'],
            'recommendation': get_recommendation(final_score_ml)
        }
    }



# ============================================================================
# DISPLAY HELPER
# ============================================================================

def format_text_readable(text, line_length=100):
    """Format text with line breaks every line_length characters"""
    if not isinstance(text, str):
        return ""

    paragraphs = text.split('\n')
    formatted_lines = []

    for paragraph in paragraphs:
        if len(paragraph) <= line_length:
            formatted_lines.append(paragraph)
        else:
            words = paragraph.split()
            current_line = ""

            for word in words:
                if len(current_line) + len(word) + 1 <= line_length:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
                else:
                    if current_line:
                        formatted_lines.append(current_line)
                    current_line = word

            if current_line:
                formatted_lines.append(current_line)

    return '\n'.join(formatted_lines)


