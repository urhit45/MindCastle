"""
Hashing Vectorizer for TinyNet - 512-dimensional text vectorization
"""

import hashlib
import re
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np


class HashingVectorizer512:
    """
    512-dimensional hashing vectorizer for short text + metadata.
    
    Features:
    - Word unigrams + bigrams
    - Character 3-grams + 4-grams  
    - Metadata: hour bucket, weekday, length bucket
    - Stable hashing with SHA1
    - L2 normalization
    - Optional TF-IDF scaling (placeholder)
    """
    
    def __init__(self, n_features: int = 512, use_tfidf: bool = False, seed: int = 13):
        """
        Initialize the vectorizer.
        
        Args:
            n_features: Number of features (default: 512)
            use_tfidf: Whether to use TF-IDF scaling (default: False)
            seed: Random seed for reproducibility (default: 13)
        """
        self.n_features = n_features
        self.use_tfidf = use_tfidf
        self.seed = seed
        np.random.seed(seed)
        
        # Metadata bin mappings
        self.hour_bins = [
            (0, 2), (3, 5), (6, 8), (9, 11),
            (12, 14), (15, 17), (18, 20), (21, 23)
        ]
        
        self.length_bins = [
            (0, 50),      # Short
            (51, 120),    # Medium  
            (121, float('inf'))  # Long
        ]
    
    def _hash_feature(self, feature: str) -> int:
        """
        Hash a feature string to a feature index.
        
        Args:
            feature: Feature string to hash
            
        Returns:
            Feature index in [0, n_features)
        """
        # Use SHA1 for stable hashing across runs
        hash_bytes = hashlib.sha1(feature.encode('utf-8')).digest()
        # Convert to integer and modulo to get index
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
        return hash_int % self.n_features
    
    def _extract_word_ngrams(self, text: str) -> List[str]:
        """
        Extract word unigrams and bigrams.
        
        Args:
            text: Input text
            
        Returns:
            List of word n-gram features
        """
        # Tokenize: lowercase, strip punctuation, split on whitespace
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        features = []
        
        # Unigrams
        features.extend(words)
        
        # Bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            features.append(bigram)
        
        return features
    
    def _extract_char_ngrams(self, text: str) -> List[str]:
        """
        Extract character 3-grams and 4-grams.
        
        Args:
            text: Input text (lowercase with spaces removed)
            
        Returns:
            List of character n-gram features
        """
        # Remove spaces and convert to lowercase
        clean_text = re.sub(r'\s+', '', text.lower())
        
        features = []
        
        # 3-grams
        for i in range(len(clean_text) - 2):
            trigram = clean_text[i:i+3]
            features.append(f"char3_{trigram}")
        
        # 4-grams  
        for i in range(len(clean_text) - 3):
            fourgram = clean_text[i:i+4]
            features.append(f"char4_{fourgram}")
        
        return features
    
    def _extract_metadata_features(self, timestamp: Optional[int] = None) -> List[str]:
        """
        Extract metadata features: hour bucket, weekday, length bucket.
        
        Args:
            timestamp: Unix timestamp (optional)
            
        Returns:
            List of metadata feature strings
        """
        features = []
        
        if timestamp is not None:
            dt = datetime.fromtimestamp(timestamp)
            
            # Hour bucket (0-23 -> 8 bins)
            hour = dt.hour
            for i, (start, end) in enumerate(self.hour_bins):
                if start <= hour <= end:
                    features.append(f"meta:hr:{i}")
                    break
            
            # Weekday (0-6)
            weekday = dt.weekday()
            features.append(f"meta:wd:{weekday}")
        
        return features
    
    def _get_length_bucket(self, text: str) -> str:
        """
        Get length bucket for text.
        
        Args:
            text: Input text
            
        Returns:
            Length bucket feature string
        """
        length = len(text)
        
        for i, (start, end) in enumerate(self.length_bins):
            if start <= length <= end:
                return f"meta:len:{i}"
        
        return f"meta:len:{len(self.length_bins) - 1}"  # Fallback to last bucket
    
    def encode(self, text: str, ts: Optional[int] = None) -> np.ndarray:
        """
        Encode text into a 512-dimensional vector.
        
        Args:
            text: Input text to vectorize
            ts: Unix timestamp for metadata (optional)
            
        Returns:
            Normalized feature vector of shape (n_features,)
        """
        # Initialize feature vector
        vector = np.zeros(self.n_features, dtype=np.float32)
        
        # Extract features
        word_features = self._extract_word_ngrams(text)
        char_features = self._extract_char_ngrams(text)
        metadata_features = self._extract_metadata_features(ts)
        length_feature = [self._get_length_bucket(text)]
        
        # Combine all features
        all_features = word_features + char_features + metadata_features + length_feature
        
        # Hash features and accumulate TF values
        for feature in all_features:
            if feature:  # Skip empty features
                index = self._hash_feature(feature)
                vector[index] += 1.0
        
        # L2 normalize (avoid division by zero)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def fit_transform(self, texts: List[str], timestamps: Optional[List[int]] = None) -> np.ndarray:
        """
        Fit the vectorizer and transform texts.
        
        Args:
            texts: List of input texts
            timestamps: List of Unix timestamps (optional)
            
        Returns:
            Feature matrix of shape (n_texts, n_features)
        """
        if timestamps is None:
            timestamps = [None] * len(texts)
        
        vectors = []
        for text, ts in zip(texts, timestamps):
            vector = self.encode(text, ts)
            vectors.append(vector)
        
        return np.array(vectors)
    
    def transform(self, texts: List[str], timestamps: Optional[List[int]] = None) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: List of input texts
            timestamps: List of Unix timestamps (optional)
            
        Returns:
            Feature matrix of shape (n_texts, n_features)
        """
        return self.fit_transform(texts, timestamps)
