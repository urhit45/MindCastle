"""
Tests for HashingVectorizer512
"""

import numpy as np
import pytest
from app.ml.vectorizer import HashingVectorizer512


class TestHashingVectorizer512:
    """Test suite for HashingVectorizer512"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.vectorizer = HashingVectorizer512(n_features=512, seed=13)
        self.test_text = "Ran 2 miles, shin tight"
        self.test_ts = 1724025600  # Unix timestamp
    
    def test_determinism(self):
        """Test that same text produces identical vector across calls"""
        vector1 = self.vectorizer.encode(self.test_text, self.test_ts)
        vector2 = self.vectorizer.encode(self.test_text, self.test_ts)
        
        np.testing.assert_array_equal(vector1, vector2)
        assert vector1.shape == (512,)
    
    def test_shape(self):
        """Test that output vector has correct shape"""
        vector = self.vectorizer.encode(self.test_text)
        assert vector.shape == (512,)
        assert vector.dtype == np.float32
    
    def test_nonzero_entries(self):
        """Test that vector has reasonable number of nonzero entries"""
        vector = self.vectorizer.encode(self.test_text)
        nonzero_count = np.count_nonzero(vector)
        
        # Should have more than 5 nonzero entries for a reasonable sentence
        assert nonzero_count > 5, f"Expected >5 nonzero entries, got {nonzero_count}"
        
        # Should not have too many (avoid overfitting)
        assert nonzero_count < 100, f"Expected <100 nonzero entries, got {nonzero_count}"
    
    def test_stability(self):
        """Test that small text edits change the vector significantly"""
        text1 = "Ran 2 miles, shin tight"
        text2 = "Ran 2 miles, shin hurts"  # Small edit
        
        vector1 = self.vectorizer.encode(text1)
        vector2 = self.vectorizer.encode(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        
        # Small edit should change vector (cosine similarity < 0.98)
        assert similarity < 0.98, f"Expected similarity < 0.98, got {similarity:.3f}"
        
        # But should still be somewhat similar (> 0.5)
        assert similarity > 0.5, f"Expected similarity > 0.5, got {similarity:.3f}"
    
    def test_metadata_influence(self):
        """Test that different weekday changes at least 1 index value"""
        # Same text, different weekdays
        ts1 = 1724025600  # Monday
        ts2 = 1724112000  # Tuesday (next day)
        
        vector1 = self.vectorizer.encode(self.test_text, ts1)
        vector2 = self.vectorizer.encode(self.test_text, ts2)
        
        # Should have different values due to metadata
        assert not np.array_equal(vector1, vector2), "Vectors should differ with different metadata"
        
        # Check that at least one index has different values
        diff_indices = np.where(vector1 != vector2)[0]
        assert len(diff_indices) > 0, "No differences found with different metadata"
    
    def test_empty_text(self):
        """Test handling of empty text"""
        vector = self.vectorizer.encode("")
        assert vector.shape == (512,)
        assert np.allclose(vector, np.zeros(512))
    
    def test_single_word(self):
        """Test single word input"""
        vector = self.vectorizer.encode("hello")
        assert vector.shape == (512,)
        assert np.linalg.norm(vector) > 0  # Should have some features
    
    def test_special_characters(self):
        """Test text with special characters and punctuation"""
        text = "Hello, world! How are you? 123 @#$%"
        vector = self.vectorizer.encode(text)
        assert vector.shape == (512,)
        assert np.linalg.norm(vector) > 0
    
    def test_batch_processing(self):
        """Test batch processing with fit_transform"""
        texts = ["Hello world", "Goodbye world", "Test message"]
        vectors = self.vectorizer.fit_transform(texts)
        
        assert vectors.shape == (3, 512)
        assert vectors.dtype == np.float32
        
        # Each vector should be normalized
        for vector in vectors:
            norm = np.linalg.norm(vector)
            if norm > 0:
                assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_timestamp_handling(self):
        """Test timestamp handling and metadata extraction"""
        # With timestamp
        vector_with_ts = self.vectorizer.encode(self.test_text, self.test_ts)
        
        # Without timestamp
        vector_without_ts = self.vectorizer.encode(self.test_text)
        
        # Should be different due to metadata
        assert not np.array_equal(vector_with_ts, vector_without_ts)
    
    def test_feature_hashing_stability(self):
        """Test that feature hashing is stable across runs"""
        # Create new vectorizer with same seed
        vectorizer2 = HashingVectorizer512(n_features=512, seed=13)
        
        vector1 = self.vectorizer.encode(self.test_text, self.test_ts)
        vector2 = vectorizer2.encode(self.test_text, self.test_ts)
        
        np.testing.assert_array_equal(vector1, vector2)
    
    def test_different_seeds(self):
        """Test that different seeds produce different results"""
        vectorizer2 = HashingVectorizer512(n_features=512, seed=42)
        
        vector1 = self.vectorizer.encode(self.test_text, self.test_ts)
        vector2 = vectorizer2.encode(self.test_text, self.test_ts)
        
        # Different seeds should produce different vectors
        assert not np.array_equal(vector1, vector2)
    
    def test_length_buckets(self):
        """Test length bucket assignment"""
        short_text = "Short"
        medium_text = "This is a medium length text that should fall into the medium bucket"
        long_text = "This is a very long text that should definitely fall into the long bucket because it has many more characters than the medium bucket allows"
        
        # Test length bucket assignment
        short_vector = self.vectorizer.encode(short_text)
        medium_vector = self.vectorizer.encode(medium_text)
        long_vector = self.vectorizer.encode(long_text)
        
        # All should have different length metadata features
        assert not np.array_equal(short_vector, medium_vector)
        assert not np.array_equal(medium_vector, long_vector)
        assert not np.array_equal(short_vector, long_vector)
    
    def test_hour_buckets(self):
        """Test hour bucket assignment"""
        # Test different hours
        morning_ts = 1724025600  # Early morning
        afternoon_ts = 1724054400  # Afternoon
        evening_ts = 1724083200   # Evening
        
        morning_vector = self.vectorizer.encode(self.test_text, morning_ts)
        afternoon_vector = self.vectorizer.encode(self.test_text, afternoon_ts)
        evening_vector = self.vectorizer.encode(self.test_text, evening_ts)
        
        # Should be different due to hour metadata
        assert not np.array_equal(morning_vector, afternoon_vector)
        assert not np.array_equal(afternoon_vector, evening_vector)
        assert not np.array_equal(morning_vector, evening_vector)
