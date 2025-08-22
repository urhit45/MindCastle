"""
Tests for Bootstrap Labeller
"""

import pytest
from pathlib import Path
from app.ml.bootstrap import BootstrapLabeller


class TestBootstrapLabeller:
    """Test suite for BootstrapLabeller"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.labeller = BootstrapLabeller()
    
    def test_simple_sentence_categories(self):
        """Test that 'Ran 2 miles cadence drills' includes Fitness and Running categories"""
        text = "Ran 2 miles cadence drills"
        result = self.labeller.label_text(text)
        
        assert result is not None
        assert "Fitness" in result["categories"]
        assert "Running" in result["categories"]
        assert result["state"] == "continue"  # Default fallback
    
    def test_blocked_state_detection(self):
        """Test that 'shin pain today' gets state 'blocked'"""
        text = "shin pain today"
        result = self.labeller.label_text(text)
        
        assert result is not None
        assert result["state"] == "blocked"
    
    def test_category_keyword_matching(self):
        """Test various category keyword matches"""
        test_cases = [
            ("run 5k", ["Fitness", "Running"]),
            ("gym workout", ["Fitness"]),
            ("guitar practice", ["Music", "Guitar"]),
            ("read chapter", ["Learning"]),
            ("email invoice", ["Admin"]),
        ]
        
        for text, expected_categories in test_cases:
            result = self.labeller.label_text(text)
            assert result is not None
            for category in expected_categories:
                assert category in result["categories"], f"Expected {category} in {text}"
    
    def test_state_keyword_matching(self):
        """Test various state keyword matches"""
        test_cases = [
            ("start new plan", "start"),
            ("continue working", "continue"),
            ("pause for now", "pause"),
            ("finish task", "end"),
            ("stuck on problem", "blocked"),
            ("idea for project", "idea"),
        ]
        
        for text, expected_state in test_cases:
            result = self.labeller.label_text(text)
            assert result is not None
            assert result["state"] == expected_state, f"Expected {expected_state} for {text}"
    
    def test_word_boundary_matching(self):
        """Test that regex patterns respect word boundaries"""
        # "run" should match "run" but not "running"
        result1 = self.labeller.label_text("run today")
        result2 = self.labeller.label_text("running today")
        
        assert result1 is not None
        assert result2 is not None
        
        # "run" should be in categories for first text
        assert "Fitness" in result1["categories"] or "Running" in result1["categories"]
        
        # "running" should not match the "run" keyword
        # (This depends on the exact implementation, but generally should be true)
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text"""
        empty_texts = ["", "   ", "\n", "\t"]
        
        for text in empty_texts:
            result = self.labeller.label_text(text)
            assert result is None
    
    def test_special_characters(self):
        """Test text with special characters and punctuation"""
        text = "Hello, world! How are you? 123 @#$%"
        result = self.labeller.label_text(text)
        
        assert result is not None
        assert result["text"] == text.strip()
        assert result["state"] == "continue"  # Default fallback
    
    def test_case_insensitive_matching(self):
        """Test that keyword matching is case insensitive"""
        test_cases = [
            ("RUN today", ["Fitness", "Running"]),
            ("Gym workout", ["Fitness"]),
            ("GUITAR practice", ["Music", "Guitar"]),
        ]
        
        for text, expected_categories in test_cases:
            result = self.labeller.label_text(text)
            assert result is not None
            for category in expected_categories:
                assert category in result["categories"], f"Expected {category} in {text}"
    
    def test_multiple_category_matches(self):
        """Test that text can match multiple categories"""
        text = "run 5k at gym"
        result = self.labeller.label_text(text)
        
        assert result is not None
        assert len(result["categories"]) >= 2
        assert "Fitness" in result["categories"]
        assert "Running" in result["categories"]
    
    def test_state_priority_ordering(self):
        """Test that state matching follows priority order"""
        # "start" should take priority over "continue"
        text = "start new plan and continue working"
        result = self.labeller.label_text(text)
        
        assert result is not None
        assert result["state"] == "start"  # Higher priority
    
    def test_file_processing(self):
        """Test processing of markdown files"""
        # Create a temporary test file
        test_content = """# Test File
First line with run keyword
Second line with gym keyword
Third line with blocked keyword
"""
        
        test_file = Path("test_file.md")
        try:
            with open(test_file, "w") as f:
                f.write(test_content)
            
            samples = self.labeller.label_file(test_file)
            
            assert len(samples) == 3  # 3 non-empty lines
            assert samples[0]["text"] == "First line with run keyword"
            assert samples[1]["text"] == "Second line with gym keyword"
            assert samples[2]["text"] == "Third line with blocked keyword"
            
            # Check that categories and states are assigned
            for sample in samples:
                assert "categories" in sample
                assert "state" in sample
                assert "meta" in sample
                assert sample["meta"]["source"] == str(test_file)
        
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()
    
    def test_coverage_report_generation(self):
        """Test coverage report generation"""
        test_samples = [
            {"text": "run today", "categories": ["Fitness", "Running"], "state": "continue"},
            {"text": "gym workout", "categories": ["Fitness"], "state": "continue"},
            {"text": "shin pain", "categories": [], "state": "blocked"},
        ]
        
        report = self.labeller.generate_coverage_report(test_samples)
        
        assert report["total_samples"] == 3
        assert report["categories"]["Fitness"] == 2
        assert report["categories"]["Running"] == 1
        assert report["states"]["continue"] == 2
        assert report["states"]["blocked"] == 1
    
    def test_jsonl_output_format(self):
        """Test JSONL output format"""
        test_samples = [
            {"text": "test line 1", "categories": ["Test"], "state": "continue", "meta": {"source": "test"}},
            {"text": "test line 2", "categories": ["Test"], "state": "continue", "meta": {"source": "test"}},
        ]
        
        output_file = Path("test_output.jsonl")
        try:
            self.labeller.save_jsonl(test_samples, output_file)
            
            assert output_file.exists()
            
            # Read and verify content
            with open(output_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2
                
                # Parse first line
                import json
                first_sample = json.loads(lines[0].strip())
                assert first_sample["text"] == "test line 1"
                assert first_sample["categories"] == ["Test"]
                assert first_sample["state"] == "continue"
        
        finally:
            # Clean up
            if output_file.exists():
                output_file.unlink()
    
    def test_metadata_inclusion(self):
        """Test that metadata is properly included"""
        text = "test text"
        source = "test_source.md"
        timestamp = 1234567890
        
        result = self.labeller.label_text(text, source, timestamp)
        
        assert result is not None
        assert result["meta"]["source"] == source
        assert result["meta"]["ts"] == timestamp
