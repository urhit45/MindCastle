"""
Keyword Bootstrap Labeller for TinyNet
Auto-creates training data from simple keyword/regex rules
"""

import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


class BootstrapLabeller:
    """
    Bootstrap labeller that maps text to categories and states using keyword rules.
    """
    
    def __init__(self):
        """Initialize the bootstrap labeller with keyword rules"""
        
        # Category keyword mappings
        self.category_keywords = {
            "Fitness": ["run", "miles", "gym", "workout", "lift", "cardio", "exercise"],
            "Running": ["run", "mile", "5k", "cadence", "splits", "tempo"],
            "Strength": ["squat", "deadlift", "bench", "press", "hypertrophy"],
            "Music": ["music", "song", "practice", "metronome", "record"],
            "Guitar": ["guitar", "chords", "voicings", "strum", "pick", "fret"],
            "Learning": ["read", "chapter", "course", "tutorial", "lecture"],
            "Admin": ["email", "invoice", "submit", "renew", "form", "document"],
            # Keep other categories empty for now
            "AI": [],
            "Finance": [],
            "Social": [],
            "Health": [],
            "Cooking": [],
            "Travel": [],
            "Work": [],
            "SideProject": [],
            "Design": [],
            "Reading": [],
            "Writing": [],
            "Mindfulness": [],
            "Household": []
        }
        
        # State keyword mappings with priority order
        self.state_keywords = {
            "start": ["start", "kick off", "begin", "new plan"],
            "continue": ["continue", "kept", "resume", "again", "back to"],
            "pause": ["pause", "hold", "break from"],
            "end": ["finish", "completed", "done"],
            "blocked": ["stuck", "blocked", "shin", "pain", "issue", "error"],
            "idea": ["idea", "brainstorm", "draft", "maybe"]
        }
        
        # Compile regex patterns for word-boundary aware matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        self.category_patterns = {}
        for category, keywords in self.category_keywords.items():
            patterns = []
            for keyword in keywords:
                # Word boundary aware pattern
                pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                patterns.append(pattern)
            self.category_patterns[category] = patterns
        
        self.state_patterns = {}
        for state, keywords in self.state_keywords.items():
            patterns = []
            for keyword in keywords:
                pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                patterns.append(pattern)
            self.state_patterns[state] = patterns
    
    def _find_category_matches(self, text: str) -> List[str]:
        """
        Find matching categories for the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of matching category names (title-cased)
        """
        text_lower = text.lower()
        matches = []
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    matches.append(category)
                    break  # Only match once per category
        
        return matches
    
    def _find_state_match(self, text: str) -> str:
        """
        Find the best matching state for the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Best matching state name
        """
        text_lower = text.lower()
        best_match = "continue"  # Default fallback
        
        # Check each state in priority order
        for state, patterns in self.state_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return state
        
        return best_match
    
    def label_text(self, text: str, source: str = "", timestamp: Optional[int] = None) -> Dict:
        """
        Label a single text with categories and state.
        
        Args:
            text: Input text to label
            source: Source file or identifier
            timestamp: Unix timestamp (optional)
            
        Returns:
            Dictionary with labels and metadata
        """
        # Skip empty lines
        if not text.strip():
            return None
        
        # Find categories and state
        categories = self._find_category_matches(text)
        state = self._find_state_match(text)
        
        # If no categories found, use "Misc" if available
        if not categories:
            if "Misc" in self.category_keywords:
                categories = ["Misc"]
        
        # Create result
        result = {
            "text": text.strip(),
            "categories": categories,
            "state": state,
            "meta": {
                "source": source,
                "ts": timestamp
            }
        }
        
        return result
    
    def label_file(self, file_path: Path) -> List[Dict]:
        """
        Label all lines in a file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of labeled samples
        """
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        sample = self.label_text(
                            line, 
                            source=str(file_path),
                            timestamp=None
                        )
                        if sample:
                            samples.append(sample)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return samples
    
    def label_files(self, file_paths: List[Path]) -> List[Dict]:
        """
        Label multiple files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of all labeled samples
        """
        all_samples = []
        
        for file_path in file_paths:
            samples = self.label_file(file_path)
            all_samples.extend(samples)
        
        return all_samples
    
    def save_jsonl(self, samples: List[Dict], output_path: Path):
        """
        Save samples to JSONL format.
        
        Args:
            samples: List of labeled samples
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
    
    def generate_coverage_report(self, samples: List[Dict]) -> Dict:
        """
        Generate a coverage report for categories and states.
        
        Args:
            samples: List of labeled samples
            
        Returns:
            Dictionary with coverage statistics
        """
        category_counts = {}
        state_counts = {}
        
        # Initialize counts
        for category in self.category_keywords.keys():
            category_counts[category] = 0
        for state in self.state_keywords.keys():
            state_counts[state] = 0
        
        # Count occurrences
        for sample in samples:
            for category in sample["categories"]:
                if category in category_counts:
                    category_counts[category] += 1
            
            state = sample["state"]
            if state in state_counts:
                state_counts[state] += 1
        
        return {
            "total_samples": len(samples),
            "categories": category_counts,
            "states": state_counts
        }
    
    def print_coverage_report(self, samples: List[Dict]):
        """Print a formatted coverage report"""
        report = self.generate_coverage_report(samples)
        
        print(f"\n📊 Label Coverage Report")
        print(f"Total samples: {report['total_samples']}")
        
        print(f"\n📁 Categories:")
        for category, count in report['categories'].items():
            if count > 0:
                print(f"  {category}: {count}")
        
        print(f"\n🔄 States:")
        for state, count in report['states'].items():
            if count > 0:
                print(f"  {state}: {count}")
        
        print()
