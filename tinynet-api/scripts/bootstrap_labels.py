#!/usr/bin/env python3
"""
CLI script for TinyNet Bootstrap Labeller
Usage: python scripts/bootstrap_labels.py data/raw/*.md --out data/train.jsonl
"""

import argparse
import sys
from pathlib import Path
from app.ml.bootstrap import BootstrapLabeller


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="TinyNet Bootstrap Labeller - Auto-create training data from markdown files"
    )
    
    parser.add_argument(
        "input_files",
        nargs="+",
        type=str,
        help="Input markdown files to process (glob patterns supported)"
    )
    
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="data/train.jsonl",
        help="Output JSONL file path (default: data/train.jsonl)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Resolve input files
    input_paths = []
    for pattern in args.input_files:
        paths = list(Path(".").glob(pattern))
        if not paths:
            print(f"Warning: No files found matching pattern '{pattern}'")
        input_paths.extend(paths)
    
    if not input_paths:
        print("Error: No input files found")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 TinyNet Bootstrap Labeller")
    print(f"Input files: {len(input_paths)}")
    print(f"Output: {output_path}")
    print()
    
    # Initialize labeller
    labeller = BootstrapLabeller()
    
    # Process files
    print("📖 Processing files...")
    all_samples = []
    
    for file_path in input_paths:
        if args.verbose:
            print(f"  Processing: {file_path}")
        
        samples = labeller.label_file(file_path)
        all_samples.extend(samples)
        
        if args.verbose:
            print(f"    Found {len(samples)} samples")
    
    print(f"✅ Total samples: {len(all_samples)}")
    
    # Save results
    print(f"💾 Saving to {output_path}...")
    labeller.save_jsonl(all_samples, output_path)
    
    # Generate coverage report
    labeller.print_coverage_report(all_samples)
    
    print(f"🎉 Done! Training data saved to {output_path}")
    
    # Verify output file exists and has expected content
    if output_path.exists():
        with open(output_path, 'r') as f:
            lines = f.readlines()
            print(f"📊 Output file has {len(lines)} lines")
            
            if len(lines) >= 10:
                print("✅ Output file has >= 10 lines as expected")
            else:
                print(f"⚠️  Output file has {len(lines)} lines (expected >= 10)")
    else:
        print("❌ Output file was not created")
        sys.exit(1)


if __name__ == "__main__":
    main()
