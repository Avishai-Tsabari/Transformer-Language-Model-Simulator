#!/usr/bin/env python3
"""
Script to prepare sample_corpus.txt from existing dataset
"""

import shutil
from pathlib import Path

def main():
    """Prepare sample_corpus.txt from existing dataset."""
    print("📝 Preparing sample_corpus.txt for training...")
    
    dataset_dir = Path("datasets")
    simple_lyrics = dataset_dir / "taylor_swift_simple.txt"
    
    if simple_lyrics.exists():
        # Copy the simple lyrics to sample_corpus.txt
        shutil.copy2(simple_lyrics, "sample_corpus.txt")
        print("✓ sample_corpus.txt created with Taylor Swift lyrics")
        
        # Show file size
        size = simple_lyrics.stat().st_size
        print(f"  File size: {size:,} characters")
        
        # Show preview
        with open("sample_corpus.txt", "r", encoding="utf-8") as f:
            preview = f.read(200)
            print(f"  Preview: {preview}...")
            
    else:
        print("✗ Taylor Swift lyrics not found in dataset folder.")
        print("Please run: python scrape_lyrics.py")
        return
    
    print("\n🎉 Corpus prepared!")
    print("You can now train your model with:")
    print("  python train.py")


if __name__ == "__main__":
    main()
