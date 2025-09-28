"""
BPE (Byte Pair Encoding) tokenizer implementation with configurable limits.
"""
import re
import pickle
import os
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict
import json


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer with configurable limits to prevent infinite merging.
    """
    
    def __init__(self, 
                 vocab_size: int = 1000,
                 min_frequency: int = 2,
                 max_merges: Optional[int] = None,
                 special_tokens: Optional[List[str]] = None,
                 end_of_word_token: str = "</w>"):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (including special tokens)
            min_frequency: Minimum frequency for a merge to be considered
            max_merges: Maximum number of merges to perform (None = no limit)
            special_tokens: List of special tokens to include
            end_of_word_token: Token to append to end of words
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_merges = max_merges
        self.end_of_word_token = end_of_word_token
        
        # Special tokens
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.special_tokens = special_tokens
        
        # Vocabulary mappings
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_built = False
        
        # BPE-specific data
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
    def build_vocab(self, text: str) -> None:
        """
        Build BPE vocabulary from text corpus.
        
        Args:
            text: Input text corpus
        """
        print("Building BPE vocabulary...")
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split text into words
        words = text.split()
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Initialize vocabulary with special tokens
        self.word_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        self.idx_to_word = {i: token for i, token in enumerate(self.special_tokens)}
        
        # Add all characters as initial vocabulary
        chars = set()
        for word in word_counts.keys():
            chars.update(word)
        
        # Add characters to vocabulary
        for char in sorted(chars):
            if char not in self.word_to_idx:
                self.word_to_idx[char] = len(self.word_to_idx)
                self.idx_to_word[len(self.idx_to_word)] = char
        
        # Add end-of-word token if not already present
        if self.end_of_word_token not in self.word_to_idx:
            self.word_to_idx[self.end_of_word_token] = len(self.word_to_idx)
            self.idx_to_word[len(self.idx_to_word)] = self.end_of_word_token
        
        print(f"Initial vocabulary size: {len(self.word_to_idx)}")
        
        # Perform BPE merges
        self._perform_bpe_merges(word_counts)
        
        self.vocab_built = True
        print(f"BPE vocabulary built with {len(self.word_to_idx)} tokens")
        print(f"Number of merges performed: {len(self.merges)}")
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        return text.strip()
    
    def _perform_bpe_merges(self, word_counts: Counter) -> None:
        """
        Perform BPE merges to build vocabulary.
        
        Args:
            word_counts: Counter of word frequencies
        """
        # Convert words to character sequences with end-of-word token
        word_vocab = {}
        for word, count in word_counts.items():
            word_vocab[word] = count
        
        # Get initial character frequencies
        char_counts = defaultdict(int)
        for word, count in word_counts.items():
            for char in word:
                char_counts[char] += count
        
        # Perform merges
        merge_count = 0
        while len(self.word_to_idx) < self.vocab_size:
            # Check if we've reached max merges
            if self.max_merges is not None and merge_count >= self.max_merges:
                print(f"Reached maximum merges limit: {self.max_merges}")
                break
            
            # Find most frequent pair
            pair_counts = defaultdict(int)
            for word, count in word_vocab.items():
                pairs = self._get_pairs(word)
                for pair in pairs:
                    pair_counts[pair] += count
            
            # Filter by minimum frequency
            valid_pairs = {pair: count for pair, count in pair_counts.items() 
                          if count >= self.min_frequency}
            
            if not valid_pairs:
                print("No more valid pairs found (below min_frequency)")
                break
            
            # Get most frequent pair
            best_pair = max(valid_pairs, key=valid_pairs.get)
            best_count = valid_pairs[best_pair]
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.word_to_idx:
                self.word_to_idx[merged_token] = len(self.word_to_idx)
                self.idx_to_word[len(self.idx_to_word)] = merged_token
                
                # Record the merge
                self.merges.append(best_pair)
                self.bpe_ranks[best_pair] = merge_count
                merge_count += 1
                
                # Update word vocabulary
                new_word_vocab = {}
                for word, count in word_vocab.items():
                    new_word = self._merge_pair_in_word(word, best_pair)
                    new_word_vocab[new_word] = count
                word_vocab = new_word_vocab
                
                print(f"Merge {merge_count}: '{best_pair[0]}' + '{best_pair[1]}' -> '{merged_token}' (freq: {best_count})")
            else:
                # This pair already exists, skip it
                break
        
        print(f"BPE training completed with {merge_count} merges")
    
    def _get_pairs(self, word: str) -> List[Tuple[str, str]]:
        """Get all adjacent pairs in a word."""
        pairs = []
        prev_char = word[0]
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        return pairs
    
    def _merge_pair_in_word(self, word: str, pair: Tuple[str, str]) -> str:
        """Merge a specific pair in a word."""
        # Add end-of-word token if not present
        if not word.endswith(self.end_of_word_token):
            word = word + self.end_of_word_token
        
        # Replace all occurrences of the pair
        merged = ''.join(pair)
        return word.replace(pair[0] + pair[1], merged)
    
    def _apply_bpe(self, word: str) -> List[str]:
        """
        Apply BPE to a single word.
        
        Args:
            word: Input word
            
        Returns:
            List of BPE tokens
        """
        if word in self.word_to_idx:
            return [word]
        
        # Start with characters
        word = word + self.end_of_word_token
        pairs = self._get_pairs(word)
        
        if not pairs:
            return [word]
        
        # Apply merges in order
        while True:
            # Find the highest priority pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
            
            # Merge the pair
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                # Find next occurrence of first character
                j = word.find(first, i)
                if j == -1:
                    # No more occurrences, add remaining characters
                    new_word.extend(word[i:])
                    break
                
                # Add characters before the match
                new_word.extend(word[i:j])
                i = j
                
                # Check if we can merge with the next character
                if (i + len(first) < len(word) and 
                    word[i:i+len(first)] == first and 
                    word[i+len(first)] == second):
                    # Merge the pair
                    new_word.append(first + second)
                    i += len(first) + len(second)
                else:
                    # Just add the first character
                    new_word.append(first)
                    i += len(first)
            
            word = ''.join(new_word)
            pairs = self._get_pairs(word)
            
            if not pairs:
                break
        
        return word.split()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token indices.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS tokens
            
        Returns:
            List of token indices
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into words
        words = text.split()
        
        # Apply BPE to each word
        tokens = []
        for word in words:
            bpe_tokens = self._apply_bpe(word)
            for token in bpe_tokens:
                if token in self.word_to_idx:
                    tokens.append(self.word_to_idx[token])
                else:
                    tokens.append(self.word_to_idx['<UNK>'])
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.word_to_idx['<SOS>']] + tokens + [self.word_to_idx['<EOS>']]
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token indices to text.
        
        Args:
            token_ids: List of token indices
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.idx_to_word:
                token = self.idx_to_word[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        # Join tokens and remove end-of-word markers
        text = ''.join(tokens)
        text = text.replace(self.end_of_word_token, ' ')
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'merges': self.merges,
            'bpe_ranks': self.bpe_ranks,
            'special_tokens': self.special_tokens,
            'end_of_word_token': self.end_of_word_token,
            'min_frequency': self.min_frequency,
            'max_merges': self.max_merges
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"BPE vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.vocab_size = vocab_data['vocab_size']
        self.merges = vocab_data['merges']
        self.bpe_ranks = vocab_data['bpe_ranks']
        self.special_tokens = vocab_data['special_tokens']
        self.end_of_word_token = vocab_data['end_of_word_token']
        self.min_frequency = vocab_data['min_frequency']
        self.max_merges = vocab_data['max_merges']
        self.vocab_built = True
        print(f"BPE vocabulary loaded from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_idx)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token indices."""
        return {token: self.word_to_idx[token] for token in self.special_tokens if token in self.word_to_idx}
    
    def get_stats(self) -> Dict[str, any]:
        """Get tokenizer statistics."""
        return {
            'vocab_size': len(self.word_to_idx),
            'num_merges': len(self.merges),
            'special_tokens': self.special_tokens,
            'end_of_word_token': self.end_of_word_token,
            'min_frequency': self.min_frequency,
            'max_merges': self.max_merges
        }


if __name__ == "__main__":
    # Test BPE tokenizer
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.tokenizer import load_text_corpus
    
    # Load sample text
    text = load_text_corpus("sample_corpus.txt")
    
    # Test with different configurations
    configs = [
        {"vocab_size": 100, "max_merges": 50},
        {"vocab_size": 500, "max_merges": 200},
        {"vocab_size": 1000, "max_merges": 500},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing BPE with config: {config}")
        print(f"{'='*60}")
        
        tokenizer = BPETokenizer(**config)
        tokenizer.build_vocab(text)
        
        # Test encoding/decoding
        test_text = "hello world this is a test"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        print(f"Original: '{test_text}'")
        print(f"Encoded: {encoded}")
        print(f"Decoded: '{decoded}'")
        print(f"Stats: {tokenizer.get_stats()}")
