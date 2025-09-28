"""
Tokenizer factory for choosing between different tokenization methods.
"""
from typing import Union, Dict, Any
from .tokenizer import SimpleTokenizer
from .simple_bpe_tokenizer import SimpleBPETokenizer


class TokenizerFactory:
    """
    Factory class for creating different types of tokenizers.
    """
    
    @staticmethod
    def create_tokenizer(tokenizer_type: str = "word", **kwargs) -> Union[SimpleTokenizer, SimpleBPETokenizer]:
        """
        Create a tokenizer of the specified type.
        
        Args:
            tokenizer_type: Type of tokenizer ("word" or "bpe")
            **kwargs: Additional arguments passed to the tokenizer constructor
            
        Returns:
            Tokenizer instance
        """
        if tokenizer_type.lower() == "word":
            return SimpleTokenizer(**kwargs)
        elif tokenizer_type.lower() == "bpe":
            return SimpleBPETokenizer(**kwargs)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. "
                           f"Supported types: 'word', 'bpe'")
    
    @staticmethod
    def get_default_config(tokenizer_type: str = "word") -> Dict[str, Any]:
        """
        Get default configuration for a tokenizer type.
        
        Args:
            tokenizer_type: Type of tokenizer ("word" or "bpe")
            
        Returns:
            Default configuration dictionary
        """
        if tokenizer_type.lower() == "word":
            return {
                "vocab_size": 1500
            }
        elif tokenizer_type.lower() == "bpe":
            return {
                "vocab_size": 1500,
                "min_frequency": 2,
                "max_merges": None,  # No limit by default
                "special_tokens": ['<PAD>', '<UNK>', '<SOS>', '<EOS>'],
                "end_of_word_token": "</w>"
            }
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> Union[SimpleTokenizer, SimpleBPETokenizer]:
        """
        Create a tokenizer from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with 'type' and other parameters
            
        Returns:
            Tokenizer instance
        """
        tokenizer_type = config.pop("type", "word")
        return TokenizerFactory.create_tokenizer(tokenizer_type, **config)


def create_tokenizer_from_config(config: Dict[str, Any]) -> Union[SimpleTokenizer, SimpleBPETokenizer]:
    """
    Convenience function to create a tokenizer from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tokenizer instance
    """
    return TokenizerFactory.create_from_config(config)


if __name__ == "__main__":
    # Test tokenizer factory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.tokenizer import load_text_corpus
    
    # Load sample text
    text = load_text_corpus("sample_corpus.txt")
    
    # Test word tokenizer
    print("Testing Word Tokenizer:")
    word_tokenizer = TokenizerFactory.create_tokenizer("word", vocab_size=100)
    word_tokenizer.build_vocab(text)
    print(f"Word vocab size: {word_tokenizer.get_vocab_size()}")
    
    # Test BPE tokenizer
    print("\nTesting BPE Tokenizer:")
    bpe_tokenizer = TokenizerFactory.create_tokenizer("bpe", vocab_size=100, max_merges=50)
    bpe_tokenizer.build_vocab(text)
    print(f"BPE vocab size: {bpe_tokenizer.get_vocab_size()}")
    
    # Test with config
    print("\nTesting with config:")
    config = {
        "type": "bpe",
        "vocab_size": 200,
        "max_merges": 100,
        "min_frequency": 3
    }
    config_tokenizer = TokenizerFactory.create_from_config(config)
    config_tokenizer.build_vocab(text)
    print(f"Config tokenizer vocab size: {config_tokenizer.get_vocab_size()}")
