"""
Demo script to test the transformer components.
"""
import torch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.tokenizer import SimpleTokenizer, load_text_corpus, create_sample_corpus
from src.data.dataset import TextDataset
from src.model.transformer import TransformerLM
from src.model.lightning_module import TransformerLightningModule
from src.utils.generation import TextGenerator, ProbabilityVisualizer


def test_tokenizer():
    """Test the tokenizer."""
    print("Testing tokenizer...")
    
    # Create sample corpus
    create_sample_corpus("demo_corpus.txt")
    
    # Load and tokenize text
    text = load_text_corpus("demo_corpus.txt")
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.build_vocab(text)
    
    # Test encoding/decoding
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print("Tokenizer test passed!\n")
    
    return tokenizer, text


def test_dataset(tokenizer, text):
    """Test the dataset."""
    print("Testing dataset...")
    
    # Encode text
    token_ids = tokenizer.encode(text)
    
    # Create dataset
    dataset = TextDataset(token_ids, sequence_length=32, stride=1)
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loading
    input_seq, target_seq = dataset[0]
    print(f"Input shape: {input_seq.shape}")
    print(f"Target shape: {target_seq.shape}")
    print("Dataset test passed!\n")
    
    return token_ids


def test_transformer(vocab_size):
    """Test the transformer model."""
    print("Testing transformer...")
    
    # Create model
    model = TransformerLM(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    start_tokens = torch.tensor([[1, 2, 3]])
    generated = model.generate(start_tokens, max_length=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    print("Transformer test passed!\n")
    
    return model


def test_lightning_module(vocab_size):
    """Test the lightning module."""
    print("Testing lightning module...")
    
    model = TransformerLightningModule(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test training step
    loss = model.training_step((input_ids, target_ids), 0)
    print(f"Training loss: {loss}")
    
    # Test validation step
    val_loss = model.validation_step((input_ids, target_ids), 0)
    print(f"Validation loss: {val_loss}")
    print("Lightning module test passed!\n")
    
    return model


def test_generation(model, tokenizer):
    """Test text generation."""
    print("Testing text generation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = TextGenerator(model, tokenizer, device)
    
    # Test full response generation
    prompt = "The quick brown fox"
    response = generator.generate_full_response(prompt, max_length=50, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Test token-by-token generation
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    candidates = generator.get_next_token_candidates(input_ids, temperature=1.0, top_k=5)
    
    print("\nNext token candidates:")
    for i, candidate in enumerate(candidates):
        print(f"{i+1}. '{candidate['token_text']}' (prob: {candidate['probability']:.4f})")
    
    # Test probability visualization
    bars = ProbabilityVisualizer.create_probability_bars(candidates)
    print("\nProbability bars:")
    for bar in bars:
        print(bar)
    
    print("Generation test passed!\n")


def main():
    """Run all tests."""
    print("="*50)
    print("TRANSFORMER LANGUAGE MODEL DEMO")
    print("="*50)
    
    try:
        # Test tokenizer
        tokenizer, text = test_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        # Test dataset
        token_ids = test_dataset(tokenizer, text)
        
        # Test transformer
        model = test_transformer(vocab_size)
        
        # Test lightning module
        lightning_model = test_lightning_module(vocab_size)
        
        # Test generation
        test_generation(lightning_model, tokenizer)
        
        print("="*50)
        print("ALL TESTS PASSED! 🎉")
        print("="*50)
        print("\nNext steps:")
        print("1. Run training: python train.py --create_sample")
        print("2. Run Gradio app: python -m src.app.gradio_app")
        print("3. Monitor training: tensorboard --logdir=./logs")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
