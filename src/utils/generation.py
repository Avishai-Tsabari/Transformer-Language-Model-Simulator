"""
Text generation utilities with temperature control and sampling strategies.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class TextGenerator:
    """
    Text generator with various sampling strategies and temperature control.
    """
    
    def __init__(self, model, tokenizer, device: str = 'cpu'):
        """
        Initialize text generator.
        
        Args:
            model: Trained transformer model
            tokenizer: Tokenizer instance
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate_full_response(self,
                             prompt: str,
                             max_length: int = 100,
                             temperature: float = 1.0,
                             top_k: Optional[int] = None,
                             top_p: Optional[float] = None,
                             repetition_penalty: float = 1.0) -> str:
        """
        Generate a complete response from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Penalty for repeated tokens
            
        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        generated_ids = self._generate_sequence(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        
        return generated_text
    
    def get_next_token_candidates(self,
                                input_ids: List[int],
                                temperature: float = 1.0,
                                top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top-k candidates for next token with probabilities.
        
        Args:
            input_ids: Current input token sequence
            temperature: Sampling temperature
            top_k: Number of top candidates to return
            
        Returns:
            List of dictionaries with token info and probabilities
        """
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            # Get logits for next token
            logits = self.model(input_tensor)
            
            # Handle temperature = 0 (deterministic selection)
            if temperature == 0.0:
                next_token_logits = logits[:, -1, :]
                # For temperature 0, we want the most likely token (argmax)
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                # Create one-hot probabilities (1.0 for top token, 0.0 for others)
                probabilities = torch.zeros_like(top_k_logits)
                probabilities[:, 0] = 1.0  # Top token gets probability 1.0
            else:
                next_token_logits = logits[:, -1, :] / temperature
                # Get top-k tokens and probabilities
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probabilities = F.softmax(top_k_logits, dim=-1)
            
            # Convert to list of dictionaries
            candidates = []
            for i in range(top_k):
                token_id = top_k_indices[0, i].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                probability = probabilities[0, i].item()
                
                candidates.append({
                    'token_id': token_id,
                    'token_text': token_text,
                    'probability': probability
                })
        
        return candidates
    
    def _generate_sequence(self,
                          input_tensor: torch.Tensor,
                          max_length: int,
                          temperature: float,
                          top_k: Optional[int],
                          top_p: Optional[float],
                          repetition_penalty: float) -> torch.Tensor:
        """
        Generate a sequence of tokens.
        
        Args:
            input_tensor: Starting input tensor
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Penalty for repeated tokens
            
        Returns:
            Generated token sequence
        """
        generated = input_tensor.clone()
        
        for _ in range(max_length - input_tensor.size(1)):
            # Get logits for next token
            logits = self.model(generated)
            
            # Handle temperature = 0 (deterministic selection)
            if temperature == 0.0:
                next_token_logits = logits[:, -1, :]
                # For temperature 0, select the most likely token (argmax)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in generated[0]:
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def interactive_generation(self,
                             initial_prompt: str = "",
                             max_tokens: int = 100) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Interactive token-by-token generation.
        
        Args:
            initial_prompt: Starting prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Tuple of (current_text, next_token_candidates)
        """
        # Encode initial prompt
        if initial_prompt:
            input_ids = self.tokenizer.encode(initial_prompt, add_special_tokens=True)
        else:
            input_ids = [self.tokenizer.char_to_idx[self.tokenizer.SOS_TOKEN]]
        
        # Get next token candidates
        candidates = self.get_next_token_candidates(input_ids)
        
        # Decode current text
        current_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        return current_text, candidates
    
    def add_token_to_sequence(self,
                            current_ids: List[int],
                            token_id: int) -> List[int]:
        """
        Add a token to the current sequence.
        
        Args:
            current_ids: Current token sequence
            token_id: Token ID to add
            
        Returns:
            Updated token sequence
        """
        return current_ids + [token_id]
    
    def get_sequence_text(self, token_ids: List[int]) -> str:
        """
        Convert token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class ProbabilityVisualizer:
    """
    Utility class for visualizing token probabilities.
    """
    
    @staticmethod
    def create_probability_bars(candidates: List[Dict[str, Any]], 
                              max_width: int = 50,
                              token_width: int = 15) -> List[str]:
        """
        Create ASCII bar charts for token probabilities with aligned bars.
        
        Args:
            candidates: List of token candidates with probabilities
            max_width: Maximum width of bars
            token_width: Fixed width for token strings
            
        Returns:
            List of strings representing probability bars
        """
        if not candidates:
            return []
        
        # Normalize probabilities to 0-1 range
        max_prob = max(candidate['probability'] for candidate in candidates)
        normalized_probs = [candidate['probability'] / max_prob for candidate in candidates]
        
        bars = []
        for i, (candidate, norm_prob) in enumerate(zip(candidates, normalized_probs)):
            # Create bar
            bar_length = int(norm_prob * max_width)
            bar = '█' * bar_length + '░' * (max_width - bar_length)
            
            # Format token text (escape special characters)
            token_text = candidate['token_text']
            if token_text == ' ':
                token_text = '[SPACE]'
            elif token_text == '\n':
                token_text = '[NEWLINE]'
            elif token_text == '\t':
                token_text = '[TAB]'
            
            # Truncate long tokens and add ellipsis
            display_text = token_text
            if len(token_text) > token_width:
                display_text = token_text[:token_width-3] + '...'
            
            # Create formatted line with aligned bars
            line = f"{i+1:2d}. {display_text:<{token_width}} {bar} {candidate['probability']:.4f}"
            bars.append(line)
        
        return bars
    
    @staticmethod
    def create_html_probability_bars(candidates: List[Dict[str, Any]], 
                                   selected_idx: int = -1,
                                   token_width: int = 15) -> str:
        """
        Create HTML bar charts for token probabilities with alignment and highlighting.
        
        Args:
            candidates: List of token candidates with probabilities
            selected_idx: Index of selected token to highlight (-1 for none)
            token_width: Fixed width for token strings
            
        Returns:
            HTML string with probability bars
        """
        if not candidates:
            return ""
        
        html = """
        <div style='font-size: 14px;'>
            <style>
            @keyframes highlight {
                0% { background-color: #ffeb3b; border: 2px solid #ff9800; }
                50% { background-color: #fff59d; border: 2px solid #ff9800; }
                100% { background-color: transparent; border: 2px solid transparent; }
            }
            .token-row {
                margin: 2px 0; 
                display: flex; 
                align-items: center;
                transition: all 0.3s ease;
            }
            .highlighted {
                background-color: #ffeb3b;
                border: 2px solid #ff9800;
                border-radius: 4px;
                padding: 2px;
                animation: highlight 1.5s ease-out forwards;
            }
            </style>
        """
        
        for i, candidate in enumerate(candidates):
            probability = candidate['probability']
            percentage = probability * 100
            
            # Format token text
            token_text = candidate['token_text']
            if token_text == ' ':
                token_text = '[SPACE]'
            elif token_text == '\n':
                token_text = '[NEWLINE]'
            elif token_text == '\t':
                token_text = '[TAB]'
            
            # Create display text with truncation
            display_text = token_text
            full_text = token_text
            if len(token_text) > token_width:
                display_text = token_text[:token_width-3] + '...'
            
            # Create bar
            bar_width = int(probability * 200)  # Max width of 200px
            bar_color = f"hsl({120 * probability:.0f}, 70%, 50%)"  # Green gradient
            
            # Add highlighting class if selected
            highlight_class = "highlighted" if (i == selected_idx) else ""
            
            html += f"""
            <div class='token-row {highlight_class}'>
                <span style='width: 20px; text-align: right; margin-right: 10px;'>{i+1:2d}.</span>
                <span style='width: {token_width * 8}px; margin-right: 10px;' title='{full_text}'>{display_text}</span>
                <div style='width: 200px; height: 20px; background: #f0f0f0; border-radius: 3px; overflow: hidden; margin-right: 10px;'>
                    <div style='width: {bar_width}px; height: 100%; background: {bar_color}; transition: width 0.3s ease;'></div>
                </div>
                <span style='width: 60px; text-align: right;'>{percentage:.2f}%</span>
            </div>
            """
        
        html += "</div>"
        return html


if __name__ == "__main__":
    # Test text generator
    print("Text generation utilities test completed!")
