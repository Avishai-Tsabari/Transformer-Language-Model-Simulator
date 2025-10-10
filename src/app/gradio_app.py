"""
Gradio interface for the transformer language model.
"""
import gradio as gr
import torch
import os
import sys
import pickle
from typing import List, Dict, Any, Tuple
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.tokenizer import SimpleTokenizer
from src.data.simple_bpe_tokenizer import SimpleBPETokenizer
from src.model.lightning_module import TransformerLightningModule
from src.utils.generation import TextGenerator, ProbabilityVisualizer


class GradioApp:
    """
    Gradio application for transformer language model.
    """
    
    def __init__(self, model_path: str = None, vocab_path: str = None):
        """
        Initialize the Gradio app.
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
        """
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.current_sequence = []
        self.current_text = ""
        
        # Load model and tokenizer if paths provided
        if model_path and vocab_path:
            self.load_model(model_path, vocab_path)
    
    def _detect_tokenizer_type(self, vocab_path: str) -> str:
        """
        Detect tokenizer type by examining the vocabulary file.
        
        Args:
            vocab_path: Path to vocabulary file
            
        Returns:
            Tokenizer type ("word" or "bpe")
        """
        try:
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
            
            # Check if it's a BPE tokenizer (has tokenizer_path key)
            if 'tokenizer_path' in vocab_data:
                return "bpe"
            # Check if it's a word tokenizer (has word_to_idx key)
            elif 'word_to_idx' in vocab_data:
                return "word"
            else:
                # Default to BPE if we can't determine
                print("Warning: Could not determine tokenizer type, defaulting to BPE")
                return "bpe"
        except Exception as e:
            print(f"Warning: Error detecting tokenizer type: {e}, defaulting to BPE")
            return "bpe"
    
    def load_model(self, model_path: str, vocab_path: str):
        """
        Load trained model and tokenizer.
        
        Args:
            model_path: Path to model checkpoint
            vocab_path: Path to vocabulary file
        """
        try:
            # Detect tokenizer type by examining vocab file
            tokenizer_type = self._detect_tokenizer_type(vocab_path)
            
            # Load appropriate tokenizer
            if tokenizer_type == "bpe":
                self.tokenizer = SimpleBPETokenizer()
            else:
                self.tokenizer = SimpleTokenizer()
            
            self.tokenizer.load_vocab(vocab_path)
            
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model = TransformerLightningModule.load_from_checkpoint(model_path)
            self.model.eval()
            
            # Initialize generator
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.generator = TextGenerator(self.model, self.tokenizer, device)
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            self.generator = None
    
    def generate_full_response(self, 
                             prompt: str, 
                             max_length: int, 
                             temperature: float,
                             top_k: int,
                             top_p: float) -> str:
        """
        Generate a complete response.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Generated text
        """
        if not self.generator:
            return "Error: Model not loaded. Please load a trained model first."
        
        try:
            response = self.generator.generate_full_response(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if top_p > 0 else None
            )
            return response
        except Exception as e:
            return f"Error generating response: {e}"
    
    def start_token_generation(self, prompt: str) -> Tuple[str, str, str]:
        """
        Start token-by-token generation.
        
        Args:
            prompt: Initial prompt
            
        Returns:
            Tuple of (current_text, probability_bars_html, next_tokens_json)
        """
        if not self.generator:
            return "Error: Model not loaded.", "", "[]"
        
        try:
            # Initialize sequence
            if prompt:
                self.current_sequence = self.tokenizer.encode(prompt, add_special_tokens=True)
            else:
                # Get SOS token - handle both tokenizer types
                if hasattr(self.tokenizer, 'get_special_tokens'):
                    # BPE tokenizer
                    special_tokens = self.tokenizer.get_special_tokens()
                    sos_token_id = special_tokens.get('SOS', 2)  # Default to 2 if not found
                else:
                    # Word tokenizer
                    sos_token_id = self.tokenizer.word_to_idx[self.tokenizer.SOS_TOKEN]
                self.current_sequence = [sos_token_id]
            
            self.current_text = self.tokenizer.decode(self.current_sequence, skip_special_tokens=True)
            
            # Get next token candidates
            candidates = self.generator.get_next_token_candidates(
                self.current_sequence, 
                temperature=1.0,  # Default temperature
                top_k=10
            )
            
            # Create HTML probability bars
            probability_bars_html = ProbabilityVisualizer.create_html_probability_bars(candidates)
            
            # Create JSON for next tokens
            next_tokens_json = json.dumps(candidates, indent=2)
            
            return self.current_text, probability_bars_html, next_tokens_json
            
        except Exception as e:
            return f"Error: {e}", "", "[]"
    
    def update_probabilities_only(self, 
                                temperature: float) -> Tuple[str, str, str]:
        """
        Update probability display without adding tokens.
        
        Args:
            temperature: Sampling temperature
            
        Returns:
            Tuple of (probability_bars_html, next_tokens_json, status)
        """
        if not self.generator or not self.current_sequence:
            return "", "[]", "No active generation session"
        
        try:
            # Get next token candidates with current temperature
            candidates = self.generator.get_next_token_candidates(
                self.current_sequence,
                temperature=temperature,
                top_k=10
            )
            
            # Create HTML probability bars
            probability_bars_html = ProbabilityVisualizer.create_html_probability_bars(candidates)
            
            # Create JSON for next tokens
            next_tokens_json = json.dumps(candidates, indent=2)
            
            status = f"Probabilities updated for temperature {temperature:.1f}"
            
            return probability_bars_html, next_tokens_json, status
            
        except Exception as e:
            return "", "[]", f"Error: {e}"

    def predict_next_token(self, 
                          temperature: float, 
                          selected_token_idx: int = 0,
                          use_probability_sampling: bool = False) -> Tuple[str, str, str, str]:
        """
        Predict next token and update sequence.
        
        Args:
            temperature: Sampling temperature
            selected_token_idx: Index of selected token (0=disabled, 1-10=token position) - only used if use_probability_sampling=False
            use_probability_sampling: If True, sample based on probabilities; if False, use selected_token_idx
            
        Returns:
            Tuple of (current_text, probability_bars_html, next_tokens_json, status)
        """
        if not self.generator or not self.current_sequence:
            return "Error: No active generation session.", "", "[]", "Error"
        
        try:
            # Get next token candidates with current temperature
            candidates = self.generator.get_next_token_candidates(
                self.current_sequence,
                temperature=temperature,
                top_k=10
            )
            
            # Select token based on method
            actual_selected_idx = -1  # For highlighting
            if use_probability_sampling or selected_token_idx == 0:
                # Sample based on probabilities (either explicitly requested or when 0 is selected)
                import random
                probabilities = [candidate['probability'] for candidate in candidates]
                actual_selected_idx = random.choices(range(len(candidates)), weights=probabilities)[0]
                selected_token = candidates[actual_selected_idx]
            else:
                # Use index-based selection (1-10 map to 0-9 in candidates array)
                if selected_token_idx < 1 or selected_token_idx > 10:
                    return self.current_text, "", "[]", "Invalid token selection (must be 0-10)"
                if selected_token_idx - 1 >= len(candidates):
                    return self.current_text, "", "[]", "Invalid token selection"
                actual_selected_idx = selected_token_idx - 1
                selected_token = candidates[actual_selected_idx]
            
            # Add selected token to sequence
            self.current_sequence = self.generator.add_token_to_sequence(
                self.current_sequence, 
                selected_token['token_id']
            )
            
            # Update current text
            self.current_text = self.generator.get_sequence_text(self.current_sequence)
            
            # Get next token candidates for the updated sequence
            next_candidates = self.generator.get_next_token_candidates(
                self.current_sequence,
                temperature=temperature,
                top_k=10
            )
            
            # Create HTML probability bars WITHOUT highlighting (new list)
            probability_bars_html = ProbabilityVisualizer.create_html_probability_bars(
                next_candidates, 
                selected_idx=-1  # No highlighting for new list
            )
            
            # Create JSON for next tokens
            next_tokens_json = json.dumps(next_candidates, indent=2)
            
            if use_probability_sampling or (selected_token_idx == 0 and not use_probability_sampling):
                status = f"Added token: '{selected_token['token_text']}' (prob: {selected_token['probability']:.4f}) [sampled]"
            else:
                status = f"Added token: '{selected_token['token_text']}' (prob: {selected_token['probability']:.4f}) [selected position {selected_token_idx}]"
            
            return self.current_text, probability_bars_html, next_tokens_json, status
            
        except Exception as e:
            return f"Error: {e}", "", "[]", "Error"
    
    
    def reset_generation(self) -> Tuple[str, str, str, str]:
        """
        Reset the generation session.
        
        Returns:
            Tuple of (current_text, probability_bars_html, next_tokens_json, status)
        """
        self.current_sequence = []
        self.current_text = ""
        return "", "", "[]", "Generation session reset"
    
    def analyze_token_frequencies(self, corpus_path: str = "sample_corpus.txt") -> Tuple[str, str]:
        """
        Analyze token frequencies from the training corpus.
        
        Args:
            corpus_path: Path to the corpus file
            
        Returns:
            Tuple of (top_tokens_html, bottom_tokens_html)
        """
        if not self.tokenizer:
            return "Error: Tokenizer not loaded", "Error: Tokenizer not loaded"
        
        try:
            # Read corpus file
            with open(corpus_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Tokenize the entire corpus
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Count token frequencies
            token_counts = {}
            for token_id in tokens:
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
            
            # Get total token count
            total_tokens = len(tokens)
            
            # Convert to list of (token_id, count, percentage) tuples
            token_freq_data = []
            for token_id, count in token_counts.items():
                percentage = (count / total_tokens) * 100
                # Get token text, handling special tokens properly
                token_text = self._get_token_display_text(token_id)
                token_freq_data.append((token_id, token_text, count, percentage))
            
            # Sort by frequency (descending)
            token_freq_data.sort(key=lambda x: x[2], reverse=True)
            
            # Get top 20 and bottom 20 tokens
            top_tokens = token_freq_data[:20]
            bottom_tokens = token_freq_data[-20:]
            
            # Create HTML for top tokens
            top_html = self._create_token_frequency_html(top_tokens, "Most Frequent Tokens")
            
            # Create HTML for bottom tokens  
            bottom_html = self._create_token_frequency_html(bottom_tokens, "Least Frequent Tokens")
            
            return top_html, bottom_html
            
        except Exception as e:
            return f"Error analyzing frequencies: {e}", f"Error analyzing frequencies: {e}"
    
    def _get_token_display_text(self, token_id: int) -> str:
        """
        Get display text for a token, handling special tokens properly.
        
        Args:
            token_id: Token ID to get display text for
            
        Returns:
            Display text for the token
        """
        try:
            # Special token mappings based on what we know from the vocabulary
            special_tokens = {
                0: "<PAD>",
                1: "<UNK>", 
                2: "<SOS>",
                3: "<EOS>",
                4: "<LINE_BREAK>",
                5: "<SONG_BREAK>"
            }
            
            # Check if it's a special token first
            if token_id in special_tokens:
                return special_tokens[token_id]
            
            # For regular tokens, try to get from vocabulary first
            if hasattr(self.tokenizer, 'idx_to_word') and token_id in self.tokenizer.idx_to_word:
                token_text = self.tokenizer.idx_to_word[token_id]
                return token_text
            
            # Fallback: try decoding without skipping special tokens first
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            
            # If that gives us something, return it
            if token_text:
                return token_text
            
            # If still empty, try with skipping special tokens
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            # If still empty, it might be a special token that gets skipped
            if not token_text and token_id <= 5:
                return special_tokens.get(token_id, f"<ID:{token_id}>")
            
            return token_text if token_text else f"<ID:{token_id}>"
            
        except:
            return f"<ID:{token_id}>"
    
    def _create_token_frequency_html(self, token_data: list, title: str) -> str:
        """
        Create HTML display for token frequency data.
        
        Args:
            token_data: List of (token_id, token_text, count, percentage) tuples
            title: Title for the section
            
        Returns:
            HTML string
        """
        html = f"<div style='font-family: monospace; font-size: 12px;'>"
        html += f"<h4 style='margin-bottom: 10px; color: #2563eb;'>{title}</h4>"
        
        for i, (token_id, token_text, count, percentage) in enumerate(token_data):
            # Don't escape HTML for special tokens, only escape & for safety
            token_text = token_text.replace('&', '&amp;')
            
            # Create a simple bar visualization based on frequency
            max_count = max([x[2] for x in token_data]) if token_data else 1
            bar_width = min(100, (count / max_count) * 100)
            
            html += f"""
            <div style='margin-bottom: 5px; padding: 2px 5px; background-color: #f8fafc; border-radius: 3px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-weight: bold; color: #1e40af;'>{i+1:2d}.</span>
                    <span style='flex: 1; margin: 0 10px; color: #374151;'>"{token_text}"</span>
                    <span style='color: #6b7280; font-size: 11px;'>{count:,}</span>
                    <span style='color: #9ca3af; font-size: 11px;'>({percentage:.2f}%)</span>
                </div>
                <div style='width: 100%; background-color: #e5e7eb; border-radius: 2px; height: 4px; margin-top: 2px;'>
                    <div style='width: {bar_width}%; background-color: #3b82f6; height: 4px; border-radius: 2px;'></div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Transformer Language Model", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 Transformer Language Model")
            gr.Markdown("Generate text with a small transformer model. Choose between full response generation or interactive token-by-token prediction.")
            
            with gr.Tabs():
                # Token-by-Token Generation Tab
                with gr.Tab("Interactive Token Generation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Start Generation")
                            start_prompt = gr.Textbox(
                                label="Initial Prompt (optional):",
                                placeholder="Enter starting text...",
                                lines=2
                            )
                            start_btn = gr.Button("Start Generation", variant="primary")
                            
                            gr.Markdown("### Current Sequence")
                            current_text = gr.Textbox(
                                label="Generated Text:",
                                lines=6,
                                max_lines=6,
                                interactive=False,
                                show_copy_button=True
                            )
                            
                            gr.Markdown("### Controls")
                            with gr.Row():
                                temp_slider = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Temperature"
                                )
                                token_idx = gr.Number(
                                    value=0, minimum=0, maximum=10, step=1,
                                    label="Select Token (0=disabled, 1-10=from list)"
                                )
                            
                            with gr.Row():
                                use_prob_sampling = gr.Checkbox(
                                    value=False,
                                    label="Force Probability Sampling (overrides token selection)"
                                )
                            
                            with gr.Row():
                                predict_btn = gr.Button("Predict Next Token", variant="primary")
                                reset_btn = gr.Button("Reset", variant="secondary")
                            
                            status_text = gr.Textbox(
                                label="Status:",
                                interactive=False
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Next Token Probabilities")
                            probability_display = gr.HTML(
                                label="Probability Bars:",
                                value="<div style='font-size: 14px;'>No data available. Start generation to see probabilities.</div>"
                            )
                            
                            gr.Markdown("### Raw Token Data")
                            tokens_json = gr.JSON(
                                label="Token Data (JSON):"
                            )
            
                            # Full Response Generation Tab
                
                
                with gr.Tab("Full Response Generation"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="Enter your prompt:",
                                placeholder="Type your prompt here...",
                                lines=3
                            )
                            
                            with gr.Row():
                                max_length = gr.Slider(
                                    minimum=10, maximum=200, value=50, step=10,
                                    label="Max Length"
                                )
                                temperature = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                                    label="Temperature"
                                )
                            
                            with gr.Row():
                                top_k = gr.Slider(
                                    minimum=0, maximum=50, value=10, step=1,
                                    label="Top-K (0 = disabled)"
                                )
                                top_p = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.9, step=0.05,
                                    label="Top-P (0 = disabled)"
                                )
                            
                            generate_btn = gr.Button("Generate Response", variant="primary")
                        
                        with gr.Column(scale=3):
                            response_output = gr.Textbox(
                                label="Generated Response:",
                                lines=10,
                                interactive=False
                            )
                
                # Token Analysis Tab
                with gr.Tab("Token Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Token Frequency Analysis")
                            gr.Markdown("Analyze the most and least frequent tokens from the training corpus.")
                            
                            analyze_btn = gr.Button("Analyze Token Frequencies", variant="primary")
                            
                            gr.Markdown("### Analysis Info")
                            analysis_info = gr.Markdown(
                                "Click 'Analyze Token Frequencies' to analyze actual token usage from the training corpus. "
                                "This shows the top 20 most frequent and bottom 20 least frequent tokens with real frequency counts and percentages."
                            )
                        
                        with gr.Column(scale=1):
                            top_tokens_display = gr.HTML(
                                label="Most Frequent Tokens:",
                                value="<div style='font-size: 14px; color: #6b7280;'>Click 'Analyze Token Frequencies' to load data.</div>"
                            )
                        
                        with gr.Column(scale=1):
                            bottom_tokens_display = gr.HTML(
                                label="Least Frequent Tokens:",
                                value="<div style='font-size: 14px; color: #6b7280;'>Click 'Analyze Token Frequencies' to load data.</div>"
                            )
                
            # Event handlers for full response generation
            generate_btn.click(
                fn=self.generate_full_response,
                inputs=[prompt_input, max_length, temperature, top_k, top_p],
                outputs=[response_output]
            )
            
            # Event handlers for token-by-token generation
            start_btn.click(
                fn=self.start_token_generation,
                inputs=[start_prompt],
                outputs=[current_text, probability_display, tokens_json]
            )
            
            predict_btn.click(
                fn=self.predict_next_token,
                inputs=[temp_slider, token_idx, use_prob_sampling],
                outputs=[current_text, probability_display, tokens_json, status_text]
            )
            
            reset_btn.click(
                fn=self.reset_generation,
                outputs=[current_text, probability_display, tokens_json, status_text]
            )
            
            # Real-time temperature updates (only update probabilities, don't add tokens)
            temp_slider.change(
                fn=self.update_probabilities_only,
                inputs=[temp_slider],
                outputs=[probability_display, tokens_json, status_text]
            )
            
            # Event handler for token analysis
            analyze_btn.click(
                fn=self.analyze_token_frequencies,
                inputs=[],
                outputs=[top_tokens_display, bottom_tokens_display]
            )
        
        return interface
    
    def launch(self, 
               model_path: str = None, 
               vocab_path: str = None,
               share: bool = False,
               server_port: int = 7861):
        """
        Launch the Gradio interface.
        
        Args:
            model_path: Path to model checkpoint
            vocab_path: Path to vocabulary file
            share: Whether to create a public link
            server_port: Port to run the server on
        """
        if model_path and vocab_path:
            self.load_model(model_path, vocab_path)
        
        interface = self.create_interface()
        interface.launch(share=share, server_port=server_port)


def main():
    """Main function to run the Gradio app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Transformer Language Model Gradio App")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, help="Path to vocabulary file")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7861, help="Server port")
    
    args = parser.parse_args()
    
    app = GradioApp()
    app.launch(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        share=args.share,
        server_port=args.port
    )


if __name__ == "__main__":
    main()
