"""Custom model classes for diffusion language models."""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration import LlamaForDiffusionConfig, MistralForDiffusionConfig


class DiffusionModelMixin:
    """Mixin class providing diffusion functionality to language models."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._special_token_ids = None
        
    def _cache_special_token_ids(self, tokenizer=None):
        """Cache special token IDs to avoid repeated tokenizer access."""
        if tokenizer is None:
            self._special_token_ids = set()
            return
            
        special_tokens = set()
        
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            special_tokens.add(tokenizer.bos_token_id)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            special_tokens.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            special_tokens.add(tokenizer.pad_token_id)
            
        self._special_token_ids = special_tokens
    
    def _forward_process(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        eps: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward noising process. A timestep is sampled along the process, and tokens are
        masked with probability determined by the configured noise schedule.

        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            labels: Labels for SFT training [batch_size, seq_len].
            eps: Small epsilon value for minimum masking probability.

        Returns:
            noisy_batch: Input with some tokens masked.
            masked_indices: Boolean mask indicating which tokens were masked.
            p_mask: Masking probabilities for each token [batch_size, seq_len].
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Sample random timesteps for each sample in batch
        t = torch.rand(batch_size, device=device)

        # Calculate masking probability with epsilon
        p_mask = (1 - eps) * t + eps  # [batch_size]
        p_mask = p_mask[:, None].repeat(1, seq_len)  # [batch_size, seq_len]

        # Don't mask padding tokens if attention_mask is provided
        if attention_mask is not None:
            valid_mask = attention_mask.bool()
            p_mask = p_mask * valid_mask.float()

        # Create mask to exclude special tokens
        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self._special_token_ids:
            for token_id in self._special_token_ids:
                special_token_mask |= input_ids == token_id

        # Create random mask based on p_mask
        masked_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
        masked_indices = masked_indices & ~special_token_mask
        if attention_mask is not None:
            masked_indices = masked_indices & attention_mask.bool()

        # For SFT data, only mask answer tokens
        if labels is not None:
            answer_mask = labels != -100
            masked_indices = masked_indices & answer_mask

        # Create masked input
        mask_token_id = self.config.mask_token_id
        noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

        return noisy_batch, masked_indices, p_mask

    def _create_bidirectional_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Create bidirectional attention mask to override default causal masking. Handles
        sample-packed sequences where different samples are identified by different
        attention mask values.

        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            bidirectional_mask: 4D attention mask [batch_size, 1, seq_len, seq_len].
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None or not self.config.sample_packing:
            return torch.ones(
                batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device
            )

        # Create attention mask by comparing sample IDs element-wise
        mask_i = attention_mask.unsqueeze(2)  # [batch_size, seq_len, 1]
        mask_j = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]

        # Tokens can attend to each other if they have the same non-zero sample ID
        bidirectional_mask = (mask_i == mask_j) & (mask_i > 0)

        # Add head dimension: [batch_size, 1, seq_len, seq_len]
        bidirectional_mask = bidirectional_mask.unsqueeze(1)

        return bidirectional_mask

    def _compute_diffusion_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        masked_indices: torch.Tensor | None = None,
        p_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute diffusion loss given logits and masking information.

        Args:
            input_ids: Ground truth token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            labels: Labels for SFT training [batch_size, seq_len].
            logits: Model logits [batch_size, seq_len, vocab_size].
            masked_indices: Boolean mask indicating which tokens were masked.
            p_mask: Masking probabilities for each token [batch_size, seq_len].

        Returns:
            loss: Cross-entropy loss.
        """
        if masked_indices.sum() > 0:
            valid_indices = torch.where(masked_indices)
            batch_indices, seq_indices = valid_indices

            masked_logits = logits[batch_indices, seq_indices]
            masked_targets = input_ids[batch_indices, seq_indices]
            masked_p_mask = p_mask[batch_indices, seq_indices]

            # Compute cross-entropy loss without reduction
            token_loss = F.cross_entropy(
                masked_logits.float(), masked_targets, reduction="none"
            )

            if self.config.importance_weighting:
                masked_p_mask = masked_p_mask.float()
                weighted_loss = token_loss / masked_p_mask
            else:
                weighted_loss = token_loss

            # Final loss: sum weighted losses, normalize
            if labels is not None:
                # For SFT data: normalize by answer length per sample
                answer_mask = labels != -100
                answer_lengths = answer_mask.sum(dim=1).float()  # [batch_size]

                # Get batch indices for masked tokens
                masked_batch_indices = batch_indices

                # Sum losses per sample and divide by answer length
                loss_per_sample = torch.zeros(
                    input_ids.shape[0], device=input_ids.device
                )
                for i in range(input_ids.shape[0]):
                    sample_mask = masked_batch_indices == i
                    if sample_mask.sum() > 0:
                        sample_loss = weighted_loss[sample_mask].sum()
                        loss_per_sample[i] = sample_loss / answer_lengths[i]

                loss = loss_per_sample.mean()
            else:
                # Original normalization for non-SFT data
                loss = weighted_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        return loss


class LlamaForDiffusionLM(DiffusionModelMixin, LlamaForCausalLM):
    """
    Llama model for diffusion language modeling.
    
    This model extends LlamaForCausalLM with diffusion training capabilities,
    including bidirectional attention and forward diffusion process.
    """
    
    config_class = LlamaForDiffusionConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize diffusion-specific attributes
        self._special_token_ids = None
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for special token handling."""
        self._cache_special_token_ids(tokenizer)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with diffusion training logic.
        
        During training, applies forward diffusion process and bidirectional attention.
        During inference, behaves like standard causal language model.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.training and input_ids is not None:
            # Apply diffusion process during training
            original_input_ids = input_ids.clone()
            
            # Apply forward process to get noisy input
            noisy_input_ids, masked_indices, p_mask = self._forward_process(
                input_ids, attention_mask, labels, self.config.eps
            )
            
            # Create bidirectional attention mask
            bidirectional_attention_mask = self._create_bidirectional_attention_mask(
                input_ids, attention_mask
            )
            
            # Forward pass with noisy input and bidirectional attention
            outputs = super().forward(
                input_ids=noisy_input_ids,
                attention_mask=bidirectional_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,  # Don't use standard loss computation
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
            
            # Compute diffusion loss
            loss = self._compute_diffusion_loss(
                original_input_ids,
                attention_mask,
                labels,
                outputs.logits,
                masked_indices,
                p_mask,
            )
            
            if return_dict:
                outputs.loss = loss
                return outputs
            else:
                return (loss,) + outputs[1:]
        else:
            # Standard forward pass for inference
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )


class MistralForDiffusionLM(DiffusionModelMixin, MistralForCausalLM):
    """
    Mistral model for diffusion language modeling.
    
    This model extends MistralForCausalLM with diffusion training capabilities,
    including bidirectional attention and forward diffusion process.
    """
    
    config_class = MistralForDiffusionConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize diffusion-specific attributes
        self._special_token_ids = None
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for special token handling."""
        self._cache_special_token_ids(tokenizer)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with diffusion training logic.
        
        During training, applies forward diffusion process and bidirectional attention.
        During inference, behaves like standard causal language model.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.training and input_ids is not None:
            # Apply diffusion process during training
            original_input_ids = input_ids.clone()
            
            # Apply forward process to get noisy input
            noisy_input_ids, masked_indices, p_mask = self._forward_process(
                input_ids, attention_mask, labels, self.config.eps
            )
            
            # Create bidirectional attention mask
            bidirectional_attention_mask = self._create_bidirectional_attention_mask(
                input_ids, attention_mask
            )
            
            # Forward pass with noisy input and bidirectional attention
            outputs = super().forward(
                input_ids=noisy_input_ids,
                attention_mask=bidirectional_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,  # Don't use standard loss computation
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
            
            # Compute diffusion loss
            loss = self._compute_diffusion_loss(
                original_input_ids,
                attention_mask,
                labels,
                outputs.logits,
                masked_indices,
                p_mask,
            )
            
            if return_dict:
                outputs.loss = loss
                return outputs
            else:
                return (loss,) + outputs[1:]
        else:
            # Standard forward pass for inference
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )