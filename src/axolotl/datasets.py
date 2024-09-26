"""Module containing Dataset functionality"""

import logging
import os
import re
import random, time
from typing import List, Optional

import torch
from datasets import Dataset, IterableDataset

from .prompt_tokenizers import PromptTokenizingStrategy

# We want this to be a wrapper for an existing dataset that we have loaded
# lets use the concept of middlewares to wrap each dataset, for example
# ConstantLengthDataset(ShuffledDataset([TokenizedPromptDataset(alpaca_dataset)]))
# let's check to ensure we don't truncate an item in the middle, we'll use
# the collators later on to pad the datasets

LOG = logging.getLogger("axolotl")

def colorize_special_tokens_text(decoded_text, special_tokens):
    color_map = {
        'bos_token': '\033[1;95m',  # Bold Magenta
        'eos_token': '\033[1;94m',  # Bold Blue
        'unk_token': '\033[1;91m',  # Bold Red
        'pad_token': '\033[1;92m',  # Bold Green
        'chat_placeholder_token': '\033[1;96m',  # Bold Cyan
    }
    
    additional_colors = ['\033[1;93m', '\033[1;90m', '\033[1;97m']  # Bold Yellow, Bold Dark Grey, Bold White
    additional_special_tokens = special_tokens.get('additional_special_tokens', [])
    
    for idx, token in enumerate(additional_special_tokens):
        color_map[f'additional_special_token_{idx}'] = additional_colors[idx % len(additional_colors)]

    end_color = '\033[0m'  # Reset to default color

    for token_type, token_value in special_tokens.items():
        color = color_map.get(token_type, '')
        if isinstance(token_value, list):
            for value in token_value:
                decoded_text = re.sub(re.escape(value), f"{color}{value}{end_color}", decoded_text)
        else:
            decoded_text = re.sub(re.escape(token_value), f"{color}{token_value}{end_color}", decoded_text)
    
    return decoded_text

def colorize_special_tokens(decoded_text, special_tokens):
    # Temporarily replace <bos> with <"bos"> to avoid HTML tag interpretation
    bos_token = special_tokens.get('bos_token', '<bos>')
    temp_bos_token = '<"bos">'
    decoded_text = decoded_text.replace(bos_token, temp_bos_token)

    color_map = {
        'bos_token': 'color: magenta; font-weight: bold;',  # Bold Magenta
        'eos_token': 'color: orange; font-weight: bold;',     # Bold Blue
        'unk_token': 'color: red; font-weight: bold;',      # Bold Red
        'pad_token': 'color: green; font-weight: bold;',    # Bold Green
        'chat_placeholder_token': 'color: cyan; font-weight: bold;',  # Bold Cyan
    }
    
    additional_colors = [
        'color: yellow; font-weight: bold;',   # Bold Yellow
        'color: darkgray; font-weight: bold;', # Bold Dark Grey
        'color: white; font-weight: bold;'     # Bold White
    ]
    additional_special_tokens = special_tokens.get('additional_special_tokens', [])
    
    for idx, token in enumerate(additional_special_tokens):
        color_map[f'additional_special_token_{idx}'] = additional_colors[idx % len(additional_colors)]

    for token_type, token_value in special_tokens.items():
        if token_type == 'bos_token':
            continue  # Skip <bos> since it was handled separately
        color = color_map.get(token_type, '')
        if isinstance(token_value, list):
            for value in token_value:
                decoded_text = re.sub(re.escape(value), f'<span style="{color}">{value}</span>', decoded_text)
        else:
            decoded_text = re.sub(re.escape(token_value), f'<span style="{color}">{token_value}</span>', decoded_text)
    
    # Replace the temporary <"bos"> back to <bos> with styling
    decoded_text = decoded_text.replace(temp_bos_token, f'<span style="{color_map["bos_token"]}">&lt;bos&gt;</span>')

    return decoded_text

class TokenizedPromptDataset(Dataset):
    """
    Dataset that returns tokenized prompts from a stream of text files.
        Args:
            prompt_tokenizer (PromptTokenizingStrategy): The prompt tokenizing method for processing the data.
            dataset (dataset.Dataset): Dataset with text files.
            process_count (int): Number of processes to use for tokenizing.
            keep_in_memory (bool): Whether to keep the tokenized dataset in memory.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        prompt_tokenizer: PromptTokenizingStrategy,
        dataset: Dataset,
        process_count: Optional[int] = None,
        keep_in_memory: Optional[bool] = False,
        **kwargs,
    ):
        self.prompt_tokenizer = prompt_tokenizer
        self.process_count = process_count
        self.keep_in_memory = keep_in_memory
        super().__init__(
            self.process(dataset).data,
            **kwargs,
        )

    def process(self, dataset):
        features = dataset.features.keys()
        num_proc = min(64, self.process_count if self.process_count else os.cpu_count())

        map_kwargs = {}
        if self.prompt_tokenizer.supports_batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = 100

        mapped = dataset.map(
            self.prompt_tokenizer.tokenize_prompt,
            num_proc=num_proc,
            remove_columns=features,
            keep_in_memory=self.keep_in_memory,
            desc="Tokenizing Prompts",
            **map_kwargs,
        )

        # Get special tokens and chat template
        special_tokens = self.prompt_tokenizer.tokenizer.special_tokens_map
        chat_template = self.prompt_tokenizer.tokenizer.chat_template
        
        special_tokens['chat_placeholder_token'] = '<|im_start|>' 

        # Debug: Print special tokens and the first mapped item
        print("Special Tokens:", special_tokens)
        print("First Mapped Item:", mapped[0])
        decoded_text = self.prompt_tokenizer.tokenizer.decode(mapped[0]["input_ids"])
        print("Decoded Text:", decoded_text)
        # Print the decoding with colored special tokens
        print(colorize_special_tokens_text(decoded_text, special_tokens))

        # Sample 10 random items
        random_indices = random.sample(range(len(mapped)), 10)
        samples = [mapped[idx] for idx in random_indices]

        # find main directory
        main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
        # find current time
        current_time = time.strftime("%Y%m%d-%H%M%S")

        # Create directory if it doesn't exist
        if not os.path.exists(main_dir + '/axolotl/last_run_prepared'):
            os.makedirs(main_dir + '/axolotl/last_run_prepared')
        # Colorize and save to an HTML file
        with open(main_dir + '/axolotl/last_run_prepared/colorized_samples' + current_time + '.html', 'w') as f:
            f.write('<html><body><pre>\n')
            for sample in samples:
                decoded_text = self.prompt_tokenizer.tokenizer.decode(sample["input_ids"])
                colorized_text = colorize_special_tokens(decoded_text, special_tokens)
                f.write(colorized_text + '\n\n\n')
            f.write('</pre></body></html>')

        return mapped


# TODO this isn't the best since it can't interleave datasets
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for processing the data.
            dataset (dataset.Dataset): Dataset with text files.
            seq_length (int): Length of token sequences to return.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        tokenizer,
        datasets,
        seq_length=2048,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.datasets: List[IterableDataset] = datasets
        self.seq_length = seq_length

        vocab_size = len(tokenizer.get_vocab())

        if vocab_size <= torch.iinfo(torch.int16).max:
            self.tokens_dtype = torch.int16
        elif vocab_size <= torch.iinfo(torch.int32).max:
            self.tokens_dtype = torch.int32
        else:
            self.tokens_dtype = torch.int64

    def __iter__(self):
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "position_ids": [],
        }
        buffer_len = 0
        for dataset in self.datasets:
            idx = 0
            iterator = iter(dataset)
            more_examples = True
            while more_examples:
                try:
                    example = next(iterator)
                    idx += 1
                except StopIteration:
                    more_examples = False
                    example = None

                add_concat_token = False
                if example:
                    example_len = len(example["input_ids"])
                    add_concat_token = example["input_ids"][-1] != self.concat_token_id
                else:
                    example_len = 0

                if not example_len or (
                    buffer_len + int(add_concat_token) + example_len > self.seq_length
                ):
                    if buffer["input_ids"]:
                        input_ids = torch.cat(buffer["input_ids"], dim=-1)[
                            : self.seq_length
                        ]
                        attention_mask = torch.cat(buffer["attention_mask"], dim=-1)[
                            : self.seq_length
                        ]
                        position_ids = torch.cat(buffer["position_ids"], dim=-1)[
                            : self.seq_length
                        ]
                        labels = torch.cat(buffer["labels"], dim=-1)[: self.seq_length]
                        if labels.size() == input_ids.size() and (
                            attention_mask.size() == input_ids.size()
                        ):
                            yield {
                                "input_ids": input_ids,
                                "labels": labels,
                                "attention_mask": attention_mask,
                                "position_ids": position_ids,
                            }
                        else:
                            LOG.warning(
                                f"dropping batch due to tensor size mismatch input_ids: {input_ids.size()}, labels: {labels.size()}, attention_mask: {attention_mask.size()}"
                            )
                    buffer = {
                        "input_ids": [],
                        "attention_mask": [],
                        "labels": [],
                        "position_ids": [],
                    }
                    buffer_len = 0
                    idx = 1

                if example:
                    # FIXME
                    # just going to drop data points that are too long
                    if len(example["input_ids"]) <= self.seq_length:
                        input_ids = example["input_ids"]
                        attention_mask = example["attention_mask"]
                        labels = example["labels"]

                        if add_concat_token:
                            input_ids.append(self.concat_token_id)
                            attention_mask.append(1)
                            labels.append(self.concat_token_id)

                        input_ids_with_concat = torch.tensor(
                            input_ids, dtype=self.tokens_dtype
                        )
                        attention_mask_with_concat = torch.tensor(
                            [idx * m for m in attention_mask], dtype=torch.int16
                        )
                        labels_with_concat = torch.tensor(
                            labels, dtype=self.tokens_dtype
                        )
                        position_ids = torch.arange(
                            len(input_ids), dtype=self.tokens_dtype
                        )

                        buffer["input_ids"].append(input_ids_with_concat)
                        buffer["attention_mask"].append(attention_mask_with_concat)
                        buffer["labels"].append(labels_with_concat)
                        buffer["position_ids"].append(position_ids)
                        buffer_len += len(input_ids)
