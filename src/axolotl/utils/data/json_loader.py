"""Custom JSON loader patch for Axolotl to handle mixed content types for Qwen3-VL."""
import json
from typing import Dict, Any, List, Optional
from datasets import Dataset
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def load_mixed_content_jsonl(file_path: str) -> Dataset:
    """
    Load JSONL file with mixed content types (string and array) in messages.
    
    This is specifically designed to handle Qwen3-VL datasets where:
    - System messages have string content
    - User messages may have array content (for multimodal)
    - Assistant messages have string content
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Dataset object with loaded data
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                
                # Normalize ALL content to be JSON strings
                # This ensures consistent types across all rows
                if 'messages' in item:
                    normalized_messages = []
                    for message in item['messages']:
                        # Preserve all fields from the original message
                        normalized_message = message.copy()
                        
                        # Only normalize the content field if it exists
                        if 'content' in message:
                            normalized_message['content'] = json.dumps(message['content']) if isinstance(message['content'], (list, dict)) else message['content']
                        
                        normalized_messages.append(normalized_message)
                    item['messages'] = normalized_messages
                
                data.append(item)
                
            except json.JSONDecodeError as e:
                LOG.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                continue
            except (KeyError, TypeError, ValueError) as e:
                LOG.warning(f"Error processing line {line_num}: {e}")
                continue
    
    LOG.info(f"Loaded {len(data)} examples from {file_path}")
    
    # Create dataset using from_list with normalized data
    # All content fields are now consistently strings
    return Dataset.from_list(data)


def is_mixed_content_dataset(dataset_config: Dict[str, Any]) -> bool:
    """
    Check if this is a dataset that requires mixed content handling.
    
    Args:
        dataset_config: Dataset configuration dictionary
        
    Returns:
        True if mixed content handling is needed
    """
    # Check for explicit flag
    if dataset_config.get("mixed_content_messages", False):
        return True
    
    # Check for qwen2_vl chat template which often has mixed content
    if dataset_config.get("chat_template") == "qwen2_vl":
        return True
    
    # Check if this is a multimodal dataset with chat_template type
    if (dataset_config.get("type") == "chat_template" and 
        dataset_config.get("field_images") is not None):
        return True
    
    return False