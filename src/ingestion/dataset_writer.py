"""
Dataset Writer Module
Handles saving Q/A pairs to JSONL format and dataset splitting.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def save_to_jsonl(data: List[Dict], filepath: str | Path) -> None:
    """
    Save data to JSONL format (one JSON object per line).
    
    Args:
        data: List of dictionaries to save
        filepath: Path to output file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


def split_dataset(
    all_pairs: List[Dict],
    train_split: float = 0.8,
    shuffle: bool = True,
    seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and test sets.
    
    Args:
        all_pairs: List of all Q/A pairs
        train_split: Proportion for training set (0.0 to 1.0)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_pairs, test_pairs)
    """
    if not all_pairs:
        return [], []
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        shuffled = all_pairs.copy()
        random.shuffle(shuffled)
    else:
        shuffled = all_pairs
    
    # Calculate split point
    total = len(shuffled)
    train_size = int(total * train_split)
    
    train_pairs = shuffled[:train_size]
    test_pairs = shuffled[train_size:]
    
    return train_pairs, test_pairs


def validate_qa_pair(pair: Dict) -> bool:
    """
    Validate that a Q/A pair has required fields and non-empty content.
    
    Args:
        pair: Q/A pair dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['question', 'answer', 'chunk_id']
    
    # Check required fields exist
    for field in required_fields:
        if field not in pair:
            return False
    
    # Check non-empty content
    if not pair['question'] or len(pair['question'].strip()) == 0:
        return False
    
    if not pair['answer'] or len(pair['answer'].strip()) == 0:
        return False
    
    return True


def filter_valid_pairs(pairs: List[Dict]) -> List[Dict]:
    """
    Filter out invalid Q/A pairs.
    
    Args:
        pairs: List of Q/A pair dictionaries
        
    Returns:
        List of valid pairs
    """
    return [pair for pair in pairs if validate_qa_pair(pair)]
