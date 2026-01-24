"""
Q/A Generation Orchestrator
Coordinates question and answer generation for chunks.
"""

from typing import List, Dict
from ..utils.llm_client import LLMClient


def generate_qa_pairs(
    chunk: Dict[str, any],
    question_llm_config: dict,
    answer_llm_config: dict,
    categories: List[str]
) -> List[Dict[str, any]]:
    """
    Generate Q/A pairs for a single chunk.
    
    Args:
        chunk: Chunk dictionary with 'chunk_id' and 'text'
        question_llm_config: Configuration for question generation LLM
        answer_llm_config: Configuration for answer generation LLM
        categories: List of categories to cover
        
    Returns:
        List of Q/A pair dictionaries, each containing:
        - 'question': Generated question
        - 'answer': Generated answer
        - 'chunk_id': ID of the source chunk
        - 'category': Inferred category (hard_facts, strategic_summary, or stylistic_creative)
    """
    chunk_text = chunk['text']
    chunk_id = chunk['chunk_id']
    
    # Initialize LLM clients
    question_llm = LLMClient(question_llm_config)
    answer_llm = LLMClient(answer_llm_config)
    
    # Step A: Generate questions
    try:
        questions = question_llm.generate_questions(chunk_text, categories)
    except Exception as e:
        raise RuntimeError(f"Failed to generate questions for chunk {chunk_id}: {str(e)}")
    
    # Ensure we have exactly 10 questions (pad or truncate if needed)
    if len(questions) < 10:
        # If we got fewer questions, we could repeat some or generate more
        pass
    elif len(questions) > 10:
        questions = questions[:10]
    
    # Step B: Generate answers for each question
    qa_pairs = []
    for i, question in enumerate(questions):
        if not question or len(question.strip()) == 0:
            continue
        
        try:
            answer = answer_llm.generate_answer(chunk_text, question)
            
            # Infer category from question (simple heuristic)
            category = _infer_category(question, categories)
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'chunk_id': chunk_id,
                'category': category
            })
        except Exception as e:
            # Log error but continue with other questions
            print(f"Warning: Failed to generate answer for question {i+1} in chunk {chunk_id}: {str(e)}")
            continue
    
    return qa_pairs


def _infer_category(question: str, categories: List[str]) -> str:
    """
    Infer the category of a question based on keywords.
    
    Args:
        question: Question text
        categories: Available categories
        
    Returns:
        Category string
    """
    question_lower = question.lower()
    
    # Hard facts keywords
    hard_facts_keywords = [
        'how many', 'how much', 'what is', 'what was', 'when', 'where',
        'who', 'which', 'number', 'percentage', 'revenue', 'profit',
        'loss', 'earnings', 'cost', 'price', 'date', 'year', 'quarter'
    ]
    
    # Strategic summary keywords
    strategic_keywords = [
        'strategy', 'goal', 'plan', 'objective', 'vision', 'mission',
        'competitive', 'advantage', 'market', 'position', 'approach',
        'focus', 'priority', 'initiative', 'direction'
    ]
    
    # Stylistic/creative keywords
    stylistic_keywords = [
        'tone', 'style', 'narrative', 'communication', 'message',
        'voice', 'perspective', 'approach to', 'how does the report',
        'language', 'writing'
    ]
    
    # Count matches
    hard_facts_count = sum(1 for kw in hard_facts_keywords if kw in question_lower)
    strategic_count = sum(1 for kw in strategic_keywords if kw in question_lower)
    stylistic_count = sum(1 for kw in stylistic_keywords if kw in question_lower)
    
    # Return category with most matches, default to first category
    if hard_facts_count > strategic_count and hard_facts_count > stylistic_count:
        return 'hard_facts'
    elif strategic_count > stylistic_count:
        return 'strategic_summary'
    elif stylistic_count > 0:
        return 'stylistic_creative'
    else:
        # Default based on position in categories list
        return categories[0] if categories else 'hard_facts'
