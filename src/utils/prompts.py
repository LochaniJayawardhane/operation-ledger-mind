"""
Prompt Templates for Q/A Generation
Contains prompt templates for question and answer generation.
"""

from typing import List


def get_question_generation_prompt(chunk: str, categories: List[str]) -> str:
    """
    Generate prompt for question generation.
    
    Args:
        chunk: Text chunk to generate questions from
        categories: List of categories to cover
        
    Returns:
        Formatted prompt string
    """
    categories_str = ", ".join(categories)
    
    prompt = f"""You are an expert at generating high-quality questions from financial documents.

Given the following text chunk from an annual report, generate exactly 10 diverse questions that cover these categories: {categories_str}

Categories breakdown:
- hard_facts: Questions about specific numbers, dates, names, financial metrics, concrete facts
- strategic_summary: Questions about business strategy, goals, plans, market positioning, competitive advantages
- stylistic_creative: Questions about tone, writing style, narrative approach, communication strategy

Text chunk:
{chunk}

Instructions:
1. Generate exactly 10 questions
2. Ensure questions are answerable from the provided chunk
3. Distribute questions across all three categories
4. Make questions specific and clear
5. Avoid yes/no questions unless necessary
6. Format your response as a JSON array of strings, one question per string

Example format:
["Question 1?", "Question 2?", ...]

Return only the JSON array, no additional text."""
    
    return prompt


def get_answer_generation_prompt(chunk: str, question: str) -> str:
    """
    Generate prompt for answer generation.
    
    Args:
        chunk: Text chunk containing the context
        question: Question to answer
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert financial analyst. Answer the following question based strictly on the provided text chunk from an annual report.

Text chunk:
{chunk}

Question:
{question}

Instructions:
1. Answer the question based ONLY on the information in the text chunk above
2. If the answer is not in the chunk, state "The information is not available in the provided text"
3. Be precise and factual
4. Include relevant details from the chunk
5. Keep your answer concise but complete
6. Do not make assumptions or add information not in the chunk

Answer:"""
    
    return prompt


def get_batch_answer_generation_prompt(chunk: str, questions: List[str]) -> str:
    """
    Generate prompt for batch answer generation (all questions in one call).
    
    Args:
        chunk: Text chunk containing the context
        questions: List of questions to answer
        
    Returns:
        Formatted prompt string
    """
    questions_list = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    prompt = f"""You are an expert financial analyst. Answer the following questions based strictly on the provided text chunk from an annual report.

Text chunk:
{chunk}

Questions:
{questions_list}

Instructions:
1. Answer each question based ONLY on the information in the text chunk above
2. If the answer is not in the chunk, state "The information is not available in the provided text"
3. Be precise and factual
4. Include relevant details from the chunk
5. Keep your answers concise but complete
6. Do not make assumptions or add information not in the chunk
7. Format your response as a JSON array of strings, one answer per question in the same order
8. Each answer should correspond to the question at the same index position

Example format:
["Answer to question 1", "Answer to question 2", ...]

Return only the JSON array, no additional text."""
    
    return prompt