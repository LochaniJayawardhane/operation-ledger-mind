"""
LLM Client Wrapper Module
Handles interactions with LLM providers (OpenAI, HuggingFace).
"""

import json
import time
from typing import List, Dict, Optional
from openai import OpenAI


class LLMClient:
    """Wrapper for LLM API calls with retry logic."""
    
    def __init__(self, config: dict):
        """
        Initialize LLM client.
        
        Args:
            config: Configuration dictionary containing provider settings
        """
        self.config = config
        self.provider = config.get('provider', 'openai')
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 300)
        self.timeout = config.get('timeout_seconds', 60)
        self.max_retries = config.get('max_retries', 3)
        
        # Initialize OpenAI client if using OpenAI
        if self.provider == 'openai':
            self.client = OpenAI(timeout=self.timeout)
        else:
            self.client = None
    
    def _call_openai(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Make API call to OpenAI.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Response text
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"OpenAI API call failed after {self.max_retries} attempts: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text
        """
        if self.provider == 'openai':
            return self._call_openai(prompt, temperature, max_tokens)
        else:
            raise NotImplementedError(f"Provider {self.provider} not yet implemented")
    
    def generate_questions(self, chunk: str, categories: List[str]) -> List[str]:
        """
        Generate questions from a text chunk.
        
        Args:
            chunk: Text chunk to generate questions from
            categories: List of categories to cover
            
        Returns:
            List of generated questions
        """
        from .prompts import get_question_generation_prompt
        
        prompt = get_question_generation_prompt(chunk, categories)
        response = self.generate(prompt)
        
        # Parse JSON response
        try:
            # Try to extract JSON array from response
            response = response.strip()
            # Remove markdown code blocks if present
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
                response = response.strip()
            
            questions = json.loads(response)
            if isinstance(questions, list):
                return [str(q).strip() for q in questions if q]
            else:
                raise ValueError("Response is not a list")
        except json.JSONDecodeError as e:
            # Fallback: try to extract questions from text
            # Look for lines that end with '?'
            questions = [line.strip() for line in response.split('\n') 
                        if line.strip().endswith('?')]
            if len(questions) >= 5:  # At least some questions found
                return questions[:10]  # Return up to 10
            else:
                raise ValueError(f"Failed to parse questions from response: {response[:200]}")
    
    def generate_answer(self, chunk: str, question: str) -> str:
        """
        Generate answer to a question based on chunk context.
        
        Args:
            chunk: Text chunk containing context
            question: Question to answer
            
        Returns:
            Generated answer
        """
        from .prompts import get_answer_generation_prompt
        
        prompt = get_answer_generation_prompt(chunk, question)
        answer = self.generate(prompt)
        return answer.strip()
