import ollama
from typing import Dict, List, Generator, Any, Optional, Union

from src.config import (LLM_MODEL_NAME, LLM_MAX_TOKENS, LLM_TEMPERATURE, 
                        LLM_TOP_P, LLM_TOP_K, LLM_PRESENCE_PENALTY, 
                        LLM_FREQUENCY_PENALTY, LLM_SYSTEM_PROMPT)


class OllamaGenerator:
    def __init__(self):
        self.model_name = LLM_MODEL_NAME
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.top_p = LLM_TOP_P
        self.top_k = LLM_TOP_K
        self.presence_penalty = LLM_PRESENCE_PENALTY
        self.frequency_penalty = LLM_FREQUENCY_PENALTY
        self.system_prompt = LLM_SYSTEM_PROMPT
        
    def construct_prompt(self, query: str, context_chunks: List[Dict[str, Any]], 
                         chat_history: Optional[List[Dict[str, str]]] = None, 
                         initial_topic: Optional[str] = None) -> str:
        """
        Construct a prompt using the query, context chunks, and chat history,
        while maintaining focus on the initial topic.
        
        Args:
            query: The user's current question.
            context_chunks: The retrieved context chunks for the current query.
            chat_history: A list of previous user/assistant messages.
                          Each item is a dict with "role" ('user' or 'assistant') and "content".
            initial_topic: The initial query/topic to maintain focus on.
            
        Returns:
            A formatted prompt string.
        """
        prompt_elements = []
        
        # Add a topic reminder if we have an initial topic
        if initial_topic:
            prompt_elements.append(f"Initial Question/Topic: {initial_topic}\n")
            prompt_elements.append("Please keep your responses focused on this initial topic and the codebase context below.\n")
        
        # Add chat history
        if chat_history:
            prompt_elements.append("Previous conversation:")
            for message in chat_history:
                role_display = "User" if message['role'] == 'user' else "Assistant"
                prompt_elements.append(f"{role_display}: {message['content']}")
        
        # Current query and its context
        prompt_elements.append(f"Current Question: {query}")
        
        if context_chunks:
            prompt_elements.append("Context (you MUST cite sources using square brackets only, e.g. [Source 1], never round brackets):")
            for i, chunk in enumerate(context_chunks, 1):
                file_name = chunk['file_path'].split('/')[-1]
                label = f"[Source {i}: {file_name}, lines {chunk['start_line']}-{chunk['end_line']}]"
                prompt_elements.append(f"{label}\n{chunk['content']}")
        
        return "\n\n".join(prompt_elements)
    
    def generate(self, prompt: str, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using the Ollama model.
        
        Args:
            prompt: The prompt to send to the model (query + context).
            stream: If True, stream the response; otherwise, return the full response.
            
        Returns:
            Either the complete generated response (str) or a generator that yields
            chunks of the response (Generator).
        """
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "system": self.system_prompt,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "reasoning_effort": "high",
            }
        }
        
        if stream:
            return (chunk['response'] for chunk in ollama.generate(**params, stream=True))
        else:
            response = ollama.generate(**params)
            return response['response']
