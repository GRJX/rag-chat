import ollama
from typing import Dict, List, Generator, Any, Optional

from utils.config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS, LLM_SYSTEM_PROMPT
from typing import Union

class OllamaGenerator:
    def __init__(self):
        """
        Initialize the Ollama generator using environment configuration.
        """
        self.model_name = LLM_MODEL_NAME
        self.temperature = LLM_TEMPERATURE
        self.top_p = LLM_TOP_P
        self.max_tokens = LLM_MAX_TOKENS
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
            prompt_elements.append("Context:")
            for chunk in context_chunks:
                prompt_elements.append(chunk['content'])
        
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
                "top_p": self.top_p
            }
        }
        
        if stream:
            return (chunk['response'] for chunk in ollama.generate(**params, stream=True))
        else:
            response = ollama.generate(**params)
            return response['response']
