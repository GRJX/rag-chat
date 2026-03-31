import re
import ollama
from typing import Dict, List, Generator, Any, Optional, Union

from src.config import (LLM_MODEL_NAME, LLM_MAX_TOKENS, LLM_TEMPERATURE, 
                        LLM_TOP_P, LLM_TOP_K, LLM_PRESENCE_PENALTY, 
                        LLM_FREQUENCY_PENALTY, LLM_SYSTEM_PROMPT, LLM_SEED,
                        NO_ANSWER_RESPONSE)


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
        self.seed = LLM_SEED
        self.no_answer_response = NO_ANSWER_RESPONSE
        
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

        # Sources first — ground the model before anything else
        if context_chunks:
            prompt_elements.append("SOURCES:")
            for i, chunk in enumerate(context_chunks, 1):
                file_name = chunk['file_path'].split('/')[-1]
                label = f"[{i}] {file_name}, lines {chunk['start_line']}-{chunk['end_line']}"
                prompt_elements.append(f"{label}\n{chunk['content']}")
            prompt_elements.append(
                "STRICT INSTRUCTIONS:\n"
                "- Answer using ONLY the SOURCES above. Do NOT use any outside knowledge.\n"
                "- Quote or closely paraphrase the relevant text from the sources.\n"
                "- Cite each claim with the source number, e.g. [1].\n"
                "- Every factual statement in your answer MUST have a citation.\n"
                "- If the sources do not contain the answer, respond ONLY with: "
                f"'{self.no_answer_response}'\n"
                "- Do NOT speculate, guess, or infer beyond what is explicitly stated.\n"
                "- Do NOT make up facts, dates, numbers, or names."
            )
        else:
            prompt_elements.append(
                f"No source excerpts were retrieved for this question. "
                f"Respond ONLY with: '{self.no_answer_response}'"
            )

        # Chat history for follow-up context
        if chat_history:
            prompt_elements.append("CONVERSATION SO FAR:")
            for message in chat_history:
                role_display = "User" if message['role'] == 'user' else "Assistant"
                prompt_elements.append(f"{role_display}: {message['content']}")

        prompt_elements.append(f"QUESTION: {query}")

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
                "seed": self.seed,
                "reasoning_effort": "high",
            }
        }
        
        if stream:
            return (chunk['response'] for chunk in ollama.generate(**params, stream=True))
        else:
            response = ollama.generate(**params)
            return response['response']

    def validate_response(self, response: str, num_sources: int) -> str:
        """
        Validate a generated response for hallucination signals.
        If sources were provided but no citations appear in the response,
        append a disclaimer warning. If no sources were available,
        force the no-answer response.
        
        Args:
            response: The generated response text.
            num_sources: Number of source chunks that were provided.
            
        Returns:
            The original or corrected response.
        """
        # If no sources were provided, force the standard refusal
        if num_sources == 0:
            return self.no_answer_response

        # Check if the response already is the no-answer response
        if self.no_answer_response.lower() in response.lower():
            return response

        # Verify citations are present when sources were provided
        citation_pattern = re.compile(r'\[\d+\]')
        has_citations = bool(citation_pattern.search(response))

        if not has_citations:
            return (
                f"{response}\n\n"
                f"**Note:** This response could not be verified against the provided sources. "
                f"Please review the source documents directly."
            )

        # Verify cited source numbers are within range
        cited_nums = {int(n) for n in re.findall(r'\[(\d+)\]', response)}
        invalid_refs = {n for n in cited_nums if n < 1 or n > num_sources}
        if invalid_refs:
            return (
                f"{response}\n\n"
                f"**Note:** This response references sources {invalid_refs} which were not provided. "
                f"Please verify the answer against the actual source documents."
            )

        return response
