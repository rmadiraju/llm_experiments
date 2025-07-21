import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Bedrock
from langchain.schema import BaseLanguageModel
from config_parser import LLMConfig

class LLMFactory:
    """Factory for creating LLM instances based on configuration"""
    
    def __init__(self):
        self._llm_cache: Dict[str, BaseLanguageModel] = {}
    
    def create_llm(self, config: LLMConfig) -> BaseLanguageModel:
        """Create an LLM instance based on configuration"""
        
        # Check cache first
        cache_key = f"{config.provider}_{config.model}_{config.temperature}_{config.max_tokens}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        llm = None
        
        if config.provider == "openai":
            llm = self._create_openai_llm(config)
        elif config.provider == "anthropic":
            llm = self._create_anthropic_llm(config)
        elif config.provider == "bedrock":
            llm = self._create_bedrock_llm(config)
        elif config.provider == "ollama":
            llm = self._create_ollama_llm(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        
        # Cache the LLM instance
        self._llm_cache[cache_key] = llm
        return llm
    
    def _create_openai_llm(self, config: LLMConfig) -> ChatOpenAI:
        """Create OpenAI LLM instance"""
        api_key = os.getenv(config.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {config.api_key_env or 'OPENAI_API_KEY'}")
        
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=api_key
        )
    
    def _create_anthropic_llm(self, config: LLMConfig) -> ChatAnthropic:
        """Create Anthropic LLM instance"""
        api_key = os.getenv(config.api_key_env or "ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(f"Anthropic API key not found in environment variable: {config.api_key_env or 'ANTHROPIC_API_KEY'}")
        
        return ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_key=api_key
        )
    
    def _create_bedrock_llm(self, config: LLMConfig) -> Bedrock:
        """Create AWS Bedrock LLM instance"""
        if not config.region:
            raise ValueError("Region is required for Bedrock LLM")
        
        return Bedrock(
            model_id=config.model,
            region_name=config.region,
            model_kwargs={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        )
    
    def _create_ollama_llm(self, config: LLMConfig):
        """Create Ollama LLM instance"""
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError("langchain-community is required for Ollama support")
        
        base_url = os.getenv(config.api_key_env or "OLLAMA_BASE_URL", "http://localhost:11434")
        
        return Ollama(
            model=config.model,
            base_url=base_url,
            temperature=config.temperature
        )
    
    def clear_cache(self):
        """Clear the LLM cache"""
        self._llm_cache.clear() 