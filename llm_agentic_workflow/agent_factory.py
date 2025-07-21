from typing import Dict, Any, List, Optional
from langchain.agents import create_react_agent, AgentExecutor
from langchain.schema import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from config_parser import AgentConfig
from llm_factory import LLMFactory
from tool_factory import ToolFactory

class AgentFactory:
    """Factory for creating agent instances based on configuration"""
    
    def __init__(self, llm_factory: LLMFactory, tool_factory: ToolFactory):
        self.llm_factory = llm_factory
        self.tool_factory = tool_factory
        self._agent_cache: Dict[str, AgentExecutor] = {}
    
    def create_agent(self, config: AgentConfig, llm_config: Any) -> AgentExecutor:
        """Create an agent instance based on configuration"""
        
        # Check cache first
        cache_key = f"{config.type}_{config.llm_config}_{','.join(config.tools)}"
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]
        
        # Create LLM
        llm = self.llm_factory.create_llm(llm_config)
        
        # Create tools
        tools = []
        for tool_name in config.tools:
            tool_config = self.tool_factory.get_tool_config(tool_name)
            tool = self.tool_factory.create_tool(tool_config)
            tools.append(tool)
        
        # Create agent based on type
        if config.type == "react_agent":
            agent = self._create_react_agent(llm, tools, config.system_prompt)
        else:
            raise ValueError(f"Unsupported agent type: {config.type}")
        
        # Cache the agent instance
        self._agent_cache[cache_key] = agent
        return agent
    
    def _create_react_agent(self, llm: BaseLanguageModel, tools: List[BaseTool], system_prompt: str) -> AgentExecutor:
        """Create a ReAct agent"""
        
        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            template=f"{system_prompt}\n\n{{input}}\n\n{{agent_scratchpad}}"
        )
        
        # Create the agent
        agent = create_react_agent(llm, tools, prompt_template)
        
        # Create the executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
        
        return agent_executor
    
    def clear_cache(self):
        """Clear the agent cache"""
        self._agent_cache.clear() 