import yaml
import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    provider: str
    model: str
    method: str  # "chat" or "generate"
    temperature: float = 0.1
    max_tokens: int = 4000
    api_key_env: Optional[str] = None
    region: Optional[str] = None

class ToolSchema(BaseModel):
    type: str
    description: str

class ToolConfig(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, ToolSchema]
    output_schema: Dict[str, ToolSchema]

class AgentConfig(BaseModel):
    type: str
    llm_config: str
    system_prompt: str
    tools: List[str] = []
    next: Optional[str] = None

class ConditionConfig(BaseModel):
    type: str
    function: str
    parameters: Dict[str, Any]

class ConditionalEdgeConfig(BaseModel):
    from_agent: str = Field(alias="from")
    to_agent: str = Field(alias="to")
    condition: ConditionConfig

class WorkflowConfig(BaseModel):
    name: str
    description: str
    settings: Dict[str, Any]
    llm_configs: Dict[str, LLMConfig]
    tools: Dict[str, ToolConfig]
    agents: Dict[str, AgentConfig]
    conditional_edges: Optional[Dict[str, ConditionalEdgeConfig]] = None
    custom_functions: Optional[Dict[str, Dict[str, Any]]] = None

class ConfigParser:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        
    def load_config(self) -> WorkflowConfig:
        """Load and validate the YAML configuration"""
        with open(self.config_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            
        # Extract workflow config
        workflow_data = yaml_data.get('workflow', {})
        
        # Validate and create config objects
        config = WorkflowConfig(**workflow_data)
        self.config = config
        return config
    
    def get_llm_config(self, config_name: str) -> LLMConfig:
        """Get LLM configuration by name"""
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self.config.llm_configs[config_name]
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get agent configuration by name"""
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self.config.agents[agent_name]
    
    def get_tool_config(self, tool_name: str) -> ToolConfig:
        """Get tool configuration by name"""
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self.config.tools[tool_name]
    
    def get_conditional_edges(self) -> Dict[str, ConditionalEdgeConfig]:
        """Get all conditional edges"""
        if not self.config:
            raise ValueError("Config not loaded. Call load_config() first.")
        return self.config.conditional_edges or {}
    
    def validate_config(self) -> List[str]:
        """Validate the configuration and return any errors"""
        errors = []
        
        if not self.config:
            errors.append("Config not loaded")
            return errors
            
        # Validate that all referenced LLM configs exist
        for agent_name, agent_config in self.config.agents.items():
            if agent_config.llm_config not in self.config.llm_configs:
                errors.append(f"Agent '{agent_name}' references unknown LLM config '{agent_config.llm_config}'")
        
        # Validate that all referenced tools exist
        for agent_name, agent_config in self.config.agents.items():
            for tool_name in agent_config.tools:
                if tool_name not in self.config.tools:
                    errors.append(f"Agent '{agent_name}' references unknown tool '{tool_name}'")
        
        # Validate that all referenced agents in 'next' exist
        for agent_name, agent_config in self.config.agents.items():
            if agent_config.next and agent_config.next not in self.config.agents:
                errors.append(f"Agent '{agent_name}' references unknown next agent '{agent_config.next}'")
        
        # Validate conditional edges
        for edge_name, edge_config in self.get_conditional_edges().items():
            if edge_config.from_agent not in self.config.agents:
                errors.append(f"Conditional edge '{edge_name}' references unknown from agent '{edge_config.from_agent}'")
            if edge_config.to_agent not in self.config.agents:
                errors.append(f"Conditional edge '{edge_name}' references unknown to agent '{edge_config.to_agent}'")
        
        return errors 