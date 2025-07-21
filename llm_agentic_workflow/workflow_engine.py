from typing import Dict, Any, List, Optional, Callable
from langgraph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor
import asyncio
import time

from config_parser import ConfigParser, WorkflowConfig, AgentConfig, ConditionalEdgeConfig
from llm_factory import LLMFactory
from tool_factory import ToolFactory
from agent_factory import AgentFactory
from condition_evaluator import ConditionEvaluator

class WorkflowState:
    """State object for the workflow"""
    def __init__(self):
        self.messages: List[BaseMessage] = []
        self.context: Dict[str, Any] = {}
        self.current_agent: Optional[str] = None
        self.iteration_count: int = 0
        self.max_iterations: int = 10
        self.start_time: float = time.time()
        self.timeout: int = 300  # 5 minutes default

class WorkflowEngine:
    """Main workflow engine that orchestrates the multi-agent workflow"""
    
    def __init__(self, config_path: str):
        self.config_parser = ConfigParser(config_path)
        self.config = None
        self.llm_factory = LLMFactory()
        self.tool_factory = ToolFactory()
        self.agent_factory = AgentFactory(self.llm_factory, self.tool_factory)
        self.condition_evaluator = ConditionEvaluator()
        self.graph = None
        
    def load_config(self) -> WorkflowConfig:
        """Load and validate the workflow configuration"""
        self.config = self.config_parser.load_config()
        
        # Validate configuration
        errors = self.config_parser.validate_config()
        if errors:
            raise ValueError(f"Configuration errors: {errors}")
        
        return self.config
    
    def register_custom_tool(self, name: str, func: Callable):
        """Register a custom tool"""
        self.tool_factory.register_custom_tool(name, func)
    
    def register_custom_condition(self, name: str, func: Callable):
        """Register a custom condition function"""
        self.condition_evaluator.register_custom_function(name, func)
    
    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        if not self.config:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        for agent_name, agent_config in self.config.agents.items():
            workflow.add_node(agent_name, self._create_agent_node(agent_name, agent_config))
        
        # Add conditional edges
        for edge_name, edge_config in self.config_parser.get_conditional_edges().items():
            workflow.add_conditional_edges(
                edge_config.from_agent,
                self._create_conditional_edge(edge_config),
                {
                    edge_config.to_agent: lambda x: True,
                    END: lambda x: False
                }
            )
        
        # Add regular edges based on 'next' configuration
        for agent_name, agent_config in self.config.agents.items():
            if agent_config.next:
                workflow.add_edge(agent_name, agent_config.next)
            elif agent_name not in [edge.from_agent for edge in self.config_parser.get_conditional_edges().values()]:
                # If no conditional edge and no next, add to END
                workflow.add_edge(agent_name, END)
        
        # Set entry point
        entry_point = list(self.config.agents.keys())[0]
        workflow.set_entry_point(entry_point)
        
        self.graph = workflow.compile()
        return self.graph
    
    def _create_agent_node(self, agent_name: str, agent_config: AgentConfig):
        """Create a node function for an agent"""
        def agent_node(state: WorkflowState) -> WorkflowState:
            """Execute an agent and update state"""
            
            # Check iteration limit
            if state.iteration_count >= state.max_iterations:
                state.messages.append(AIMessage(content="Maximum iterations reached"))
                return state
            
            # Check timeout
            if time.time() - state.start_time > state.timeout:
                state.messages.append(AIMessage(content="Workflow timeout reached"))
                return state
            
            # Update state
            state.current_agent = agent_name
            state.iteration_count += 1
            
            try:
                # Create agent
                llm_config = self.config_parser.get_llm_config(agent_config.llm_config)
                agent = self.agent_factory.create_agent(agent_config, llm_config)
                
                # Prepare input
                if state.messages:
                    input_text = state.messages[-1].content
                else:
                    input_text = "Start the workflow"
                
                # Execute agent
                result = agent.invoke({"input": input_text})
                
                # Update state
                state.messages.append(AIMessage(content=result["output"]))
                state.context[agent_name] = {
                    "output": result["output"],
                    "intermediate_steps": result.get("intermediate_steps", [])
                }
                
            except Exception as e:
                error_msg = f"Error in agent '{agent_name}': {str(e)}"
                state.messages.append(AIMessage(content=error_msg))
                state.context[agent_name] = {"error": error_msg}
            
            return state
        
        return agent_node
    
    def _create_conditional_edge(self, edge_config: ConditionalEdgeConfig):
        """Create a conditional edge function"""
        def conditional_edge(state: WorkflowState) -> bool:
            """Evaluate condition for conditional edge"""
            
            try:
                # Get context for the from agent
                from_agent_context = state.context.get(edge_config.from_agent, {})
                
                # Evaluate condition
                result = self.condition_evaluator.evaluate_condition(
                    edge_config.condition, 
                    from_agent_context
                )
                
                return result
                
            except Exception as e:
                print(f"Error evaluating condition: {e}")
                return False
        
        return conditional_edge
    
    async def run_workflow(self, initial_input: str = "Start the workflow") -> Dict[str, Any]:
        """Run the workflow asynchronously"""
        if not self.graph:
            raise ValueError("Workflow not built. Call build_workflow() first.")
        
        # Create initial state
        initial_state = WorkflowState()
        initial_state.messages.append(HumanMessage(content=initial_input))
        initial_state.max_iterations = self.config.settings.get("max_iterations", 10)
        initial_state.timeout = self.config.settings.get("timeout", 300)
        
        # Run the workflow
        try:
            result = await self.graph.ainvoke(initial_state)
            return {
                "success": True,
                "final_state": result,
                "messages": [msg.content for msg in result.messages],
                "context": result.context,
                "iterations": result.iteration_count
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "iterations": getattr(result, 'iteration_count', 0) if 'result' in locals() else 0
            }
    
    def run_workflow_sync(self, initial_input: str = "Start the workflow") -> Dict[str, Any]:
        """Run the workflow synchronously"""
        return asyncio.run(self.run_workflow(initial_input))
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow"""
        if not self.config:
            return {"error": "Configuration not loaded"}
        
        return {
            "name": self.config.name,
            "description": self.config.description,
            "agents": list(self.config.agents.keys()),
            "tools": list(self.config.tools.keys()),
            "llm_configs": list(self.config.llm_configs.keys()),
            "conditional_edges": list(self.config_parser.get_conditional_edges().keys()) if self.config_parser.get_conditional_edges() else []
        } 