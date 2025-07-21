#!/usr/bin/env python3
"""
LangGraph-based Agentic Workflow System

This system allows you to create multi-agent workflows using YAML configuration.
Each agent can use different LLMs, tools, and be connected with conditional logic.
"""

import asyncio
import json
from typing import Dict, Any
from workflow_engine import WorkflowEngine

def needs_revision(context: Dict[str, Any], max_iterations: int = 3) -> bool:
    """Custom function to determine if a report needs revision"""
    # Check if we've exceeded max iterations
    if context.get("iteration_count", 0) >= max_iterations:
        return False
    
    # Check if the output contains revision indicators
    output = context.get("output", "").lower()
    revision_indicators = ["revise", "revision", "improve", "enhance", "better"]
    
    for indicator in revision_indicators:
        if indicator in output:
            return True
    
    # Check confidence score if available
    confidence = context.get("confidence", 1.0)
    if confidence < 0.7:
        return True
    
    return False

def custom_quality_check(context: Dict[str, Any], min_score: float = 0.8) -> bool:
    """Custom function to check quality of output"""
    # Extract quality metrics from context
    output = context.get("output", "")
    
    # Simple quality heuristics
    quality_score = 0.0
    
    # Check for completeness (has content)
    if len(output.strip()) > 50:
        quality_score += 0.3
    
    # Check for structure (has sections/points)
    if any(char in output for char in ["1.", "2.", "3.", "-", "*"]):
        quality_score += 0.3
    
    # Check for professional language
    professional_words = ["analysis", "conclusion", "recommendation", "summary"]
    if any(word in output.lower() for word in professional_words):
        quality_score += 0.4
    
    return quality_score >= min_score

async def main():
    """Main function demonstrating the workflow system"""
    
    # Initialize the workflow engine
    engine = WorkflowEngine("config_example.yaml")
    
    # Register custom functions
    engine.register_custom_condition("needs_revision", needs_revision)
    engine.register_custom_condition("custom_quality_check", custom_quality_check)
    
    # Register custom tools
    def custom_data_analyzer(data: str) -> str:
        """Custom tool for data analysis"""
        return f"Analysis of data: {data[:100]}... (analyzed)"
    
    engine.register_custom_tool("data_analyzer", custom_data_analyzer)
    
    try:
        # Load configuration
        print("Loading workflow configuration...")
        config = engine.load_config()
        print(f"Loaded workflow: {config.name}")
        print(f"Description: {config.description}")
        print(f"Agents: {list(config.agents.keys())}")
        print(f"Tools: {list(config.tools.keys())}")
        
        # Build workflow
        print("\nBuilding workflow...")
        graph = engine.build_workflow()
        print("Workflow built successfully!")
        
        # Get workflow info
        info = engine.get_workflow_info()
        print(f"\nWorkflow Info: {json.dumps(info, indent=2)}")
        
        # Run workflow
        print("\nRunning workflow...")
        result = await engine.run_workflow("Analyze the document 'sample.txt' and generate a comprehensive report")
        
        if result["success"]:
            print("✅ Workflow completed successfully!")
            print(f"Iterations: {result['iterations']}")
            print(f"Messages: {len(result['messages'])}")
            
            print("\n📋 Final Messages:")
            for i, msg in enumerate(result['messages'], 1):
                print(f"{i}. {msg[:100]}...")
            
            print("\n🔧 Agent Contexts:")
            for agent, context in result['context'].items():
                print(f"{agent}: {context.get('output', 'No output')[:100]}...")
        else:
            print(f"❌ Workflow failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_simple_example():
    """Run a simple example without async"""
    engine = WorkflowEngine("config_example.yaml")
    
    try:
        # Load and build
        config = engine.load_config()
        graph = engine.build_workflow()
        
        # Run synchronously
        result = engine.run_workflow_sync("Process this document and create a summary")
        
        print("Simple Example Result:")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Iterations: {result['iterations']}")
            print(f"Final message: {result['messages'][-1] if result['messages'] else 'No messages'}")
        else:
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Error in simple example: {e}")

if __name__ == "__main__":
    print("🚀 LangGraph Agentic Workflow System")
    print("=" * 50)
    
    # Run the main async example
    asyncio.run(main())
    
    print("\n" + "=" * 50)
    print("Simple Example:")
    run_simple_example() 