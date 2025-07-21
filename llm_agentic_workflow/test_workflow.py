#!/usr/bin/env python3
"""
Test script for the LangGraph Agentic Workflow System
"""

import asyncio
import json
from workflow_engine import WorkflowEngine

def create_simple_config():
    """Create a simple test configuration"""
    config_content = """
workflow:
  name: "Simple Test Workflow"
  description: "A simple test workflow with two agents"
  
  settings:
    max_iterations: 5
    timeout: 60
    
  llm_configs:
    test_llm:
      provider: "openai"
      model: "gpt-3.5-turbo"
      method: "chat"
      temperature: 0.1
      max_tokens: 1000
      api_key_env: "OPENAI_API_KEY"
      
  tools:
    calculator:
      name: "calculator"
      description: "Perform mathematical calculations"
      input_schema:
        expression:
          type: "string"
          description: "Mathematical expression to evaluate"
      output_schema:
        result:
          type: "number"
          description: "Calculation result"
          
  agents:
    analyzer:
      type: "react_agent"
      llm_config: "test_llm"
      system_prompt: "You are a helpful assistant that analyzes input and provides insights."
      tools: ["calculator"]
      next: "summarizer"
      
    summarizer:
      type: "react_agent"
      llm_config: "test_llm"
      system_prompt: "You are a summarization expert. Create concise summaries of the provided content."
      tools: []
      next: null
"""
    
    with open("test_config.yaml", "w") as f:
        f.write(config_content)
    
    return "test_config.yaml"

async def test_workflow():
    """Test the workflow system"""
    
    # Create test configuration
    config_path = create_simple_config()
    
    # Initialize workflow engine
    engine = WorkflowEngine(config_path)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config = engine.load_config()
        print(f"✅ Loaded: {config.name}")
        
        # Build workflow
        print("🔨 Building workflow...")
        graph = engine.build_workflow()
        print("✅ Workflow built successfully!")
        
        # Get workflow info
        info = engine.get_workflow_info()
        print(f"📊 Workflow Info: {json.dumps(info, indent=2)}")
        
        # Test inputs
        test_inputs = [
            "What is 15 + 27?",
            "Analyze the number 42 and provide insights",
            "Calculate the area of a circle with radius 5"
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n🧪 Test {i}: {test_input}")
            print("-" * 50)
            
            result = await engine.run_workflow(test_input)
            
            if result["success"]:
                print(f"✅ Success! Iterations: {result['iterations']}")
                print(f"📝 Final output: {result['messages'][-1] if result['messages'] else 'No output'}")
                
                # Show agent contexts
                for agent, context in result['context'].items():
                    output = context.get('output', 'No output')
                    print(f"🤖 {agent}: {output[:100]}...")
            else:
                print(f"❌ Failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test file
        import os
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")

def test_sync_workflow():
    """Test synchronous workflow execution"""
    
    config_path = create_simple_config()
    engine = WorkflowEngine(config_path)
    
    try:
        config = engine.load_config()
        graph = engine.build_workflow()
        
        result = engine.run_workflow_sync("What is 10 * 5?")
        
        print("\n🔄 Synchronous Test Result:")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Iterations: {result['iterations']}")
            print(f"Output: {result['messages'][-1] if result['messages'] else 'No output'}")
        else:
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Sync test error: {e}")
    
    finally:
        import os
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")

if __name__ == "__main__":
    print("🧪 Testing LangGraph Agentic Workflow System")
    print("=" * 60)
    
    # Run async test
    asyncio.run(test_workflow())
    
    print("\n" + "=" * 60)
    print("🔄 Testing Synchronous Execution:")
    test_sync_workflow()
    
    print("\n✅ All tests completed!") 