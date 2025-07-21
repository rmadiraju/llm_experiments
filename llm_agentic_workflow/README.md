# LangGraph Agentic Workflow System

A flexible, YAML-configurable multi-agent workflow system built on LangGraph that supports multiple LLM providers, custom tools, and conditional routing between agents.

## 🏗️ Architecture

The system consists of several key components:

### Core Components

1. **ConfigParser** (`config_parser.py`)
   - Loads and validates YAML configuration
   - Defines data models for LLM configs, agents, tools, and conditions
   - Provides validation and error checking

2. **LLMFactory** (`llm_factory.py`)
   - Creates LLM instances based on configuration
   - Supports OpenAI, Anthropic, and AWS Bedrock
   - Implements caching for performance

3. **ToolFactory** (`tool_factory.py`)
   - Creates tool instances based on configuration
   - Supports built-in tools (web search, file reader, calculator)
   - Allows registration of custom tools

4. **AgentFactory** (`agent_factory.py`)
   - Creates agent instances with specified LLMs and tools
   - Currently supports ReAct agents
   - Extensible for other agent types

5. **ConditionEvaluator** (`condition_evaluator.py`)
   - Evaluates conditions for conditional edges
   - Supports content-based, threshold-based, and custom function conditions
   - Allows registration of custom condition functions

6. **WorkflowEngine** (`workflow_engine.py`)
   - Main orchestrator that builds and runs LangGraph workflows
   - Manages state, iterations, and timeouts
   - Handles both synchronous and asynchronous execution

## 📋 Configuration

The system uses YAML configuration files with the following structure:

```yaml
workflow:
  name: "Multi-Agent Document Processor"
  description: "A workflow that processes documents using multiple specialized agents"
  
  settings:
    max_iterations: 10
    timeout: 300
    
  llm_configs:
    openai_gpt4:
      provider: "openai"
      model: "gpt-4"
      method: "chat"
      temperature: 0.1
      max_tokens: 4000
      api_key_env: "OPENAI_API_KEY"
      
  tools:
    web_search:
      name: "web_search"
      description: "Search the web for current information"
      input_schema:
        query:
          type: "string"
          description: "Search query"
      output_schema:
        results:
          type: "array"
          description: "Search results"
          
  agents:
    document_analyzer:
      type: "react_agent"
      llm_config: "openai_gpt4"
      system_prompt: "You are a document analysis expert..."
      tools: ["file_reader"]
      next: "content_summarizer"
      
  conditional_edges:
    quality_check:
      from: "fact_checker"
      to: "report_generator"
      condition:
        type: "content_based"
        function: "check_quality_threshold"
        parameters:
          min_confidence: 0.8
          required_fields: ["accuracy_score", "completeness_score"]
```

## 🚀 Usage

### Basic Usage

```python
from workflow_engine import WorkflowEngine

# Initialize the workflow engine
engine = WorkflowEngine("config.yaml")

# Load and build the workflow
config = engine.load_config()
graph = engine.build_workflow()

# Run the workflow
result = engine.run_workflow_sync("Process this document")
```

### Advanced Usage with Custom Functions

```python
# Register custom condition functions
def needs_revision(context, max_iterations=3):
    # Your custom logic here
    return True

engine.register_custom_condition("needs_revision", needs_revision)

# Register custom tools
def custom_analyzer(data):
    return f"Analyzed: {data}"

engine.register_custom_tool("custom_analyzer", custom_analyzer)

# Run workflow
result = await engine.run_workflow("Your input here")
```

## 🔧 Supported Features

### LLM Providers
- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo
- **Anthropic**: Claude-3 Sonnet, Claude-3 Haiku, Claude-3 Opus
- **AWS Bedrock**: Claude, Llama, Titan models

### Agent Types
- **ReAct Agents**: Reasoning and acting agents with tool usage
- **Extensible**: Easy to add new agent types

### Tools
- **Built-in**: Web search, file reader, calculator
- **Custom**: Register any Python function as a tool
- **External APIs**: Easy integration with external services

### Conditional Logic
- **Content-based**: Check for keywords, quality thresholds
- **Threshold-based**: Score and confidence checks
- **Custom functions**: Any Python function for complex logic

### Workflow Patterns
- **Sequential**: Simple chain of agents
- **Conditional**: Branching based on conditions
- **Loops**: Iterative processing with limits
- **Parallel**: Multiple agents working simultaneously (planned)

## 📦 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
# For AWS Bedrock, configure AWS credentials
```

3. Create your configuration file and run:
```bash
python main.py
```

## 🎯 Example Workflows

### Document Processing Pipeline
1. **Document Analyzer**: Extracts key information
2. **Content Summarizer**: Creates concise summaries
3. **Fact Checker**: Verifies information accuracy
4. **Report Generator**: Creates final reports

### Research Assistant
1. **Query Analyzer**: Understands research questions
2. **Web Researcher**: Searches for relevant information
3. **Data Analyzer**: Processes and analyzes data
4. **Report Writer**: Creates comprehensive reports

### Quality Assurance
1. **Content Creator**: Generates initial content
2. **Quality Checker**: Evaluates content quality
3. **Revision Agent**: Makes improvements if needed
4. **Final Reviewer**: Approves final output

## 🔍 Monitoring and Debugging

The system provides comprehensive logging and state tracking:

```python
# Get workflow information
info = engine.get_workflow_info()
print(f"Agents: {info['agents']}")
print(f"Tools: {info['tools']}")

# Monitor execution
result = await engine.run_workflow("input")
print(f"Iterations: {result['iterations']}")
print(f"Messages: {result['messages']}")
print(f"Context: {result['context']}")
```

## 🛠️ Extending the System

### Adding New Agent Types

```python
# In agent_factory.py
def _create_custom_agent(self, llm, tools, system_prompt):
    # Your custom agent creation logic
    pass

# Update the create_agent method to handle your new type
if config.type == "custom_agent":
    agent = self._create_custom_agent(llm, tools, config.system_prompt)
```

### Adding New Tools

```python
# In tool_factory.py
def _create_custom_tool(self, config):
    def custom_tool(input_data):
        # Your tool logic here
        return "result"
    
    return Tool(
        name=config.name,
        description=config.description,
        func=custom_tool
    )
```

### Adding New LLM Providers

```python
# In llm_factory.py
def _create_custom_llm(self, config):
    # Your custom LLM creation logic
    pass

# Update the create_llm method
elif config.provider == "custom":
    llm = self._create_custom_llm(config)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Add tests for new functionality
5. Submit a pull request


## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review example configurations
3. Open an issue with detailed information
4. Provide configuration files and error messages 
