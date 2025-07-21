# Java Spring Boot Code Generator

A sophisticated code generation system that uses LangGraph workflows to generate complete Java Spring Boot applications from markdown requirements.

## 🏗️ Architecture

The code generation system consists of 5 specialized agents working together:

### 1. **Requirements Analyzer**
- Reads and parses markdown requirements
- Extracts key information about project structure
- Identifies required components and features
- Creates detailed implementation plan

### 2. **Architecture Planner**
- Designs application architecture
- Plans package structure and dependencies
- Defines entity relationships and database schema
- Plans API endpoints and REST structure

### 3. **Code Generator**
- Generates complete, working Java Spring Boot code
- Creates proper folder structure and package organization
- Implements controllers, services, repositories, and entities
- Follows Spring Boot best practices

### 4. **Code Verifier**
- Validates generated code for syntax correctness
- Checks for proper Spring Boot annotations and structure
- Ensures all required components are present
- Provides feedback and improvement suggestions

### 5. **Documentation Generator**
- Creates comprehensive README.md
- Generates API documentation with Swagger
- Documents setup and deployment instructions
- Adds code comments and JavaDoc

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Ollama (for local LLM)
# Install Ollama from https://ollama.ai/
ollama pull llama2:13b

# Set environment variables
export OLLAMA_BASE_URL="http://localhost:11434"
```

### 2. Create Requirements File

Create a markdown file with your project requirements:

```markdown
# My Spring Boot Application

## Project Overview
Create a REST API for managing users...

## Functional Requirements
- User registration and authentication
- CRUD operations for users
- Role-based access control

## Technical Requirements
- Spring Boot 3.x
- Java 17
- PostgreSQL database
- JWT authentication
```

### 3. Generate Code

```bash
# Basic usage
python code_generator.py requirements.md

# With custom project name
python code_generator.py requirements.md --project-name my-app

# Async execution
python code_generator.py requirements.md --async
```

## 📁 Generated Structure

The system generates a complete Spring Boot project structure:

```
generated-code/
├── src/
│   ├── main/
│   │   ├── java/com/yourapp/
│   │   │   ├── controller/
│   │   │   ├── service/
│   │   │   ├── repository/
│   │   │   ├── entity/
│   │   │   ├── dto/
│   │   │   ├── config/
│   │   │   ├── exception/
│   │   │   └── util/
│   │   └── resources/
│   │       ├── application.yml
│   │       └── application-dev.yml
│   └── test/
│       └── java/com/yourapp/
├── pom.xml
├── README.md
├── Dockerfile
└── docker-compose.yml
```

## 🔧 Configuration

### LLM Configuration

The system supports multiple LLM providers:

```yaml
llm_configs:
  llama_local:
    provider: "ollama"
    model: "llama2:13b"
    method: "chat"
    temperature: 0.1
    max_tokens: 4000
    
  bedrock_claude:
    provider: "bedrock"
    model: "anthropic.claude-3-sonnet-20240229-v1:0"
    method: "generate"
    temperature: 0.1
    max_tokens: 4000
    region: "us-east-1"
```

### Environment Variables

```bash
# For Ollama (local)
export OLLAMA_BASE_URL="http://localhost:11434"

# For AWS Bedrock
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# For OpenAI (fallback)
export OPENAI_API_KEY="your-openai-key"
```

## 🛠️ Tools

The system includes specialized tools for code generation:

### File Operations
- **file_writer**: Write code files to filesystem
- **directory_creator**: Create directory structures
- **file_reader**: Read content from files

### Code Validation
- **code_validator**: Validate Java code syntax and structure
- Checks for proper Spring Boot annotations
- Validates package organization and imports
- Provides improvement suggestions

## 🔄 Workflow Process

### 1. Requirements Analysis
```python
# Agent analyzes requirements and creates plan
requirements_analyzer = Agent(
    llm=llama_local,
    tools=[file_reader],
    system_prompt="You are a requirements analysis expert..."
)
```

### 2. Architecture Planning
```python
# Agent designs application architecture
architecture_planner = Agent(
    llm=llama_local,
    tools=[],
    system_prompt="You are a software architecture expert..."
)
```

### 3. Code Generation
```python
# Agent generates complete code
code_generator = Agent(
    llm=llama_local,
    tools=[file_writer, directory_creator],
    system_prompt="You are a Java Spring Boot code generation expert..."
)
```

### 4. Code Verification
```python
# Agent validates generated code
code_verifier = Agent(
    llm=llama_local,
    tools=[code_validator, file_reader],
    system_prompt="You are a code verification expert..."
)
```

### 5. Documentation Generation
```python
# Agent creates documentation
documentation_generator = Agent(
    llm=llama_local,
    tools=[file_writer],
    system_prompt="You are a technical documentation expert..."
)
```

## 🎯 Example Usage

### Simple Todo API

```bash
# Create requirements file
cat > todo_requirements.md << EOF
# Todo API - Spring Boot Application

## Functional Requirements
- Create, read, update, delete todos
- Mark todos as completed
- REST API endpoints

## Technical Requirements
- Spring Boot 3.x
- Java 17
- H2 database
- Maven build
EOF

# Generate code
python code_generator.py todo_requirements.md --project-name todo-api
```

### User Management System

```bash
# Use the provided sample requirements
python code_generator.py sample_requirements.md --project-name user-management
```

## 🔍 Monitoring and Debugging

### View Generation Progress
```python
from code_generator import CodeGenerator

generator = CodeGenerator()
result = generator.run_sync("requirements.md")

print(f"Success: {result['success']}")
print(f"Iterations: {result['iterations']}")
print(f"Messages: {len(result['messages'])}")

# Show generated files
files = generator.get_generated_files()
for file_path in files:
    print(f"Generated: {file_path}")
```

### Check Agent Results
```python
# View each agent's output
for agent, context in result['context'].items():
    print(f"{agent}: {context.get('output', 'No output')}")
```

## 🧪 Testing

Run the test suite:

```bash
# Test with simple requirements
python test_code_generator.py

# Test specific scenarios
python -c "
from code_generator import CodeGenerator
generator = CodeGenerator()
result = generator.run_sync('simple_todo_requirements.md')
print('Test completed!')
"
```

## 🔧 Customization

### Add Custom Tools

```python
def custom_code_formatter(code: str) -> str:
    """Format generated code"""
    # Your formatting logic
    return formatted_code

generator = CodeGenerator()
generator.engine.register_custom_tool("code_formatter", custom_code_formatter)
```

### Add Custom Conditions

```python
def custom_quality_check(context, min_score=0.8):
    """Custom quality assessment"""
    # Your quality logic
    return quality_score >= min_score

generator.engine.register_custom_condition("custom_quality", custom_quality_check)
```

### Modify Agent Prompts

Edit the YAML configuration to customize agent behavior:

```yaml
agents:
  code_generator:
    system_prompt: |
      You are a Java Spring Boot expert.
      Focus on clean architecture and best practices.
      Always include proper error handling.
```

## 🚀 Advanced Features

### Conditional Code Generation
The system can regenerate code based on quality checks:

```yaml
conditional_edges:
  quality_check:
    from: "code_verifier"
    to: "code_generator"
    condition:
      type: "content_based"
      function: "needs_regeneration"
      parameters:
        min_quality_score: 0.8
```

### Multi-Model Support
Switch between different LLMs based on requirements:

```yaml
llm_configs:
  fast_model:
    provider: "ollama"
    model: "llama2:7b"
    
  quality_model:
    provider: "bedrock"
    model: "anthropic.claude-3-sonnet-20240229-v1:0"
```

## 📊 Performance

### Optimization Tips
1. **Use local models** (Ollama) for faster iteration
2. **Cache generated code** to avoid regeneration
3. **Parallel processing** for independent components
4. **Incremental generation** for large projects

### Monitoring
- Track generation time per agent
- Monitor code quality metrics
- Log generation errors and retries
- Measure success rates

## 🤝 Contributing

1. **Add new agent types** for different frameworks
2. **Extend tool capabilities** for more languages
3. **Improve validation logic** for better code quality
4. **Add support for more LLM providers**

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the configuration files
2. Verify LLM provider setup
3. Review generated code quality
4. Open an issue with detailed information 