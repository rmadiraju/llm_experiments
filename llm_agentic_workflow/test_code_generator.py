#!/usr/bin/env python3
"""
Test script for the Java Spring Boot Code Generator
"""

import asyncio
import os
from code_generator import CodeGenerator

def create_simple_requirements():
    """Create a simple test requirements file"""
    requirements_content = """# Simple Todo API - Spring Boot Application

## Project Overview
Create a simple Todo API using Java Spring Boot with basic CRUD operations.

## Functional Requirements

### 1. Todo Management
- **Create Todo**: Add new todo items with title and description
- **Read Todos**: Get all todos and get todo by ID
- **Update Todo**: Update todo title, description, and completion status
- **Delete Todo**: Delete todo by ID
- **Complete Todo**: Mark todo as completed

### 2. API Endpoints
- **GET** `/api/todos` - Get all todos
- **GET** `/api/todos/{id}` - Get todo by ID
- **POST** `/api/todos` - Create new todo
- **PUT** `/api/todos/{id}` - Update todo
- **DELETE** `/api/todos/{id}` - Delete todo
- **PATCH** `/api/todos/{id}/complete` - Mark todo as completed

## Technical Requirements

### 1. Technology Stack
- **Framework**: Spring Boot 3.x
- **Java Version**: Java 17
- **Database**: H2 (for simplicity)
- **Build Tool**: Maven
- **Documentation**: OpenAPI 3 (Swagger)

### 2. Project Structure
```
src/main/java/com/todo/
├── controller/
│   └── TodoController.java
├── service/
│   └── TodoService.java
├── repository/
│   └── TodoRepository.java
├── entity/
│   └── Todo.java
└── dto/
    ├── TodoDto.java
    └── CreateTodoDto.java
```

### 3. Database Schema
- **todos**: id, title, description, completed, created_at, updated_at

### 4. Configuration Files
- **application.yml**: Main configuration
- **pom.xml**: Maven dependencies

## Success Criteria
1. All CRUD operations working
2. Proper REST API structure
3. Database operations working
4. Clean, maintainable code
5. Complete documentation
"""
    
    with open("simple_todo_requirements.md", "w") as f:
        f.write(requirements_content)
    
    return "simple_todo_requirements.md"

async def test_code_generation():
    """Test the code generation system"""
    
    # Create test requirements
    requirements_file = create_simple_requirements()
    
    # Initialize code generator
    generator = CodeGenerator("code_generation_config.yaml")
    
    try:
        print("🧪 Testing Code Generation System")
        print("=" * 50)
        
        # Run code generation
        result = await generator.generate_code(requirements_file, "todo-api")
        
        # Print results
        generator.print_generation_summary(result)
        
        # Show generated files
        files = generator.get_generated_files()
        if files:
            print(f"\n📁 Generated Files ({len(files)}):")
            for file_path in files:
                print(f"   📄 {file_path}")
                
                # Show file size
                try:
                    size = os.path.getsize(file_path)
                    print(f"      Size: {size} bytes")
                except:
                    pass
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test file
        if os.path.exists("simple_todo_requirements.md"):
            os.remove("simple_todo_requirements.md")

def test_sync_generation():
    """Test synchronous code generation"""
    
    requirements_file = create_simple_requirements()
    generator = CodeGenerator("code_generation_config.yaml")
    
    try:
        print("\n🔄 Testing Synchronous Generation")
        print("=" * 40)
        
        result = generator.run_sync(requirements_file, "todo-api-sync")
        
        print(f"✅ Sync generation completed!")
        print(f"Success: {result['success']}")
        print(f"Iterations: {result['iterations']}")
        
        files = generator.get_generated_files()
        print(f"Files generated: {len(files)}")
        
    except Exception as e:
        print(f"❌ Sync test failed: {e}")
    
    finally:
        if os.path.exists("simple_todo_requirements.md"):
            os.remove("simple_todo_requirements.md")

def test_with_sample_requirements():
    """Test with the provided sample requirements"""
    
    if not os.path.exists("sample_requirements.md"):
        print("❌ sample_requirements.md not found")
        return
    
    generator = CodeGenerator("code_generation_config.yaml")
    
    try:
        print("\n📋 Testing with Sample Requirements")
        print("=" * 40)
        
        result = generator.run_sync("sample_requirements.md", "user-management-system")
        
        generator.print_generation_summary(result)
        
    except Exception as e:
        print(f"❌ Sample requirements test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Java Spring Boot Code Generator")
    print("=" * 60)
    
    # Test 1: Async generation with simple requirements
    asyncio.run(test_code_generation())
    
    # Test 2: Sync generation
    test_sync_generation()
    
    # Test 3: With sample requirements
    test_with_sample_requirements()
    
    print("\n✅ All tests completed!") 