#!/usr/bin/env python3
"""
Demo script for the Java Spring Boot Code Generator
Shows different code generation scenarios and capabilities
"""

import asyncio
import os
from code_generator import CodeGenerator

def create_demo_requirements():
    """Create demo requirements for different scenarios"""
    
    # Scenario 1: Simple REST API
    simple_api = """# Simple REST API - Spring Boot Application

## Project Overview
Create a simple REST API for managing books using Spring Boot.

## Functional Requirements
- **Book Management**: CRUD operations for books
- **Search**: Search books by title, author, or genre
- **Pagination**: Support pagination for large datasets

## API Endpoints
- GET `/api/books` - Get all books (with pagination)
- GET `/api/books/{id}` - Get book by ID
- POST `/api/books` - Create new book
- PUT `/api/books/{id}` - Update book
- DELETE `/api/books/{id}` - Delete book
- GET `/api/books/search` - Search books

## Technical Requirements
- Spring Boot 3.x
- Java 17
- H2 Database
- Maven build
- OpenAPI documentation

## Database Schema
- **books**: id, title, author, genre, isbn, published_date, created_at, updated_at
"""
    
    # Scenario 2: E-commerce API
    ecommerce_api = """# E-commerce API - Spring Boot Application

## Project Overview
Create a comprehensive e-commerce API with user management, product catalog, and order processing.

## Functional Requirements
- **User Management**: Registration, authentication, profiles
- **Product Catalog**: Categories, products, inventory
- **Shopping Cart**: Add/remove items, quantity management
- **Order Processing**: Create orders, payment integration
- **Admin Panel**: Product management, user management

## API Endpoints
### Authentication
- POST `/api/auth/register` - User registration
- POST `/api/auth/login` - User login
- POST `/api/auth/logout` - User logout

### Products
- GET `/api/products` - Get all products
- GET `/api/products/{id}` - Get product by ID
- POST `/api/products` - Create product (admin)
- PUT `/api/products/{id}` - Update product (admin)
- DELETE `/api/products/{id}` - Delete product (admin)

### Orders
- GET `/api/orders` - Get user orders
- POST `/api/orders` - Create order
- GET `/api/orders/{id}` - Get order details

## Technical Requirements
- Spring Boot 3.x
- Java 17
- PostgreSQL database
- JWT authentication
- Spring Security
- Swagger documentation
"""
    
    # Write demo requirements
    with open("demo_simple_api.md", "w") as f:
        f.write(simple_api)
    
    with open("demo_ecommerce.md", "w") as f:
        f.write(ecommerce_api)
    
    return ["demo_simple_api.md", "demo_ecommerce.md"]

async def demo_simple_api():
    """Demo 1: Simple REST API generation"""
    print("🎯 Demo 1: Simple REST API Generation")
    print("=" * 50)
    
    generator = CodeGenerator("code_generation_config.yaml")
    
    try:
        result = await generator.generate_code("demo_simple_api.md", "book-api")
        
        if result["success"]:
            print("✅ Simple API generation completed!")
            print(f"📁 Generated files: {len(generator.get_generated_files())}")
            
            # Show some generated files
            files = generator.get_generated_files()
            for file_path in files[:5]:  # Show first 5 files
                print(f"   📄 {file_path}")
            
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
        else:
            print(f"❌ Failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

async def demo_ecommerce_api():
    """Demo 2: Complex E-commerce API generation"""
    print("\n🎯 Demo 2: E-commerce API Generation")
    print("=" * 50)
    
    generator = CodeGenerator("code_generation_config.yaml")
    
    try:
        result = await generator.generate_code("demo_ecommerce.md", "ecommerce-api")
        
        if result["success"]:
            print("✅ E-commerce API generation completed!")
            print(f"📁 Generated files: {len(generator.get_generated_files())}")
            
            # Show file structure
            files = generator.get_generated_files()
            categories = {}
            for file_path in files:
                category = file_path.split('/')[1] if len(file_path.split('/')) > 1 else 'root'
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
            
            print("📊 File categories:")
            for category, count in categories.items():
                print(f"   {category}: {count} files")
        else:
            print(f"❌ Failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_with_sample_requirements():
    """Demo 3: Using the provided sample requirements"""
    print("\n🎯 Demo 3: User Management System")
    print("=" * 50)
    
    if not os.path.exists("sample_requirements.md"):
        print("❌ sample_requirements.md not found")
        return
    
    generator = CodeGenerator("code_generation_config.yaml")
    
    try:
        result = generator.run_sync("sample_requirements.md", "user-management-system")
        
        if result["success"]:
            print("✅ User Management System generation completed!")
            print(f"🔄 Iterations: {result['iterations']}")
            
            # Show agent results
            print("\n🤖 Agent Results:")
            for agent, context in result['context'].items():
                output = context.get('output', 'No output')
                print(f"   {agent}: {output[:100]}...")
        else:
            print(f"❌ Failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_workflow_info():
    """Demo 4: Show workflow information"""
    print("\n🎯 Demo 4: Workflow Information")
    print("=" * 50)
    
    generator = CodeGenerator("code_generation_config.yaml")
    
    try:
        config = generator.engine.load_config()
        info = generator.engine.get_workflow_info()
        
        print("📊 Code Generation Workflow:")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Agents: {', '.join(info['agents'])}")
        print(f"   Tools: {', '.join(info['tools'])}")
        print(f"   LLM Configs: {', '.join(info['llm_configs'])}")
        
        if info['conditional_edges']:
            print(f"   Conditional Edges: {', '.join(info['conditional_edges'])}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def cleanup_demo_files():
    """Clean up demo files"""
    demo_files = ["demo_simple_api.md", "demo_ecommerce.md"]
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)

async def main():
    """Run all demos"""
    print("🚀 Java Spring Boot Code Generator - Demo")
    print("=" * 60)
    
    # Create demo requirements
    demo_files = create_demo_requirements()
    
    try:
        # Demo 1: Simple API
        await demo_simple_api()
        
        # Demo 2: E-commerce API
        await demo_ecommerce_api()
        
        # Demo 3: Sample requirements
        demo_with_sample_requirements()
        
        # Demo 4: Workflow info
        demo_workflow_info()
        
        print("\n✅ All demos completed!")
        print("\n💡 Tips:")
        print("- Check the 'generated-code' directory for output")
        print("- Review generated code quality and structure")
        print("- Test the generated applications")
        print("- Customize agent prompts for better results")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        cleanup_demo_files()

if __name__ == "__main__":
    asyncio.run(main()) 