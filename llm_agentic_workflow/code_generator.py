#!/usr/bin/env python3
"""
Java Spring Boot Code Generator
Uses LangGraph workflow to generate complete Spring Boot applications from requirements
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional
from workflow_engine import WorkflowEngine

class CodeGenerator:
    """Main code generator class"""
    
    def __init__(self, config_path: str = "code_generation_config.yaml"):
        self.engine = WorkflowEngine(config_path)
        self.generated_code_dir = "generated-code"
        self.requirements_file = None
        
    def setup_environment(self):
        """Setup the environment for code generation"""
        # Create generated-code directory
        os.makedirs(self.generated_code_dir, exist_ok=True)
        print(f"✅ Created directory: {self.generated_code_dir}")
        
        # Register custom condition functions
        self._register_custom_functions()
        
    def _register_custom_functions(self):
        """Register custom functions for the workflow"""
        
        def needs_regeneration(context: Dict[str, Any], min_quality_score: float = 0.8, required_components: list = None) -> bool:
            """Check if code needs regeneration based on quality metrics"""
            output = context.get("output", "").lower()
            
            # Check for error indicators
            error_indicators = ["error", "exception", "failed", "invalid", "missing"]
            for indicator in error_indicators:
                if indicator in output:
                    return True
            
            # Check for missing components
            if required_components:
                for component in required_components:
                    if component not in output:
                        return True
            
            # Check quality score (simple heuristic)
            quality_score = 0.0
            if "public class" in output:
                quality_score += 0.3
            if "import" in output:
                quality_score += 0.2
            if "annotation" in output or "@" in output:
                quality_score += 0.2
            if "spring" in output:
                quality_score += 0.3
            
            return quality_score < min_quality_score
        
        def needs_final_touch(context: Dict[str, Any], max_iterations: int = 3) -> bool:
            """Check if final improvements are needed"""
            iteration_count = context.get("iteration_count", 0)
            
            if iteration_count >= max_iterations:
                return False
            
            output = context.get("output", "").lower()
            
            # Check for improvement indicators
            improvement_indicators = ["improve", "enhance", "better", "optimize", "refactor"]
            for indicator in improvement_indicators:
                if indicator in output:
                    return True
            
            return False
        
        # Register the functions
        self.engine.register_custom_condition("needs_regeneration", needs_regeneration)
        self.engine.register_custom_condition("needs_final_touch", needs_final_touch)
    
    def load_requirements(self, requirements_file: str) -> str:
        """Load requirements from markdown file"""
        if not os.path.exists(requirements_file):
            raise FileNotFoundError(f"Requirements file not found: {requirements_file}")
        
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = f.read()
        
        self.requirements_file = requirements_file
        print(f"✅ Loaded requirements from: {requirements_file}")
        return requirements
    
    async def generate_code(self, requirements_file: str, project_name: str = None) -> Dict[str, Any]:
        """Generate code from requirements"""
        
        # Setup environment
        self.setup_environment()
        
        # Load requirements
        requirements = self.load_requirements(requirements_file)
        
        # Load and build workflow
        print("📋 Loading code generation workflow...")
        config = self.engine.load_config()
        graph = self.engine.build_workflow()
        print("✅ Workflow built successfully!")
        
        # Prepare input with requirements and project context
        input_text = f"""
Generate a Java Spring Boot application based on the following requirements:

{requirements}

Project Context:
- Project Name: {project_name or 'spring-boot-app'}
- Output Directory: {self.generated_code_dir}
- Technology Stack: Spring Boot 3.x, Java 17, Maven, PostgreSQL
- Security: JWT Authentication, Spring Security
- Documentation: OpenAPI 3 (Swagger)

Please analyze the requirements, plan the architecture, generate the code, verify it, and create documentation.
"""
        
        # Run the workflow
        print("🚀 Starting code generation...")
        result = await self.engine.run_workflow(input_text)
        
        return result
    
    def run_sync(self, requirements_file: str, project_name: str = None) -> Dict[str, Any]:
        """Run code generation synchronously"""
        return asyncio.run(self.generate_code(requirements_file, project_name))
    
    def get_generated_files(self) -> list:
        """Get list of generated files"""
        files = []
        for root, dirs, filenames in os.walk(self.generated_code_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                files.append(file_path)
        return files
    
    def print_generation_summary(self, result: Dict[str, Any]):
        """Print a summary of the code generation process"""
        print("\n" + "="*60)
        print("🎉 CODE GENERATION SUMMARY")
        print("="*60)
        
        if result["success"]:
            print(f"✅ Status: Success")
            print(f"🔄 Iterations: {result['iterations']}")
            print(f"📝 Messages: {len(result['messages'])}")
            
            # Show generated files
            files = self.get_generated_files()
            print(f"📁 Generated Files: {len(files)}")
            for file_path in files:
                print(f"   📄 {file_path}")
            
            # Show agent contexts
            print(f"\n🤖 Agent Results:")
            for agent, context in result['context'].items():
                output = context.get('output', 'No output')
                print(f"   {agent}: {output[:100]}...")
        else:
            print(f"❌ Status: Failed")
            print(f"🚨 Error: {result['error']}")
    
    def cleanup(self):
        """Clean up temporary files"""
        # Remove any temporary files if needed
        pass

def main():
    """Main function to run the code generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Java Spring Boot Code Generator")
    parser.add_argument("requirements", help="Path to requirements markdown file")
    parser.add_argument("--project-name", help="Name of the project", default="spring-boot-app")
    parser.add_argument("--config", help="Path to workflow config", default="code_generation_config.yaml")
    parser.add_argument("--async", action="store_true", help="Run asynchronously")
    
    args = parser.parse_args()
    
    # Initialize code generator
    generator = CodeGenerator(args.config)
    
    try:
        if args.async:
            # Run asynchronously
            result = asyncio.run(generator.generate_code(args.requirements, args.project_name))
        else:
            # Run synchronously
            result = generator.run_sync(args.requirements, args.project_name)
        
        # Print summary
        generator.print_generation_summary(result)
        
    except Exception as e:
        print(f"❌ Error during code generation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main() 