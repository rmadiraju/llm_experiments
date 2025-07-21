import os
import json
import requests
import re
from typing import Dict, Any, List, Callable
from langchain.tools import BaseTool, Tool
from langchain.schema import BaseOutputParser
from config_parser import ToolConfig

class ToolFactory:
    """Factory for creating tool instances based on configuration"""
    
    def __init__(self):
        self._tool_cache: Dict[str, BaseTool] = {}
        self._custom_tools: Dict[str, Callable] = {}
    
    def register_custom_tool(self, name: str, func: Callable):
        """Register a custom tool function"""
        self._custom_tools[name] = func
    
    def create_tool(self, config: ToolConfig) -> BaseTool:
        """Create a tool instance based on configuration"""
        
        # Check cache first
        if config.name in self._tool_cache:
            return self._tool_cache[config.name]
        
        tool = None
        
        if config.name == "web_search":
            tool = self._create_web_search_tool(config)
        elif config.name == "file_reader":
            tool = self._create_file_reader_tool(config)
        elif config.name == "calculator":
            tool = self._create_calculator_tool(config)
        elif config.name == "file_writer":
            tool = self._create_file_writer_tool(config)
        elif config.name == "directory_creator":
            tool = self._create_directory_creator_tool(config)
        elif config.name == "code_validator":
            tool = self._create_code_validator_tool(config)
        elif config.name in self._custom_tools:
            tool = self._create_custom_tool(config)
        else:
            raise ValueError(f"Unknown tool type: {config.name}")
        
        # Cache the tool instance
        self._tool_cache[config.name] = tool
        return tool
    
    def _create_web_search_tool(self, config: ToolConfig) -> BaseTool:
        """Create web search tool"""
        def web_search(query: str) -> str:
            """Search the web for information"""
            # This is a simplified implementation
            # In a real implementation, you might use a search API like Google Custom Search
            try:
                # For demo purposes, return a mock response
                return f"Search results for '{query}': Found 5 relevant results about {query}."
            except Exception as e:
                return f"Error performing web search: {str(e)}"
        
        return Tool(
            name=config.name,
            description=config.description,
            func=web_search
        )
    
    def _create_file_reader_tool(self, config: ToolConfig) -> BaseTool:
        """Create file reader tool"""
        def file_reader(file_path: str) -> str:
            """Read content from a file"""
            try:
                if not os.path.exists(file_path):
                    return f"Error: File '{file_path}' not found"
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                return f"File content from '{file_path}':\n{content}"
            except Exception as e:
                return f"Error reading file '{file_path}': {str(e)}"
        
        return Tool(
            name=config.name,
            description=config.description,
            func=file_reader
        )
    
    def _create_calculator_tool(self, config: ToolConfig) -> BaseTool:
        """Create calculator tool"""
        def calculator(expression: str) -> str:
            """Evaluate mathematical expressions"""
            try:
                # Remove any potentially dangerous characters
                expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
                
                # Evaluate the expression
                result = eval(expression)
                return f"Result of '{expression}': {result}"
            except Exception as e:
                return f"Error evaluating expression '{expression}': {str(e)}"
        
        return Tool(
            name=config.name,
            description=config.description,
            func=calculator
        )
    
    def _create_file_writer_tool(self, config: ToolConfig) -> BaseTool:
        """Create file writer tool"""
        def file_writer(file_path: str, content: str) -> str:
            """Write content to a file"""
            try:
                # Ensure the directory exists
                import os
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                return f"Successfully wrote content to '{file_path}'"
            except Exception as e:
                return f"Error writing to file '{file_path}': {str(e)}"
        
        return Tool(
            name=config.name,
            description=config.description,
            func=file_writer
        )
    
    def _create_directory_creator_tool(self, config: ToolConfig) -> BaseTool:
        """Create directory creator tool"""
        def directory_creator(directory_path: str) -> str:
            """Create a directory"""
            try:
                import os
                os.makedirs(directory_path, exist_ok=True)
                return f"Successfully created directory '{directory_path}'"
            except Exception as e:
                return f"Error creating directory '{directory_path}': {str(e)}"
        
        return Tool(
            name=config.name,
            description=config.description,
            func=directory_creator
        )
    
    def _create_code_validator_tool(self, config: ToolConfig) -> BaseTool:
        """Create code validator tool"""
        def code_validator(code: str, file_type: str) -> str:
            """Validate Java code syntax and structure"""
            try:
                errors = []
                suggestions = []
                
                # Basic Java syntax validation
                if file_type.lower() == "java":
                    # Check for basic Java structure
                    if "public class" not in code and "public interface" not in code and "public enum" not in code:
                        if "class" in code or "interface" in code or "enum" in code:
                            errors.append("Missing 'public' modifier")
                    
                    # Check for proper package declaration
                    if "package" not in code and "import" in code:
                        suggestions.append("Consider adding package declaration")
                    
                    # Check for proper imports
                    if "import" in code and ";" not in code.split("import")[1].split("\n")[0]:
                        errors.append("Invalid import statement")
                    
                    # Check for proper annotations
                    if "@" in code and "(" not in code.split("@")[1].split("\n")[0]:
                        errors.append("Invalid annotation syntax")
                
                # XML validation for pom.xml
                elif file_type.lower() == "xml":
                    if "<project" not in code:
                        errors.append("Missing project root element")
                    if "<groupId>" not in code:
                        errors.append("Missing groupId")
                    if "<artifactId>" not in code:
                        errors.append("Missing artifactId")
                
                # Properties validation
                elif file_type.lower() == "properties":
                    if "=" not in code and ":" not in code:
                        errors.append("Invalid properties format")
                
                # YAML validation
                elif file_type.lower() == "yml" or file_type.lower() == "yaml":
                    if "spring:" not in code and "server:" not in code:
                        suggestions.append("Consider adding Spring Boot configuration")
                
                result = {
                    "is_valid": len(errors) == 0,
                    "errors": errors,
                    "suggestions": suggestions
                }
                
                return f"Validation result: {result}"
                
            except Exception as e:
                return f"Error validating code: {str(e)}"
        
        return Tool(
            name=config.name,
            description=config.description,
            func=code_validator
        )
    
    def _create_custom_tool(self, config: ToolConfig) -> BaseTool:
        """Create custom tool from registered function"""
        if config.name not in self._custom_tools:
            raise ValueError(f"Custom tool '{config.name}' not registered")
        
        func = self._custom_tools[config.name]
        
        return Tool(
            name=config.name,
            description=config.description,
            func=func
        )
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self._tool_cache.keys()) + list(self._custom_tools.keys())
    
    def clear_cache(self):
        """Clear the tool cache"""
        self._tool_cache.clear() 