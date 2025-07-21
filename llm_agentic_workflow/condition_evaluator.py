import re
import json
from typing import Dict, Any, Callable, Optional
from config_parser import ConditionConfig

class ConditionEvaluator:
    """Evaluates conditions for conditional edges in the workflow"""
    
    def __init__(self):
        self._custom_functions: Dict[str, Callable] = {}
    
    def register_custom_function(self, name: str, func: Callable):
        """Register a custom condition function"""
        self._custom_functions[name] = func
    
    def evaluate_condition(self, condition: ConditionConfig, context: Dict[str, Any]) -> bool:
        """Evaluate a condition based on its type and parameters"""
        
        if condition.type == "content_based":
            return self._evaluate_content_based_condition(condition, context)
        elif condition.type == "threshold_based":
            return self._evaluate_threshold_based_condition(condition, context)
        elif condition.type == "custom_function":
            return self._evaluate_custom_function_condition(condition, context)
        else:
            raise ValueError(f"Unsupported condition type: {condition.type}")
    
    def _evaluate_content_based_condition(self, condition: ConditionConfig, context: Dict[str, Any]) -> bool:
        """Evaluate content-based conditions"""
        function_name = condition.function
        parameters = condition.parameters
        
        if function_name == "check_quality_threshold":
            return self._check_quality_threshold(context, parameters)
        elif function_name == "contains_keywords":
            return self._contains_keywords(context, parameters)
        else:
            raise ValueError(f"Unknown content-based function: {function_name}")
    
    def _evaluate_threshold_based_condition(self, condition: ConditionConfig, context: Dict[str, Any]) -> bool:
        """Evaluate threshold-based conditions"""
        function_name = condition.function
        parameters = condition.parameters
        
        if function_name == "score_above_threshold":
            return self._score_above_threshold(context, parameters)
        elif function_name == "confidence_above_threshold":
            return self._confidence_above_threshold(context, parameters)
        else:
            raise ValueError(f"Unknown threshold-based function: {function_name}")
    
    def _evaluate_custom_function_condition(self, condition: ConditionConfig, context: Dict[str, Any]) -> bool:
        """Evaluate custom function conditions"""
        function_name = condition.function
        
        if function_name not in self._custom_functions:
            raise ValueError(f"Custom function '{function_name}' not registered")
        
        func = self._custom_functions[function_name]
        parameters = condition.parameters
        
        try:
            return func(context, **parameters)
        except Exception as e:
            print(f"Error evaluating custom function '{function_name}': {e}")
            return False
    
    def _check_quality_threshold(self, context: Dict[str, Any], parameters: Dict[str, Any]) -> bool:
        """Check if quality metrics meet the threshold"""
        min_confidence = parameters.get("min_confidence", 0.8)
        required_fields = parameters.get("required_fields", [])
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in context:
                return False
        
        # Check confidence threshold
        confidence = context.get("confidence", 0.0)
        if confidence < min_confidence:
            return False
        
        return True
    
    def _contains_keywords(self, context: Dict[str, Any], parameters: Dict[str, Any]) -> bool:
        """Check if content contains specific keywords"""
        keywords = parameters.get("keywords", [])
        content = context.get("content", "").lower()
        
        for keyword in keywords:
            if keyword.lower() in content:
                return True
        
        return False
    
    def _score_above_threshold(self, context: Dict[str, Any], parameters: Dict[str, Any]) -> bool:
        """Check if score is above threshold"""
        threshold = parameters.get("threshold", 0.5)
        score = context.get("score", 0.0)
        
        return score > threshold
    
    def _confidence_above_threshold(self, context: Dict[str, Any], parameters: Dict[str, Any]) -> bool:
        """Check if confidence is above threshold"""
        threshold = parameters.get("threshold", 0.8)
        confidence = context.get("confidence", 0.0)
        
        return confidence > threshold
    
    def get_available_functions(self) -> Dict[str, Callable]:
        """Get all available custom functions"""
        return self._custom_functions.copy() 