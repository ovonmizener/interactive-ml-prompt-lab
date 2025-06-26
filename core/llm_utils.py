"""
LLM Utilities - Large Language Model integration and prompt engineering tools
Prompt templates, API wrappers, and chain-of-thought analysis
"""

import requests
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import openai
import tiktoken
from dataclasses import dataclass
from datetime import datetime
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Data class for LLM response"""
    text: str
    model: str
    tokens_used: Dict[str, int]
    response_time: float
    finish_reason: str
    logprobs: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class PromptTemplate:
    """Prompt template management and generation"""
    
    def __init__(self):
        self.templates = {}
        self.variables = {}
        self.load_default_templates()
    
    def load_default_templates(self):
        """Load default prompt templates"""
        self.templates = {
            "text_classification": {
                "description": "Classify text into predefined categories",
                "template": """Classify the following text into one of these categories: {categories}

Text: {input_text}

Please provide your classification and a brief explanation.

Classification:""",
                "variables": ["categories", "input_text"],
                "example": {
                    "categories": "positive, negative, neutral",
                    "input_text": "I love this product! It's amazing."
                }
            },
            "text_generation": {
                "description": "Generate creative text based on a prompt",
                "template": """Write a creative story based on the following prompt:

Prompt: {prompt}

Story:""",
                "variables": ["prompt"],
                "example": {
                    "prompt": "A robot discovers emotions for the first time"
                }
            },
            "question_answering": {
                "description": "Answer questions based on given context",
                "template": """Answer the following question based on the provided context:

Context: {context}

Question: {question}

Please provide a comprehensive answer with supporting evidence from the context.

Answer:""",
                "variables": ["context", "question"],
                "example": {
                    "context": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                    "question": "What is machine learning?"
                }
            },
            "code_generation": {
                "description": "Generate code based on requirements",
                "template": """Write Python code for the following requirement:

Requirement: {requirement}

Please provide clean, well-documented code with comments explaining the logic.

Code:""",
                "variables": ["requirement"],
                "example": {
                    "requirement": "Create a function to calculate the factorial of a number"
                }
            },
            "chain_of_thought": {
                "description": "Solve problems step by step with reasoning",
                "template": """Let's solve this problem step by step:

Problem: {problem}

Let me think through this step by step:

1) First, I need to understand what's being asked...
2) Then, I'll break it down into smaller parts...
3) Finally, I'll arrive at the solution.

Solution:""",
                "variables": ["problem"],
                "example": {
                    "problem": "If a train travels 120 miles in 2 hours, what is its average speed?"
                }
            },
            "legal_analysis": {
                "description": "Analyze legal documents and provide insights",
                "template": """Analyze the following legal document and provide insights:

Document: {document}

Please provide:
1. Key legal issues identified
2. Potential risks or concerns
3. Recommendations for action

Analysis:""",
                "variables": ["document"],
                "example": {
                    "document": "Sample legal contract text..."
                }
            },
            "custom": {
                "description": "Custom prompt template",
                "template": "{custom_prompt}",
                "variables": ["custom_prompt"],
                "example": {
                    "custom_prompt": "Your custom prompt here..."
                }
            }
        }
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """Get a specific prompt template"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]
    
    def list_templates(self) -> List[str]:
        """List all available templates"""
        return list(self.templates.keys())
    
    def create_prompt(self, template_name: str, variables: Dict[str, str]) -> str:
        """
        Create a prompt from template and variables
        
        Args:
            template_name: Name of the template to use
            variables: Dictionary of variables to substitute
            
        Returns:
            Formatted prompt string
        """
        template = self.get_template(template_name)
        
        try:
            prompt = template["template"].format(**variables)
            return prompt
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")
    
    def add_template(
        self,
        name: str,
        template: str,
        variables: List[str],
        description: str = "",
        example: Optional[Dict[str, str]] = None
    ):
        """Add a new custom template"""
        self.templates[name] = {
            "description": description,
            "template": template,
            "variables": variables,
            "example": example or {}
        }
        logger.info(f"Added new template: {name}")
    
    def validate_variables(self, template_name: str, variables: Dict[str, str]) -> bool:
        """Validate that all required variables are provided"""
        template = self.get_template(template_name)
        required_vars = set(template["variables"])
        provided_vars = set(variables.keys())
        
        missing_vars = required_vars - provided_vars
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        return True

class TokenAnalyzer:
    """Token analysis and counting utilities"""
    
    def __init__(self):
        self.encoders = {}
    
    def get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get tokenizer for a specific model"""
        if model not in self.encoders:
            try:
                if "gpt-4" in model or "gpt-3.5" in model:
                    self.encoders[model] = tiktoken.encoding_for_model(model)
                else:
                    # Default to cl100k_base for other models
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Could not get encoder for {model}, using cl100k_base: {e}")
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        
        return self.encoders[model]
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text for a specific model"""
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))
    
    def analyze_tokens(
        self,
        text: str,
        model: str = "gpt-3.5-turbo"
    ) -> Dict[str, Any]:
        """
        Analyze token usage in text
        
        Args:
            text: Input text
            model: Model name for tokenization
            
        Returns:
            Token analysis dictionary
        """
        encoder = self.get_encoder(model)
        tokens = encoder.encode(text)
        
        analysis = {
            "total_tokens": len(tokens),
            "characters": len(text),
            "words": len(text.split()),
            "sentences": len(re.split(r'[.!?]+', text)),
            "token_to_char_ratio": len(tokens) / len(text) if text else 0,
            "token_to_word_ratio": len(tokens) / len(text.split()) if text.split() else 0
        }
        
        return analysis
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-3.5-turbo"
    ) -> float:
        """
        Estimate API cost for token usage
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        # Approximate costs per 1K tokens (as of 2024)
        costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
        
        # Find matching cost structure
        cost_key = None
        for key in costs.keys():
            if key in model:
                cost_key = key
                break
        
        if cost_key is None:
            # Default to gpt-3.5-turbo pricing
            cost_key = "gpt-3.5-turbo"
        
        cost = (prompt_tokens / 1000 * costs[cost_key]["input"] + 
                completion_tokens / 1000 * costs[cost_key]["output"])
        
        return cost

class LLMClient:
    """LLM API client with multiple provider support"""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        """Setup API client based on provider"""
        if self.provider == "openai":
            if self.api_key:
                openai.api_key = self.api_key
            self.client = openai
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_response(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: bool = False
    ) -> LLMResponse:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            logprobs: Whether to return log probabilities
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                response = self.client.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logprobs=logprobs
                )
                
                # Extract response data
                response_text = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                usage = response.usage
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Create LLMResponse object
                llm_response = LLMResponse(
                    text=response_text,
                    model=model,
                    tokens_used={
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    },
                    response_time=response_time,
                    finish_reason=finish_reason
                )
                
                logger.info(f"Generated response using {model} in {response_time:.2f}s")
                return llm_response
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> LLMResponse:
        """Generate response with system and user prompts"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # This would need to be adapted for the specific API being used
        # For now, we'll combine them into a single prompt
        combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
        return self.generate_response(combined_prompt, model, **kwargs)

class ChainOfThoughtAnalyzer:
    """Chain of thought analysis and extraction"""
    
    def __init__(self):
        self.thought_patterns = [
            r"Let me think step by step:",
            r"First, I need to",
            r"Then, I'll",
            r"Finally,",
            r"Step \d+:",
            r"1\)",
            r"2\)",
            r"3\)",
            r"Therefore,",
            r"Thus,",
            r"So,"
        ]
    
    def extract_chain_of_thought(self, response: str) -> Dict[str, Any]:
        """
        Extract chain of thought reasoning from response
        
        Args:
            response: LLM response text
            
        Returns:
            Chain of thought analysis
        """
        analysis = {
            "has_chain_of_thought": False,
            "steps": [],
            "final_answer": "",
            "reasoning_quality": "low"
        }
        
        # Check if response contains chain of thought patterns
        has_patterns = any(re.search(pattern, response, re.IGNORECASE) for pattern in self.thought_patterns)
        
        if has_patterns:
            analysis["has_chain_of_thought"] = True
            
            # Extract steps
            lines = response.split('\n')
            steps = []
            final_answer = ""
            
            for line in lines:
                line = line.strip()
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in self.thought_patterns):
                    steps.append(line)
                elif line and not any(keyword in line.lower() for keyword in ['step', 'first', 'then', 'finally']):
                    final_answer += line + " "
            
            analysis["steps"] = steps
            analysis["final_answer"] = final_answer.strip()
            
            # Assess reasoning quality
            if len(steps) >= 3:
                analysis["reasoning_quality"] = "high"
            elif len(steps) >= 2:
                analysis["reasoning_quality"] = "medium"
        
        return analysis
    
    def analyze_reasoning_structure(self, response: str) -> Dict[str, Any]:
        """
        Analyze the structure of reasoning in the response
        
        Args:
            response: LLM response text
            
        Returns:
            Reasoning structure analysis
        """
        analysis = {
            "sentence_count": len(re.split(r'[.!?]+', response)),
            "paragraph_count": len(response.split('\n\n')),
            "has_numbered_steps": bool(re.search(r'\d+[\.\)]', response)),
            "has_bullet_points": bool(re.search(r'[-*•]', response)),
            "has_conclusion": bool(re.search(r'(therefore|thus|so|conclusion|answer)', response, re.IGNORECASE)),
            "avg_sentence_length": np.mean([len(s.split()) for s in re.split(r'[.!?]+', response) if s.strip()])
        }
        
        return analysis

class PromptOptimizer:
    """Prompt optimization and analysis tools"""
    
    def __init__(self):
        self.token_analyzer = TokenAnalyzer()
        self.optimization_suggestions = []
    
    def analyze_prompt_effectiveness(
        self,
        prompt: str,
        response: LLMResponse,
        expected_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze prompt effectiveness
        
        Args:
            prompt: Input prompt
            response: LLM response
            expected_length: Expected response length
            
        Returns:
            Effectiveness analysis
        """
        analysis = {
            "prompt_length": len(prompt),
            "response_length": len(response.text),
            "token_efficiency": response.tokens_used["total_tokens"] / len(response.text) if response.text else 0,
            "response_time": response.response_time,
            "finish_reason": response.finish_reason,
            "suggestions": []
        }
        
        # Generate suggestions
        if len(prompt) > 1000:
            analysis["suggestions"].append("Consider shortening the prompt to reduce token usage")
        
        if response.finish_reason == "length":
            analysis["suggestions"].append("Response was truncated - consider increasing max_tokens")
        
        if response.response_time > 10:
            analysis["suggestions"].append("Response time is high - consider using a faster model")
        
        if expected_length and len(response.text) < expected_length * 0.5:
            analysis["suggestions"].append("Response is shorter than expected - consider adding more context to prompt")
        
        return analysis
    
    def generate_prompt_variations(self, base_prompt: str) -> List[str]:
        """
        Generate variations of a base prompt
        
        Args:
            base_prompt: Original prompt
            
        Returns:
            List of prompt variations
        """
        variations = []
        
        # Add system message variation
        variations.append(f"System: You are a helpful AI assistant.\n\nUser: {base_prompt}")
        
        # Add few-shot example variation
        variations.append(f"""Here's an example:

Input: "What is machine learning?"
Output: "Machine learning is a subset of AI that enables computers to learn from data."

Now, please answer:

{base_prompt}""")
        
        # Add role-based variation
        variations.append(f"You are an expert in this field. Please provide a detailed answer to: {base_prompt}")
        
        # Add step-by-step variation
        variations.append(f"Let's approach this step by step:\n\n{base_prompt}")
        
        # Add structured variation
        variations.append(f"""Please provide your response in the following format:

Question: {base_prompt}

Analysis:
[Your detailed analysis here]

Conclusion:
[Your conclusion here]""")
        
        return variations
    
    def optimize_prompt(
        self,
        prompt: str,
        target_length: Optional[int] = None,
        target_tokens: Optional[int] = None
    ) -> str:
        """
        Optimize prompt for better performance
        
        Args:
            prompt: Original prompt
            target_length: Target character length
            target_tokens: Target token count
            
        Returns:
            Optimized prompt
        """
        optimized = prompt
        
        # Remove unnecessary whitespace
        optimized = re.sub(r'\s+', ' ', optimized).strip()
        
        # Remove redundant phrases
        redundant_phrases = [
            r'please provide',
            r'kindly',
            r'if you could',
            r'would you mind',
            r'thank you in advance'
        ]
        
        for phrase in redundant_phrases:
            optimized = re.sub(phrase, '', optimized, flags=re.IGNORECASE)
        
        # Trim to target length if specified
        if target_length and len(optimized) > target_length:
            optimized = optimized[:target_length-3] + "..."
        
        # Trim to target tokens if specified
        if target_tokens:
            current_tokens = self.token_analyzer.count_tokens(optimized)
            if current_tokens > target_tokens:
                # Simple token reduction - this could be more sophisticated
                words = optimized.split()
                while self.token_analyzer.count_tokens(' '.join(words)) > target_tokens and words:
                    words.pop()
                optimized = ' '.join(words)
        
        return optimized

class LLMUtils:
    """Main class for LLM utilities"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.prompt_template = PromptTemplate()
        self.token_analyzer = TokenAnalyzer()
        self.llm_client = LLMClient(api_key)
        self.chain_of_thought = ChainOfThoughtAnalyzer()
        self.prompt_optimizer = PromptOptimizer()
    
    def generate_comprehensive_response(
        self,
        prompt: str,
        template_name: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate comprehensive LLM response with analysis
        
        Args:
            prompt: Input prompt or template name
            template_name: Template to use (if prompt is template name)
            variables: Variables for template (if using template)
            model: Model to use
            **kwargs: Additional LLM parameters
            
        Returns:
            Comprehensive response with analysis
        """
        try:
            # Generate prompt if using template
            if template_name:
                if variables is None:
                    variables = {}
                final_prompt = self.prompt_template.create_prompt(template_name, variables)
            else:
                final_prompt = prompt
            
            # Analyze prompt
            prompt_analysis = self.token_analyzer.analyze_tokens(final_prompt, model)
            
            # Generate response
            response = self.llm_client.generate_response(final_prompt, model, **kwargs)
            
            # Analyze response
            response_analysis = self.token_analyzer.analyze_tokens(response.text, model)
            chain_of_thought = self.chain_of_thought.extract_chain_of_thought(response.text)
            reasoning_structure = self.chain_of_thought.analyze_reasoning_structure(response.text)
            effectiveness = self.prompt_optimizer.analyze_prompt_effectiveness(final_prompt, response)
            
            # Calculate cost
            estimated_cost = self.token_analyzer.estimate_cost(
                response.tokens_used["prompt_tokens"],
                response.tokens_used["completion_tokens"],
                model
            )
            
            comprehensive_result = {
                "prompt": final_prompt,
                "response": response,
                "prompt_analysis": prompt_analysis,
                "response_analysis": response_analysis,
                "chain_of_thought": chain_of_thought,
                "reasoning_structure": reasoning_structure,
                "effectiveness": effectiveness,
                "estimated_cost": estimated_cost,
                "timestamp": datetime.now().isoformat()
            }
            
            return comprehensive_result
        
        except Exception as e:
            logger.error(f"Error generating comprehensive response: {str(e)}")
            raise 