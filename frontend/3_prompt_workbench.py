"""
Prompt Workbench - Streamlit page for interactive prompt engineering
Live prompt editor, LLM API calls, and token-level insights visualization
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import requests
import json
import time
import re

# Page configuration
st.set_page_config(
    page_title="Prompt Workbench - ML Playground",
    page_icon="💬",
    layout="wide"
)

def get_prompt_templates() -> Dict[str, Dict[str, Any]]:
    """Get predefined prompt templates for different use cases"""
    return {
        "Text Classification": {
            "description": "Classify text into predefined categories",
            "template": """Classify the following text into one of these categories: {categories}

Text: {input_text}

Classification:""",
            "variables": ["categories", "input_text"],
            "example": {
                "categories": "positive, negative, neutral",
                "input_text": "I love this product! It's amazing."
            }
        },
        "Text Generation": {
            "description": "Generate creative text based on a prompt",
            "template": """Write a creative story based on the following prompt:

Prompt: {prompt}

Story:""",
            "variables": ["prompt"],
            "example": {
                "prompt": "A robot discovers emotions for the first time"
            }
        },
        "Question Answering": {
            "description": "Answer questions based on given context",
            "template": """Answer the following question based on the provided context:

Context: {context}

Question: {question}

Answer:""",
            "variables": ["context", "question"],
            "example": {
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "question": "What is machine learning?"
            }
        },
        "Code Generation": {
            "description": "Generate code based on requirements",
            "template": """Write Python code for the following requirement:

Requirement: {requirement}

Code:""",
            "variables": ["requirement"],
            "example": {
                "requirement": "Create a function to calculate the factorial of a number"
            }
        },
        "Custom": {
            "description": "Create your own custom prompt template",
            "template": "{custom_prompt}",
            "variables": ["custom_prompt"],
            "example": {
                "custom_prompt": "Your custom prompt here..."
            }
        }
    }

def get_available_models() -> List[str]:
    """Get available LLM models"""
    return [
        "gpt-3.5-turbo",
        "gpt-4",
        "claude-3-sonnet",
        "claude-3-opus",
        "llama-2-7b",
        "llama-2-13b"
    ]

def create_prompt_editor() -> str:
    """Create the prompt editor interface"""
    st.subheader("✏️ Prompt Editor")
    
    templates = get_prompt_templates()
    
    # Template selection
    template_name = st.selectbox(
        "Choose a prompt template:",
        list(templates.keys()),
        help="Select a predefined template or create a custom one"
    )
    
    template = templates[template_name]
    st.info(f"**{template_name}**: {template['description']}")
    
    # Template variables
    variables = template["variables"]
    variable_values = {}
    
    if template_name == "Custom":
        # Custom prompt editor
        prompt_text = st.text_area(
            "Enter your custom prompt:",
            value=template["example"]["custom_prompt"],
            height=200,
            help="Write your custom prompt here. Use {variable_name} for dynamic parts."
        )
        return prompt_text
    else:
        # Template-based editor
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show template with placeholders
            st.text_area(
                "Template Preview:",
                value=template["template"],
                height=150,
                disabled=True
            )
        
        with col2:
            # Variable inputs
            st.write("**Template Variables:**")
            for var in variables:
                if var in template["example"]:
                    default_value = template["example"][var]
                else:
                    default_value = ""
                
                variable_values[var] = st.text_input(
                    f"{var}:",
                    value=default_value,
                    key=f"var_{var}"
                )
        
        # Build the final prompt
        try:
            final_prompt = template["template"].format(**variable_values)
            return final_prompt
        except KeyError as e:
            st.error(f"Missing variable: {e}")
            return template["template"]

def simulate_llm_response(prompt: str, model: str) -> Dict[str, Any]:
    """Simulate LLM API response"""
    # TODO: Replace with actual LLM API call
    time.sleep(1)  # Simulate API delay
    
    # Mock response
    response = {
        "text": f"This is a simulated response from {model} for the prompt: {prompt[:50]}...",
        "tokens": {
            "input_tokens": len(prompt.split()),
            "output_tokens": 25,
            "total_tokens": len(prompt.split()) + 25
        },
        "model": model,
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 25,
            "total_tokens": len(prompt.split()) + 25
        }
    }
    
    return response

def create_token_visualization(prompt: str, response: Dict[str, Any]):
    """Create token-level visualization"""
    st.subheader("🔍 Token Analysis")
    
    # Token usage metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Input Tokens", response["tokens"]["input_tokens"])
    with col2:
        st.metric("Output Tokens", response["tokens"]["output_tokens"])
    with col3:
        st.metric("Total Tokens", response["tokens"]["total_tokens"])
    with col4:
        st.metric("Model", response["model"])
    
    # Token distribution chart
    fig = go.Figure(data=[
        go.Bar(name='Input Tokens', x=['Input'], y=[response["tokens"]["input_tokens"]], marker_color='blue'),
        go.Bar(name='Output Tokens', x=['Output'], y=[response["tokens"]["output_tokens"]], marker_color='green')
    ])
    
    fig.update_layout(
        title="Token Usage Distribution",
        barmode='group',
        yaxis_title="Number of Tokens"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Word cloud or token frequency (simplified)
    st.subheader("📝 Token Breakdown")
    
    # Split prompt into words for analysis
    words = prompt.split()
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Show most common words
    if word_freq:
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig = px.bar(
            x=[word for word, freq in sorted_words],
            y=[freq for word, freq in sorted_words],
            title="Most Common Words in Prompt"
        )
        fig.update_layout(xaxis_title="Words", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

def create_prompt_variations(prompt: str) -> List[str]:
    """Generate prompt variations for testing"""
    variations = []
    
    # Add system message
    variations.append(f"System: You are a helpful AI assistant.\n\nUser: {prompt}")
    
    # Add few-shot example
    variations.append(f"""Here's an example:

Input: "What is machine learning?"
Output: "Machine learning is a subset of AI that enables computers to learn from data."

Now, please answer:

{prompt}""")
    
    # Add role-based prompt
    variations.append(f"You are an expert in this field. Please provide a detailed answer to: {prompt}")
    
    # Add step-by-step prompt
    variations.append(f"Let's approach this step by step:\n\n{prompt}")
    
    return variations

def main():
    """Main function for the Prompt Workbench page"""
    st.title("💬 Prompt Workbench")
    st.markdown("Experiment with prompts, test LLM responses, and analyze token usage.")
    
    # Model selection
    st.header("🤖 Select LLM Model")
    models = get_available_models()
    
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Choose LLM model:", models, index=0)
    with col2:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
    
    # Prompt editor
    prompt = create_prompt_editor()
    
    if prompt:
        st.subheader("📝 Final Prompt")
        st.text_area("Your prompt:", value=prompt, height=100, disabled=True)
        
        # Generate response
        if st.button("🚀 Generate Response", type="primary"):
            with st.spinner("Generating response..."):
                response = simulate_llm_response(prompt, model)
            
            # Display response
            st.subheader("🤖 LLM Response")
            st.text_area("Response:", value=response["text"], height=150, disabled=True)
            
            # Token analysis
            create_token_visualization(prompt, response)
            
            # Response metrics
            st.subheader("📊 Response Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Response Length", len(response["text"]))
            with col2:
                st.metric("Words", len(response["text"].split()))
            with col3:
                st.metric("Characters", len(response["text"]))
        
        # Prompt variations
        st.subheader("🔄 Prompt Variations")
        if st.button("Generate Variations"):
            variations = create_prompt_variations(prompt)
            
            for i, variation in enumerate(variations, 1):
                with st.expander(f"Variation {i}"):
                    st.text_area(f"Variation {i}:", value=variation, height=100, disabled=True)
                    
                    if st.button(f"Test Variation {i}", key=f"test_var_{i}"):
                        with st.spinner(f"Testing variation {i}..."):
                            var_response = simulate_llm_response(variation, model)
                        
                        st.text_area(f"Response {i}:", value=var_response["text"], height=100, disabled=True)
        
        # Prompt history
        if "prompt_history" not in st.session_state:
            st.session_state.prompt_history = []
        
        if st.button("💾 Save to History"):
            st.session_state.prompt_history.append({
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "timestamp": time.time()
            })
            st.success("Prompt saved to history!")
        
        # Show history
        if st.session_state.prompt_history:
            st.subheader("📚 Prompt History")
            for i, entry in enumerate(reversed(st.session_state.prompt_history)):
                with st.expander(f"Prompt {len(st.session_state.prompt_history) - i}"):
                    st.write(f"**Model:** {entry['model']}")
                    st.write(f"**Temperature:** {entry['temperature']}")
                    st.write(f"**Prompt:** {entry['prompt']}")
                    st.write(f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")
    
    else:
        st.info("👆 Please create a prompt to begin testing")

if __name__ == "__main__":
    main() 