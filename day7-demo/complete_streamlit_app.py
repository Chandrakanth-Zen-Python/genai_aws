"""
Streamlit GenAI Chatbot - Complete Solution (Mistral Version)
Lab 1 for Day 7: GenAI Application Development
"""

import streamlit as st
import boto3
import json

# Configure page
st.set_page_config(
    page_title="GenAI Chatbot with Bedrock",
    page_icon="🤖",
    layout="wide"
)

# Initialize Bedrock client (cached to avoid recreating)
@st.cache_resource
def get_bedrock_client():
    """Create and cache Bedrock runtime client"""
    return boto3.client('bedrock-runtime', region_name='us-east-1')

bedrock = get_bedrock_client()

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper function to format conversation for Mistral
def format_conversation_for_mistral(messages):
    """
    Format conversation history for Mistral's prompt template
    Mistral uses: <s>[INST] user message [/INST] assistant response </s>
    """
    if not messages:
        return ""
    
    formatted = "<s>"
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"[INST] {msg['content']} [/INST]"
        else:
            formatted += f" {msg['content']} </s><s>"
    
    # Remove trailing <s> if present
    return formatted.rstrip("<s>")

# App header
st.title("🤖 GenAI Chatbot with Amazon Bedrock")
st.caption("Powered by Mistral Large 2 via Amazon Bedrock")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Format the entire conversation for Mistral
                    formatted_prompt = format_conversation_for_mistral(st.session_state.messages)
                    
                    # Prepare request for Bedrock (Mistral format)
                    body = json.dumps({
                        "prompt": formatted_prompt,
                        "max_tokens": 1000,
                        "temperature": st.session_state.get("temperature", 0.7),
                        "top_p": 0.9,
                        "top_k": 50
                    })
                    
                    # Invoke Bedrock with Mistral model
                    response = bedrock.invoke_model(
                        modelId="mistral.mistral-large-2402-v1:0",
                        body=body
                    )
                    
                    # Parse Mistral response
                    response_body = json.loads(response['body'].read())
                    assistant_message = response_body['outputs'][0]['text'].strip()
                    
                    # Display response
                    st.markdown(assistant_message)
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("""
                    Make sure you have:
                    - AWS credentials configured
                    - Mistral model access enabled in Bedrock console
                    - Correct region (us-east-1)
                    """)

# Sidebar controls
with col2:
    st.header("⚙️ Settings")
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    st.session_state.temperature = temperature
    
    # Model info
    with st.expander("Model Info"):
        st.info("""
        **Mistral Large 2**
        - Size: 123B parameters
        - Context: 32K tokens
        - Languages: Multilingual
        - Pricing: $4/1M input, $12/1M output
        """)
    
    # Chat controls
    st.divider()
    
    col_clear, col_export = st.columns(2)
    
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col_export:
        if st.session_state.messages:
            chat_export = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                "💾 Export",
                chat_export,
                "chat_history.json",
                "application/json",
                use_container_width=True
            )
    
    # Statistics
    st.divider()
    st.subheader("📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    
    if st.session_state.messages:
        total_chars = sum(len(msg["content"]) for msg in st.session_state.messages)
        st.metric("Total Characters", f"{total_chars:,}")
        
        # Estimate tokens (rough: ~4 chars per token)
        estimated_tokens = total_chars // 4
        st.metric("Est. Tokens", f"{estimated_tokens:,}")
    
    # Information
    st.divider()
    st.info("""
    **About this app:**
    
    - UI: Streamlit
    - AI: Amazon Bedrock
    - Model: Mistral Large 2
    
    **Features:**
    - Multi-turn conversations
    - Adjustable temperature
    - Export chat logs
    """)

# Footer
st.divider()
st.caption("Built with ❤️ for AWS GenAI Course • Day 7 Lab 1 • Mistral Version")