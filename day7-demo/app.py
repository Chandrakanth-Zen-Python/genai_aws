import streamlit as st
import boto3
import json

# Configure page
st.set_page_config(page_title="GenAI Chatbot", page_icon="🤖")

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    """Cache the Bedrock client to avoid recreating on every rerun"""
    return boto3.client('bedrock-runtime', region_name='us-east-1')

bedrock = get_bedrock_client()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title
st.title("🤖 GenAI Chatbot with Bedrock")
st.caption("Powered by Mistral Large")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare request for Bedrock
            body = json.dumps({
                "max_tokens": 1000,
                "messages": [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]
            })

            # Invoke Bedrock
            response = bedrock.invoke_model(
                modelId="mistral.mistral-large-3-675b-instruct",
                body=body
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            assistant_message = response_body['choices'][0]['message']['content']
            
            # Display and save response
            st.markdown(assistant_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message
            })

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.info("""
    This chatbot uses:
    - **Streamlit** for UI
    - **Amazon Bedrock** for AI
    - **Mistral Large** model
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()