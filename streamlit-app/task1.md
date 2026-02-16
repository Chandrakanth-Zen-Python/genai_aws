# Lab 1: Build Your Streamlit Chatbot

## Objective
Create a Streamlit chatbot that uses Amazon Bedrock with conversation history.

## Steps

1. Create a new directory: `mkdir lab1-streamlit && cd lab1-streamlit`

2. Create `requirements.txt`:
   - streamlit
   - boto3

3. Build `app.py` with:
   - Title and description
   - Chat message display
   - User input handling
   - Bedrock integration
   - Session state for history

4. Run: `streamlit run app.py`

## Bonus Challenges
- Add a system prompt input in the sidebar
- Add temperature slider (0.0-1.0)
- Display token count for each response
- Add file upload for document Q&A

## Success Criteria
✅ App runs without errors
✅ Messages display correctly
✅ Chat history persists during session
✅ Bedrock responses appear