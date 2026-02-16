"""
AWS Lambda GenAI Handler - Complete Solution (Mistral Version)
Lab 3 for Day 7: GenAI Application Development

This function integrates with API Gateway to provide a serverless
REST API for Amazon Bedrock chat completions using Mistral Large 2.
"""

import json
import boto3
import os
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Bedrock client (outside handler for connection reuse)
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Configuration
MODEL_ID = "mistral.mistral-large-2402-v1:0"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7

def format_prompt_for_mistral(message: str) -> str:
    """
    Format a single message for Mistral's prompt template
    Mistral expects: <s>[INST] user message [/INST]
    """
    return f"<s>[INST] {message} [/INST]"

def lambda_handler(event, context):
    """
    Main Lambda handler function
    
    Expected event format from API Gateway:
    {
        "body": "{\"message\": \"user input\"}",
        "headers": {...},
        "requestContext": {...}
    }
    
    Returns API Gateway proxy response:
    {
        "statusCode": 200,
        "headers": {...},
        "body": "{\"response\": \"...\"}"
    }
    """
    
    # Log incoming request
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        elif isinstance(event.get('body'), dict):
            body = event['body']
        elif 'message' in event:
            # Non-proxy integration: event IS the request body
            body = event
        else:
            body = {}
        
        # Extract parameters
        user_message = body.get('message', '').strip()
        temperature = body.get('temperature', DEFAULT_TEMPERATURE)
        max_tokens = body.get('max_tokens', DEFAULT_MAX_TOKENS)
        
        # Validate input
        if not user_message:
            logger.warning("Empty message received")
            return create_response(
                status_code=400,
                body={"error": "Message is required and cannot be empty"}
            )
        
        if not 0.0 <= temperature <= 1.0:
            return create_response(
                status_code=400,
                body={"error": "Temperature must be between 0.0 and 1.0"}
            )
        
        if not 1 <= max_tokens <= 4096:
            return create_response(
                status_code=400,
                body={"error": "max_tokens must be between 1 and 4096"}
            )
        
        logger.info(f"Processing message (length: {len(user_message)})")
        
        # Format prompt for Mistral
        formatted_prompt = format_prompt_for_mistral(user_message)
        
        # Prepare Bedrock request (Mistral format)
        bedrock_body = json.dumps({
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50
        })
        
        # Invoke Bedrock with Mistral model
        logger.info(f"Invoking Bedrock model: {MODEL_ID}")
        bedrock_response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=bedrock_body
        )
        
        # Parse Mistral response
        response_body = json.loads(bedrock_response['body'].read())
        ai_response = response_body['outputs'][0]['text'].strip()
        
        logger.info(f"Response generated successfully")
        
        # Return success response
        return create_response(
            status_code=200,
            body={
                "response": ai_response,
                "model": "mistral-large-2",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {str(e)}")
        return create_response(
            status_code=400,
            body={"error": "Invalid JSON format", "details": str(e)}
        )
    
    except bedrock.exceptions.ValidationException as e:
        logger.error(f"Bedrock validation error: {str(e)}")
        return create_response(
            status_code=400,
            body={"error": "Invalid request to Bedrock", "details": str(e)}
        )
    
    except bedrock.exceptions.ThrottlingException as e:
        logger.error(f"Bedrock throttling: {str(e)}")
        return create_response(
            status_code=429,
            body={"error": "Too many requests. Please try again later."}
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return create_response(
            status_code=500,
            body={"error": "Internal server error", "details": str(e)}
        )

def create_response(status_code: int, body: dict) -> dict:
    """
    Create a properly formatted API Gateway response
    
    Args:
        status_code: HTTP status code
        body: Response body (will be JSON-encoded)
    
    Returns:
        API Gateway proxy response dictionary
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # Configure for production
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key",
            "Access-Control-Allow-Methods": "POST,OPTIONS"
        },
        "body": json.dumps(body)
    }

# For local testing
if __name__ == "__main__":
    # Test event simulating API Gateway
    test_event = {
        "body": json.dumps({
            "message": "What is Data Engineering?",
            "temperature": 0.7,
            "max_tokens": 500
        })
    }
    
    # Mock context
    class MockContext:
        request_id = "test-request-id"
        function_name = "genai-chatbot-mistral"
        memory_limit_in_mb = 512
    
    # Run handler
    response = lambda_handler(test_event, MockContext())
    print(json.dumps(json.loads(response['body']), indent=2))