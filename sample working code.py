# # Use the native inference API to send a text message to Amazon Titan Text.

# import boto3
# import json

# from botocore.exceptions import ClientError

# # Create a Bedrock Runtime client in the AWS Region of your choice.
# client = boto3.client("bedrock-runtime", region_name="us-east-1")

# # Set the model ID, e.g., Titan Text Premier.
# model_id = "amazon.titan-text-premier-v1:0"

# # Define the prompt for the model.
# prompt = "Describe the purpose of a 'hello world' program in one line."

# # Format the request payload using the model's native structure.
# native_request = {
#     "inputText": prompt,
#     "textGenerationConfig": {
#         "maxTokenCount": 512,
#         "temperature": 0.5,
#     },
# }

# # Convert the native request to JSON.
# request = json.dumps(native_request)

# try:
#     # Invoke the model with the request.
#     response = client.invoke_model(modelId=model_id, body=request)

# except (ClientError, Exception) as e:
#     print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
#     exit(1)

# # Decode the response body.
# model_response = json.loads(response["body"].read())

# # Extract and print the response text.
# response_text = model_response["results"][0]["outputText"]
# print(response_text)


# Use the native inference API to send a text message to Anthropic Claude.

import boto3
import json

from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Claude 3 Haiku.
model_id = "anthropic.claude-sonnet-4-5-20250929-v1:0"

# Define the prompt for the model.
prompt = "Describe the purpose of a 'hello world' program in one line."

# Format the request payload using the model's native structure.
native_request = {
    # "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.5,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ],
}

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Decode the response body.
model_response = json.loads(response["body"].read())

# Extract and print the response text.
response_text = model_response["content"][0]["text"]
print(response_text)



# # Use the native inference API to send a text message to Meta Llama 3.

# import boto3
# import json

# from botocore.exceptions import ClientError

# # Create a Bedrock Runtime client in the AWS Region of your choice.
# client = boto3.client("bedrock-runtime", region_name="us-west-2")

# # Set the model ID, e.g., Llama 3 70b Instruct.
# model_id = "meta.llama3-70b-instruct-v1:0"

# # Define the prompt for the model.
# prompt = "Describe the purpose of a 'hello world' program in one line."

# # Embed the prompt in Llama 3's instruction format.
# formatted_prompt = f"""
# <|begin_of_text|><|start_header_id|>user<|end_header_id|>
# {prompt}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """

# # Format the request payload using the model's native structure.
# native_request = {
#     "prompt": formatted_prompt,
#     "max_gen_len": 512,
#     "temperature": 0.5,
# }

# # Convert the native request to JSON.
# request = json.dumps(native_request)

# try:
#     # Invoke the model with the request.
#     response = client.invoke_model(modelId=model_id, body=request)

# except (ClientError, Exception) as e:
#     print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
#     exit(1)

# # Decode the response body.
# model_response = json.loads(response["body"].read())

# # Extract and print the response text.
# response_text = model_response["generation"]
# print(response_text)


# # Use the native inference API to send a text message to Mistral.

import boto3
import json
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Mistral Large.
model_id = "mistral.mistral-large-2402-v1:0"

# Define the prompt for the model.
prompt = "Describe the purpose of a 'hello world' program in one line."

# Embed the prompt in Mistral's instruction format.
formatted_prompt = f"<s>[INST] {prompt} [/INST]"

# Format the request payload using the model's native structure.
native_request = {
    "prompt": formatted_prompt,
    "max_tokens": 512,
    "temperature": 0.5,
}

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Decode the response body.
model_response = json.loads(response["body"].read())

# Extract and print the response text.
response_text = model_response["outputs"][0]["text"]
print(response_text)



# Generate and print an embedding with Amazon Titan Text Embeddings V2.

import boto3
import json

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Text Embeddings V2.
model_id = "amazon.titan-embed-text-v2:0"

# The text to convert to an embedding.
input_text = "Please recommend books with a theme similar to the movie 'Inception'."

# Create the request for the model.
native_request = {"inputText": input_text}

# Convert the native request to JSON.
request = json.dumps(native_request)

# Invoke the model with the request.
response = client.invoke_model(modelId=model_id, body=request)

# Decode the model's native response body.
model_response = json.loads(response["body"].read())

# Extract and print the generated embedding and the input text token count.
embedding = model_response["embedding"]
input_token_count = model_response["inputTextTokenCount"]

print("\nYour input:")
print(input_text)
print(f"Number of input tokens: {input_token_count}")
print(f"Size of the generated embedding: {len(embedding)}")
print("Embedding:")
print(embedding)






