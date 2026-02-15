import boto3
import json
import numpy as np
from numpy.linalg import norm


client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)


def generate_embedding(input_text):
    
    # Set the model ID, e.g., Titan Text Embeddings V2.
    model_id = "amazon.titan-embed-text-v2:0"

    # Create the request for the model.
    native_request = {"inputText": input_text}

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

    # Decode the model's native response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the generated embedding and the input text token count.
    # embedding = np.array(model_response["embedding"])
    embedding = model_response["embedding"]

    input_token_count = model_response["inputTextTokenCount"]

    return embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


