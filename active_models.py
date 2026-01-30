import boto3
from typing import Dict, List

def list_and_select_bedrock_models() -> Dict[str, str]:
    """
    List active Bedrock models and select one active model each for 
    Titan, Claude, Llama, and Mistral.
    
    Returns:
        Dict[str, str]: Dictionary with model families as keys and selected model IDs as values
    """
    bedrock = boto3.client('bedrock', region_name='us-east-1')
    
    # Model family prefixes to search for
    model_families = {
        'titan': [],
        'claude': [],
        'llama': [],
        'mistral': []
    }
    
    try:
        # List all available models
        response = bedrock.list_foundation_models()
        models = response.get('modelSummaries', [])
        
        # Categorize models by family
        for model in models:
            model_id = model['modelId']
            model_id_lower = model_id.lower()
            
            if 'titan' in model_id_lower:
                model_families['titan'].append(model_id)
            elif 'claude' in model_id_lower:
                model_families['claude'].append(model_id)
            elif 'llama' in model_id_lower:
                model_families['llama'].append(model_id)
            elif 'mistral' in model_id_lower:
                model_families['mistral'].append(model_id)
        
        # Select the first active model from each family
        selected_models = {}
        for family, models_list in model_families.items():
            if models_list:
                selected_models[family] = models_list[0]
                print(f"{family.capitalize()}: {models_list[0]}")
            else:
                print(f"{family.capitalize()}: No models found")
        
        return selected_models
    
    except Exception as e:
        print(f"Error listing Bedrock models: {str(e)}")
        return {}


if __name__ == "__main__":
    selected = list_and_select_bedrock_models()
    print("\nSelected Models:")
    print(selected)