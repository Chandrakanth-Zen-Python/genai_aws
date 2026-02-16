import json
import boto3
from datetime import datetime

class PromptManager:
    """Manage prompt versions with A/B testing"""
    
    def __init__(self, prompts_file='prompts.json'):
        with open(prompts_file, 'r') as f:
            self.prompts = json.load(f)
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    def get_prompt(self, version='latest'):
        """Get a specific prompt version"""
        if version == 'latest':
            # Get highest version number
            versions = sorted([p for p in self.prompts.keys()], reverse=True)
            return self.prompts[versions[0]]
        return self.prompts.get(version)
    
    def ab_test(self, query, version_a, version_b):
        """Run A/B test between two prompt versions"""
        import random
        
        # Randomly assign to A or B (50/50 split)
        selected_version = random.choice([version_a, version_b])
        prompt_data = self.prompts[selected_version]
        
        # Format full prompt
        full_prompt = f"<s>[INST] {prompt_data['prompt']}\n\nUser Question: {query} [/INST]"
        
        # Call Bedrock
        response = self.bedrock.invoke_model(
            modelId="mistral.mistral-large-2402-v1:0",
            body=json.dumps({
                "prompt": full_prompt,
                "max_tokens": 500
            })
        )
        
        result = json.loads(response['body'].read())
        
        return {
            'version_used': selected_version,
            'response': result['outputs'][0]['text'],
            'prompt_version': prompt_data['version']
        }
    
    def log_result(self, version, user_satisfaction, tokens_used):
        """Log results for analysis"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'version': version,
            'satisfaction': user_satisfaction,
            'tokens': tokens_used
        }
        
        # In production, write to DynamoDB or CloudWatch
        print(f"Logged: {json.dumps(log_entry, indent=2)}")

# Demo usage
if __name__ == "__main__":
    pm = PromptManager()
    
    # Example: A/B test two versions
    result = pm.ab_test(
        "How do I reset my password?",
        "customer_support_v2",
        "customer_support_v3"
    )
    
    print(f"Used version: {result['version_used']}")
    print(f"Response: {result['response']}")
    
    # Simulate user feedback
    pm.log_result(result['version_used'], user_satisfaction=4.5, tokens_used=115)


def llm_as_judge(question, answer):
    """Use LLM to evaluate response quality"""
    judge_prompt = f"""
    Rate the quality of this answer on a scale of 1-5:
    
    Question: {question}
    Answer: {answer}
    
    Criteria:
    - Relevance: Does it answer the question?
    - Accuracy: Are facts correct?
    - Completeness: Is anything missing?
    - Clarity: Is it easy to understand?
    
    Respond with only a number 1-5 and brief reason.
    """
    
    # Call Bedrock to judge
    # ... (implementation)
    
    return rating, reason