# Query Complexity Assessment:
# ├─ Simple (FAQ, greetings) → Mistral 7B ($0.15/$0.20 per 1M tokens)
# ├─ Moderate (explanations) → Mistral Large ($4/$12 per 1M tokens)
# └─ Complex (reasoning, code) → Claude Opus ($15/$75 per 1M tokens)


def route_to_model(query):
    """Smart routing based on complexity"""
    
    # Simple heuristics
    simple_keywords = ['hours', 'location', 'price', 'hello', 'thanks']
    complex_keywords = ['analyze', 'compare', 'recommend', 'explain why']
    
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in simple_keywords):
        return "mistral-7b"
    elif any(kw in query_lower for kw in complex_keywords):
        return "mistral-large"
    elif len(query.split()) < 10:
        return "mistral-7b"  # Short queries are usually simple
    else:
        return "mistral-large"  # Default to moderate