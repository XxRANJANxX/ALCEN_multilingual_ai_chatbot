import json


def parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM reply (tolerates code fences / chatter)."""
    
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data
