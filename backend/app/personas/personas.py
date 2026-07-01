import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi4"

def call_ollama(prompt: str, system_prompt: str) -> dict:
    """Helper to call local Ollama instance and return parsed JSON."""
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "format": "json"
    }
    
    req = urllib.request.Request(
        OLLAMA_URL, 
        data=json.dumps(data).encode("utf-8"), 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
            response_text = result.get("response", "{}")
            
            try:
                # Try to parse the JSON returned by the model
                parsed = json.loads(response_text)
                return {
                    "summary": parsed.get("summary", "No summary provided."),
                    "key_points": parsed.get("key_points", []),
                    "confidence": parsed.get("confidence", "low")
                }
            except json.JSONDecodeError:
                # Fallback if model didn't return valid JSON despite the format flag
                return {
                    "summary": response_text[:200] + "...",
                    "key_points": ["Model failed to return structured JSON."],
                    "confidence": "low"
                }
                
    except Exception as e:
        return {
            "summary": f"Ollama connection error. Ensure it is running.",
            "key_points": [str(e)],
            "confidence": "low"
        }

def build_prompt(clause: str, risk_score: str, precedents: list) -> str:
    """Constructs the prompt combining the clause, risk, and precedents."""
    prompt = f"Clause to analyze:\n\"{clause}\"\n\nModel Risk Assessment: {risk_score.upper()}\n\n"
    
    if precedents:
        prompt += "Historical Precedents from similar contracts:\n"
        for i, p in enumerate(precedents):
            prompt += f"{i+1}. [{p.get('category', 'Unknown')}] {p.get('text', '')[:150]}...\n"
    else:
        prompt += "No historical precedents available.\n"
        
    prompt += "\nRespond ONLY with valid JSON using the keys 'summary', 'key_points', and 'confidence'."
    return prompt

def lawyer_take(clause: str, risk_score: str, precedents: list) -> dict:
    system = (
        "You are a sharp, loophole-hunting attorney. "
        "Your goal is to find exploitable wording, missing definitions, or edge cases that could be used against the client in court. "
        "Be extremely critical and technical. Output must be exactly in this JSON format: "
        '{"summary": "your short summary", "key_points": ["point 1", "point 2"], "confidence": "high/medium/low"}'
    )
    return call_ollama(build_prompt(clause, risk_score, precedents), system)

def optimist_take(clause: str, risk_score: str, precedents: list) -> dict:
    system = (
        "You are a pragmatic, deal-closing business advisor. "
        "Your goal is to highlight the best-case scenario, explain why this clause is standard market practice, "
        "and frame the risk as an acceptable business compromise. "
        "Keep it encouraging but grounded. Output must be exactly in this JSON format: "
        '{"summary": "your short summary", "key_points": ["point 1", "point 2"], "confidence": "high/medium/low"}'
    )
    return call_ollama(build_prompt(clause, risk_score, precedents), system)

def pessimist_take(clause: str, risk_score: str, precedents: list) -> dict:
    system = (
        "You are a worst-case scenario risk analyst. "
        "Your goal is to imagine catastrophic downstream impacts. What happens if the counterparty goes bankrupt? "
        "What happens if there is a global pandemic? Assume the absolute worst and outline the financial ruin this clause could cause. "
        "Be alarmist but logical based on the text. Output must be exactly in this JSON format: "
        '{"summary": "your short summary", "key_points": ["point 1", "point 2"], "confidence": "high/medium/low"}'
    )
    return call_ollama(build_prompt(clause, risk_score, precedents), system)

if __name__ == "__main__":
    # Test script
    test_clause = "In no event shall either party's aggregate liability arising out of or related to this agreement exceed the total amount paid by customer hereunder in the 12 months preceding the first incident out of which the liability arose."
    
    test_precedents = [
        {"category": "Limitation of Liability", "text": "The aggregate liability of either party... shall not exceed the amounts paid under this Agreement in the twelve (12) months prior to the event giving rise to the claim."},
        {"category": "Uncapped Liability", "text": "Notwithstanding anything to the contrary, neither party's liability for indemnification obligations shall be subject to any cap."}
    ]
    
    print("Testing Lawyer Take...")
    print(json.dumps(lawyer_take(test_clause, "high", test_precedents), indent=2))
    
    print("\nTesting Optimist Take...")
    print(json.dumps(optimist_take(test_clause, "high", test_precedents), indent=2))
    
    print("\nTesting Pessimist Take...")
    print(json.dumps(pessimist_take(test_clause, "high", test_precedents), indent=2))
