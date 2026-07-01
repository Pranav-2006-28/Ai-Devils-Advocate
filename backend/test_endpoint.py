import asyncio
import os
from fastapi.testclient import TestClient
import app.main

def test_analyze():
    print("Testing /analyze endpoint with a dummy PDF...")
    client = TestClient(app.main.app)
    
    test_text = "This is a contract. Either party may terminate this agreement for convenience upon 30 days written notice. Furthermore, the aggregate liability of either party shall not exceed the amounts paid under this Agreement in the twelve (12) months prior to the event giving rise to the claim."
    
    app.main.extract_text_from_pdf = lambda x: test_text
    
    with TestClient(app.main.app) as client:
        response = client.post(
            "/analyze",
            files={"file": ("test.pdf", b"dummy_bytes", "application/pdf")}
        )
        
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Risk Score: {data['risk_score']}")
        print(f"High Risks: {data['high_risks']}")
        for finding in data['findings']:
            if finding['found']:
                print(f"\n--- FOUND RISK: {finding['label']} ---")
                print(f"Answer: {finding['answer']}")
                print(f"Precedents retrieved: {len(finding.get('precedents', []))}")
                if finding.get('lawyer_take'):
                    print(f"Lawyer Summary: {finding['lawyer_take'].get('summary', 'Error')}")
                if finding.get('optimist_take'):
                    print(f"Optimist Summary: {finding['optimist_take'].get('summary', 'Error')}")
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    test_analyze()
