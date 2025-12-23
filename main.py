import json
from models import Ticket, AnalysisResult
from query_analyzer import analyze_ticket
from solution_finder import find_solution
from evaluator import evaluate_solution
from response_composer import compose_response
from security_utils import scrub_text



# Import your existing RAG pipeline
from rag_pipeline import RAGPipeline

    
# Mock Ticket for demonstration, with long description
mock_ticket = Ticket(
    id="TICKET123",
    subject="Unable to access account",
    category="Account Issues",
    description=(
        "I have been trying to log into my account for the past few days, but I keep getting an error message saying "
        "'Account not found'. I have checked my username and password multiple times, and I'm sure they are correct. "
        "I even tried resetting my password, but I never received the reset email. This issue is causing me a lot of "
        "inconvenience as I need to access my account urgently for work purposes. Please help me resolve this issue as "
        "soon as possible. My phone number is 555-0199 and email is bob.smith@example.com if you need to reach me. Thank you."
    ),
    client_id="CLIENT456",
    timestamp="2024-06-15T10:30:00Z",
)

analysis_result = analyze_ticket(mock_ticket)
retrieval_result = find_solution(analysis_result)
evaluation = evaluate_solution(analysis_result, retrieval_result)

# Extract and parse the confidence score from the JSON string in the context
context_data = json.loads(evaluation.context[0].strip("`").replace("json", "").strip())
confidence_score = context_data.get("confidence_score")
reasoning = context_data.get("reasoning")
print(f"Confidence Score: {confidence_score}")
print(f"Reasoning: {reasoning}")

if confidence_score >= 0.6:
    print("\n--- Generating Response ---\n")
    final_response = compose_response(
        analysis=analysis_result,
        retrieval=retrieval_result,
        confidence=confidence_score,
        reasoning=reasoning
    )
    print(final_response)
else:
    print("\nConfidence too low. Escalating to human agent.")