from models import Ticket
from query_analyzer import analyze_ticket
from solution_finder import find_solution
from evaluator import evaluate_solution
from response_composer import compose_response
from security_utils import scrub_text
import json

def run_ticket_test(ticket_obj):
    print(f"\n{'='*60}")
    print(f"TESTING TICKET: {ticket_obj.subject}")
    print(f"{'='*60}")
    print(f"Description: {ticket_obj.description}")
    
    # 1. Redact PII
    ticket_obj.description = scrub_text(ticket_obj.description)
    
    # 2. Analyze
    print("\n[1] Analyzing...")
    analysis = analyze_ticket(ticket_obj)
    print(f"    Sentiment: {analysis.sentiment}")
    print(f"    Language: {analysis.language}")
    print(f"    Keywords: {analysis.keywords}")
    
    # 3. Retrieve
    print("\n[2] Retrieving...")
    retrieval = find_solution(analysis)
    print(f"    Retrieved {len(retrieval.documents)} documents.")
    
    # 4. Evaluate
    print("\n[3] Evaluating...")
    evaluation = evaluate_solution(analysis, retrieval)
    context_data = json.loads(evaluation.context[0].strip("`").replace("json", "").strip())
    confidence = context_data.get("confidence_score")
    reasoning = context_data.get("reasoning")
    print(f"    Confidence: {confidence}")
    print(f"    Reasoning: {reasoning}")
    
    # 5. Respond
    if confidence >= 0.8: # High confidence (Solution found OR Off-topic)
        print("\n[4] Generating Response...")
        response = compose_response(analysis, retrieval, confidence, reasoning)
        print(f"\n--- RESPONSE ({analysis.language}) ---\n{response}\n----------------------------------")
        
        if "Off-topic" in reasoning or "Refusal" in reasoning or "recipe" in ticket_obj.description:
             if "cook" in response.lower() or "recipe" in response.lower() or "help" in response.lower():
                 print("✅ VERIFIED: Off-topic query handled correctly.")
    else:
        print("\n[4] Confidence too low. Escalated.")

# --- SCENARIOS ---

# 1. Hallucination Test: Fake Feature
ticket_fake = Ticket(
    id="TEST1",
    subject="Quantum Sync Not Working",
    category="Technical",
    description="I'm trying to use the new Quantum Sync feature to teleport my files to the cloud instantly, but it says 'Flux Capacitor low'. How do I refill it?",
    client_id="C1",
    timestamp="2024-06-16"
)

# 2. Irrelevant Topic
ticket_irrelevant = Ticket(
    id="TEST2",
    subject="Pizza Recipe",
    category="General",
    description="Can you send me a good recipe for a pepperoni pizza? I'm really hungry.",
    client_id="C2",
    timestamp="2024-06-16"
)

# 3. Multilingual Test (French)
ticket_french = Ticket(
    id="TEST3",
    subject="Problème de connexion",
    category="Account",
    description="Je ne peux pas me connecter à mon compte. J'ai oublié mon mot de passe et je ne reçois pas l'email de réinitialisation. Aidez-moi svp.",
    client_id="C3",
    timestamp="2024-06-16"
)

# Run All
# Run All
# run_ticket_test(ticket_fake)
run_ticket_test(ticket_irrelevant)
# run_ticket_test(ticket_french)
