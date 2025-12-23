"""Test script to run the full pipeline with mock data."""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import AgenticPipeline

# Load environment
load_dotenv()


def print_divider(char="=", width=60):
    print(char * width)


def print_result(result):
    """Pretty print a pipeline result."""
    print(f"\n📋 Session: {result.session_id[:8]}...")
    print(f"🎫 Ticket: {result.ticket_id}")
    print(f"📊 Confidence: {result.confidence}%")
    print(f"🔄 Turn: {result.turn_count}")
    print(f"🚨 Escalated: {'Yes' if result.escalated else 'No'}")
    
    print("\n📝 Response:")
    print_divider("-")
    print(result.response)
    print_divider("-")


def run_test_scenario_1():
    """Scenario 1: Simple query resolved in first turn."""
    print("\n" + "="*60)
    print("🧪 SCENARIO 1: First-Turn Resolution")
    print("="*60)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    pipeline = AgenticPipeline(
        api_key=api_key,
        sessions_db="./test_sessions.db"
    )
    
    # Submit ticket
    result = pipeline.process_ticket(
        ticket_text="Bonjour, comment puis-je annuler mon abonnement Doxa ?",
        ticket_id="TKT-SCENARIO-001"
    )
    
    print_result(result)
    
    # Simulate satisfied feedback
    if result.needs_feedback and not result.escalated:
        print("\n✅ Simulating SATISFIED feedback...")
        result2, is_complete = pipeline.handle_feedback(
            session_id=result.session_id,
            satisfied=True
        )
        print(f"   Complete: {is_complete}")
        print(f"   Response: {result2.response}")
    
    return result


def run_test_scenario_2():
    """Scenario 2: Multi-turn conversation."""
    print("\n" + "="*60)
    print("🧪 SCENARIO 2: Multi-Turn Conversation")
    print("="*60)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    pipeline = AgenticPipeline(
        api_key=api_key,
        sessions_db="./test_sessions.db"
    )
    
    # Turn 1
    print("\n--- Turn 1 ---")
    result = pipeline.process_ticket(
        ticket_text="Je voudrais savoir les tarifs de Doxa",
        ticket_id="TKT-SCENARIO-002"
    )
    print_result(result)
    
    # Turn 2 - Not satisfied, ask follow-up
    if result.needs_feedback and not result.escalated:
        print("\n❌ Simulating NOT SATISFIED feedback + follow-up...")
        result2, is_complete = pipeline.handle_feedback(
            session_id=result.session_id,
            satisfied=False,
            reason_tags=["reponse_incomplete"],
            followup_message="Et pour l'offre entreprise, c'est combien ?"
        )
        print_result(result2)
        
        # Turn 3 - Now satisfied
        if result2.needs_feedback and not result2.escalated:
            print("\n✅ Simulating SATISFIED feedback...")
            result3, is_complete = pipeline.handle_feedback(
                session_id=result2.session_id,
                satisfied=True
            )
            print(f"   Complete: {is_complete}")
    
    return result


def run_test_scenario_3():
    """Scenario 3: Escalation due to off-topic query."""
    print("\n" + "="*60)
    print("🧪 SCENARIO 3: Low Confidence → Escalation")
    print("="*60)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    pipeline = AgenticPipeline(
        api_key=api_key,
        sessions_db="./test_sessions.db"
    )
    
    # Submit off-topic ticket
    result = pipeline.process_ticket(
        ticket_text="Quelle est la recette de la pizza margherita ?",
        ticket_id="TKT-SCENARIO-003"
    )
    
    print_result(result)
    
    if result.escalated:
        print("\n✅ Ticket correctly escalated due to low confidence")
    
    return result


def run_test_scenario_4():
    """Scenario 4: Max turns reached → Escalation."""
    print("\n" + "="*60)
    print("🧪 SCENARIO 4: Max Turns → Escalation")
    print("="*60)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    pipeline = AgenticPipeline(
        api_key=api_key,
        sessions_db="./test_sessions.db"
    )
    
    # Turn 1
    result = pipeline.process_ticket(
        ticket_text="Comment fonctionne le service Doxa ?",
        ticket_id="TKT-SCENARIO-004"
    )
    session_id = result.session_id
    
    # Keep saying not satisfied until escalation
    for turn in range(3):
        if result.escalated:
            break
        if result.needs_feedback:
            print(f"\n--- Feedback Turn {turn + 1}: Not satisfied ---")
            result, is_complete = pipeline.handle_feedback(
                session_id=session_id,
                satisfied=False,
                reason_tags=["pas_clair"],
                followup_message="Je ne comprends toujours pas..."
            )
            if is_complete:
                break
    
    print_result(result)
    
    if result.escalated:
        print("\n✅ Ticket correctly escalated after max turns")
    
    return result


def main():
    """Run all test scenarios."""
    print("\n" + "="*60)
    print("🚀 AGENTIC PIPELINE TEST SUITE")
    print("="*60)
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not found in .env file")
        print("   Please add: MISTRAL_API_KEY=your_key_here")
        return
    
    # Clean up test DB
    test_db = "./test_sessions.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    print(f"\n📦 Running with Mistral API")
    print(f"📂 Database: ./db")
    
    # Run scenarios
    try:
        run_test_scenario_1()
        run_test_scenario_2()
        run_test_scenario_3()
        run_test_scenario_4()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.remove(test_db)


if __name__ == "__main__":
    main()
