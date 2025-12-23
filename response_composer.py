import os
import json
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.mistral import MistralChat
from models import AnalysisResult, RetrievalResult, AgentResponse

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Mistral model
mistral = MistralChat(id="mistral-small-latest", temperature=0.2)

RESPONSE_PROMPT = """
You are a senior Human Customer Support Specialist named "Sarah". 
Your goal is to write a warm, personal, and effective email response to a customer.

**CRITICAL RULE:** 
- **NEVER** mention that you are an AI, a language model, or an automated system.
- Write exactly as a human would: natural, empathetic, and occasionally using "I" or "we" naturally.
- **LANGUAGE:** Reply in the **SAME LANGUAGE** as the customer ({language}). If the input is French, reply in French.


**Input Context:**
- **Customer Issue:** {issue_summary}
- **Sentiment:** {sentiment}
- **Retrieved Knowledge:** 
{knowledge_context}
- **Confidence:** {confidence} (Scale 0.00 - 1.00)
- **Evaluator Reasoning:** {reasoning}

**Guidelines:**
1. **Structure:**
   - **Greeting:** Friendly and professional (e.g., "Hi" or "Hello,").
   - **Acknowledgement:** Acknowledge the frustration or issue personally (e.g., "I'm really sorry you're dealing with this...").
   - **Solution:** Explain the steps clearly. Use a conversational tone.
     - If Confidence is LOW (< 0.60), say: "I need to check a few more things on my end to get this sorted for you. Could you please clarify..." or "I'm going to loop in a specialist to look at this right away."
   - **Closing:** Warm closing (e.g., "Best regards," or "Thanks,").
   
2. **Tone:** {tone_instruction}
   - If sentiment is negative, be patient and reassuring.
   - Avoid "corporate speak" where possible. Be helpful and direct.

3. **Special Handling for Off-Topic Queries:**
   - If **Evaluator Reasoning** says "Off-topic query" or similar:
     - Ignore the retrieved documents.
     - Politely refuse to answer (e.g., "I apologize, but I specialize in Doxa software support and cannot help with cooking recipes/weather etc.").
     - Do NOT offer to loop in a specialist.

4. **Format:** Return the response in plain text format, ready to send. DO NOT wrap it in JSON.
"""

response_agent = Agent(
    model=mistral,
    name="Response Composer",
    description="Generates structured customer support responses based on retrieval context.",
)

def compose_response(
    analysis: AnalysisResult, 
    retrieval: RetrievalResult, 
    confidence: float, 
    reasoning: str
) -> str:
    """
    Generates a final response for the client.
    """
    
    # Prepare knowledge context string
    knowledge_texts = []
    if retrieval and retrieval.documents:
        for i, doc in enumerate(retrieval.documents, 1):
            content = doc.get("content", "").strip()[:500] # Limit context length per doc
            knowledge_texts.append(f"[{i}] {content}")
    knowledge_context = "\n\n".join(knowledge_texts)

    # Determine tone based on sentiment
    tone_instruction = "Professional and helpful."
    if analysis.sentiment == "negative":
        tone_instruction = "Empathetic, apologetic, and reassuring. Prioritize de-escalation."

    # Format the prompt
    prompt = RESPONSE_PROMPT.format(
        issue_summary=analysis.summary or "User is facing an undefined issue.",
        sentiment=analysis.sentiment,
        knowledge_context=knowledge_context if knowledge_context else "No specific knowledge found.",
        confidence=confidence,
        reasoning=reasoning,
        tone_instruction=tone_instruction,
        language=analysis.language
    )

    # Run the agent
    response = response_agent.run(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    # Clean potential markdown wrapping if the model adds it unnecessarily
    if response_text.startswith("```") and response_text.endswith("```"):
        response_text = response_text.strip("`").replace("markdown", "").replace("text", "").strip()

    return response_text
