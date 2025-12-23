@dataclass
class Ticket:
    """Représentation d'un ticket support."""
    id: str
    subject: str
    description: str
    client_id: str
    timestamp: str


@dataclass
class AnalysisResult:
    """Résultat de l'analyse du ticket."""
    summary: str
    keywords: List[str]
    ticket_id: str


@dataclass
class SolutionResult:
    """Résultat de la recherche de solutions."""
    relevant_docs: List[Dict]
    snippets: List[str]
    confidence_score: float


@dataclass
class DecisionResult:
    """Résultat de l'évaluation et décision."""
    should_escalate: bool
    confidence: float
    escalation_reason: Optional[str]
    detected_issues: List[str]


@dataclass
class FinalResponse:
    """Réponse finale structurée."""
    response_text: str
    ticket_id: str
    escalated: bool
    sources_used: List[str]


class QueryAnalyzerAgent:
    """Agent 1: Analyse le ticket et extrait les mots-clés."""
    
    def __init__(self, api_key: str):
        self.agent = Agent(
            name="Query Analyzer",
            model=MistralChat(id="mistral-large-latest", api_key=api_key),
            instructions=[
                "Tu es un expert en analyse de tickets support.",
                "Ton rôle est de résumer le ticket et d'extraire les mots-clés pertinents.",
                "Identifie le problème principal, les détails techniques, et les termes importants.",
                "Sois concis mais précis dans ton résumé."
            ],
            markdown=True
        )
    
    def analyze(self, ticket: Ticket) -> AnalysisResult:
        """Analyse le ticket et extrait les informations clés."""
        prompt = f"""
Analyse ce ticket support:

Sujet: {ticket.subject}
Description: {ticket.description}

Fournis:
1. Un résumé en 2-3 phrases
2. Les mots-clés principaux (5-10 mots) séparés par des virgules

Format de réponse:
RÉSUMÉ: [ton résumé]
MOTS-CLÉS: [mot1, mot2, mot3, ...]
"""
        
        response = self.agent.run(prompt)
        content = response.content
        
        # Parse la réponse
        lines = content.split('\n')
        summary = ""
        keywords = []
        
        for line in lines:
            if line.startswith('RÉSUMÉ:'):
                summary = line.replace('RÉSUMÉ:', '').strip()
            elif line.startswith('MOTS-CLÉS:'):
                keywords_str = line.replace('MOTS-CLÉS:', '').strip()
                keywords = [k.strip() for k in keywords_str.split(',')]
        
        print(f"\n📊 Query Analyzer:")
        print(f"   Résumé: {summary}")
        print(f"   Mots-clés: {keywords}")
        
        return AnalysisResult(
            summary=summary,
            keywords=keywords,
            ticket_id=ticket.id
        )


class SolutionFinderAgent:
    """Agent 2: Recherche des solutions via RAG - Utilise votre RAGPipeline."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize with your existing RAG pipeline.
        
        Args:
            rag_pipeline: Instance of your RAGPipeline class
        """
        self.rag_pipeline = rag_pipeline
        self.api_key = rag_pipeline.api_key
        
        # Get the document store from the RAG pipeline
        from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
        self.document_store = QdrantDocumentStore(
            path=rag_pipeline.db_path,
            index="doxa_docs",
            embedding_dim=1024,
            recreate_index=False
        )
        
        # Build retrieval pipeline
        self.pipeline = self._build_rag_pipeline()
    
    def _build_rag_pipeline(self) -> Pipeline:
        """Construit le pipeline RAG de recherche."""
        pipeline = Pipeline()
        
        embedder = Mistral(
            api_key=Secret.from_token(self.api_key),
            model="mistral-embed"
        )
        pipeline.add_component("embedder", embedder)
        
        retriever = QdrantEmbeddingRetriever(
            document_store=self.document_store,
            top_k=5
        )
        pipeline.add_component("retriever", retriever)
        
        pipeline.connect("embedder.embedding", "retriever.query_embedding")
        
        return pipeline
    
    def find_solutions(self, analysis: AnalysisResult) -> SolutionResult:
        """Recherche des documents pertinents dans la KB."""
        # Combine résumé et mots-clés pour la recherche
        search_query = f"{analysis.summary} {' '.join(analysis.keywords)}"
        
        print(f"\n🔍 Solution Finder:")
        print(f"   Recherche: {search_query[:100]}...")
        
        result = self.pipeline.run({
            "embedder": {"text": search_query}
        })
        
        documents = result["retriever"]["documents"]
        
        relevant_docs = []
        snippets = []
        
        for doc in documents:
            relevant_docs.append({
                "source": doc.meta.get("source_file", "Unknown"),
                "content": doc.content[:300],
                "score": getattr(doc, 'score', 0.0)
            })
            snippets.append(doc.content[:500])
        
        # Calcul du score de confiance basé sur les scores de similarité
        avg_score = sum(d["score"] for d in relevant_docs) / len(relevant_docs) if relevant_docs else 0
        confidence = min(avg_score * 100, 100)
        
        print(f"   Documents trouvés: {len(relevant_docs)}")
        print(f"   Confiance: {confidence:.1f}%")
        
        return SolutionResult(
            relevant_docs=relevant_docs,
            snippets=snippets,
            confidence_score=confidence
        )


class EvaluatorDeciderAgent:
    """Agent 3: Évalue la confiance et décide si escalade nécessaire."""
    
    def __init__(self, api_key: str):
        self.agent = Agent(
            name="Evaluator & Decider",
            model=MistralChat(id="mistral-large-latest", api_key=api_key),
            instructions=[
                "Tu es un expert en évaluation de qualité de support.",
                "Analyse si la réponse proposée est adéquate ou nécessite une escalade.",
                "Détecte: données sensibles, émotions négatives fortes, problèmes non-standard.",
                "Confiance <60% = escalade automatique.",
                "Sois prudent et privilégie l'escalade en cas de doute."
            ],
            markdown=True
        )
    
    def evaluate(
        self, 
        ticket: Ticket, 
        analysis: AnalysisResult, 
        solution: SolutionResult
    ) -> DecisionResult:
        """Évalue et décide si escalade nécessaire."""
        
        prompt = f"""
Évalue cette situation de support:

TICKET:
- Sujet: {ticket.subject}
- Description: {ticket.description}

ANALYSE:
- Résumé: {analysis.summary}
- Mots-clés: {', '.join(analysis.keywords)}

SOLUTIONS TROUVÉES:
- Nombre de docs: {len(solution.relevant_docs)}
- Confiance RAG: {solution.confidence_score:.1f}%
- Extraits: {solution.snippets[0][:200] if solution.snippets else 'Aucun'}...

Décide:
1. ESCALADE NÉCESSAIRE? (OUI/NON)
2. RAISON si escalade
3. PROBLÈMES DÉTECTÉS (données sensibles, émotions négatives, non-standard)

Format:
ESCALADE: [OUI/NON]
CONFIANCE: [0-100]%
RAISON: [raison si escalade]
PROBLÈMES: [liste séparée par virgules]
"""
        
        response = self.agent.run(prompt)
        content = response.content
        
        # Parse la réponse
        should_escalate = "OUI" in content.upper()
        confidence = solution.confidence_score
        escalation_reason = None
        detected_issues = []
        
        lines = content.split('\n')
        for line in lines:
            if line.startswith('CONFIANCE:'):
                try:
                    conf_str = line.replace('CONFIANCE:', '').strip().replace('%', '')
                    confidence = float(conf_str)
                except:
                    pass
            elif line.startswith('RAISON:'):
                escalation_reason = line.replace('RAISON:', '').strip()
            elif line.startswith('PROBLÈMES:'):
                issues_str = line.replace('PROBLÈMES:', '').strip()
                detected_issues = [i.strip() for i in issues_str.split(',') if i.strip()]
        
        # Force escalade si confiance < 60%
        if confidence < 60:
            should_escalate = True
            if not escalation_reason:
                escalation_reason = "Confiance insuffisante (<60%)"
        
        print(f"\n⚖️  Evaluator & Decider:")
        print(f"   Confiance: {confidence:.1f}%")
        print(f"   Escalade: {'OUI' if should_escalate else 'NON'}")
        if escalation_reason:
            print(f"   Raison: {escalation_reason}")
        if detected_issues:
            print(f"   Problèmes: {detected_issues}")
        
        return DecisionResult(
            should_escalate=should_escalate,
            confidence=confidence,
            escalation_reason=escalation_reason,
            detected_issues=detected_issues
        )


class ResponseComposerAgent:
    """Agent 4: Génère la réponse structurée finale."""
    
    def __init__(self, api_key: str):
        self.agent = Agent(
            name="Response Composer",
            model=MistralChat(id="mistral-large-latest", api_key=api_key),
            instructions=[
                "Tu es un expert en rédaction de réponses support professionnelles.",
                "Génère des réponses structurées: remerciements, problème adressé, solution, closing.",
                "Ton ton est professionnel, empathique et clair.",
                "Cite toujours les sources utilisées.",
                "Si escalade: explique que l'équipe spécialisée va prendre en charge."
            ],
            markdown=True
        )
    
    def compose(
        self,
        ticket: Ticket,
        analysis: AnalysisResult,
        solution: SolutionResult,
        decision: DecisionResult
    ) -> FinalResponse:
        """Compose la réponse finale."""
        
        sources = [doc["source"] for doc in solution.relevant_docs]
        
        if decision.should_escalate:
            prompt = f"""
Génère une réponse d'escalade pour ce ticket:

TICKET:
- Sujet: {ticket.subject}
- Description: {ticket.description}

RAISON ESCALADE: {decision.escalation_reason}

La réponse doit:
1. Remercier le client
2. Reconnaître le problème
3. Expliquer qu'une équipe spécialisée va prendre en charge
4. Rassurer sur le suivi

Ton: Professionnel et rassurant.
"""
        else:
            prompt = f"""
Génère une réponse de support pour ce ticket:

TICKET:
- Sujet: {ticket.subject}
- Description: {ticket.description}

PROBLÈME IDENTIFIÉ: {analysis.summary}

SOLUTIONS DISPONIBLES:
{chr(10).join(f"- {snippet[:200]}..." for snippet in solution.snippets[:3])}

SOURCES: {', '.join(sources)}

La réponse doit:
1. Remercier le client
2. Résumer le problème compris
3. Proposer la solution basée sur la documentation
4. Offrir assistance supplémentaire si besoin

Ton: Professionnel, empathique et clair.
"""
        
        response = self.agent.run(prompt)
        
        print(f"\n✍️  Response Composer:")
        print(f"   Escalade: {'OUI' if decision.should_escalate else 'NON'}")
        print(f"   Sources: {sources}")
        
        return FinalResponse(
            response_text=response.content,
            ticket_id=ticket.id,
            escalated=decision.should_escalate,
            sources_used=sources
        )


class SupportAgenticPipeline:
    """Pipeline complet d'agents IA pour le support - Utilise votre RAGPipeline."""
    
    def __init__(self, api_key: str, docs_dir: str = "./docs", db_path: str = "./db"):
        """
        Initialize the agentic pipeline with your existing RAG pipeline.
        
        Args:
            api_key: Mistral API key
            docs_dir: Directory containing PDF documents
            db_path: Path to Qdrant database
        """
        self.api_key = api_key
        
        # Initialize YOUR RAG pipeline
        self.rag_pipeline = RAGPipeline(
            api_key=api_key,
            docs_dir=docs_dir,
            db_path=db_path
        )
        
        # Initialize the 4 agents
        self.query_analyzer = QueryAnalyzerAgent(api_key)
        self.solution_finder = SolutionFinderAgent(self.rag_pipeline)
        self.evaluator_decider = EvaluatorDeciderAgent(api_key)
        self.response_composer = ResponseComposerAgent(api_key)
    
    def setup_knowledge_base(self):
        """
        Run your RAG pipeline to process PDFs and build the knowledge base.
        Only needs to be run once or when documents are updated.
        """
        print("\n🚀 Setting up Knowledge Base using your RAG Pipeline...")
        document_store = self.rag_pipeline.run_full_pipeline()
        print("✅ Knowledge Base ready!")
        return document_store
    
    def process_ticket(self, ticket: Ticket) -> FinalResponse:
        """
        Traite un ticket à travers le pipeline complet d'agents.
        
        Pipeline: Query Analyzer → Solution Finder → Evaluator & Decider → Response Composer
        """
        print("\n" + "="*80)
        print(f"🎫 TRAITEMENT DU TICKET: {ticket.id}")
        print("="*80)
        print(f"Sujet: {ticket.subject}")
        print(f"Description: {ticket.description[:100]}...")
        
        # Agent 1: Analyse
        analysis = self.query_analyzer.analyze(ticket)
        
        # Agent 2: Recherche de solutions (uses YOUR RAG pipeline)
        solution = self.solution_finder.find_solutions(analysis)
        
        # Agent 3: Évaluation et décision
        decision = self.evaluator_decider.evaluate(ticket, analysis, solution)
        
        # Agent 4: Composition de la réponse
        response = self.response_composer.compose(ticket, analysis, solution, decision)
        
        print("\n" + "="*80)
        print("✅ RÉPONSE GÉNÉRÉE")
        print("="*80)
        print(response.response_text)
        print("\n" + "="*80)
        
        return response


# Example usage
if __name__ == "__main__":
    # Load API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Please set MISTRAL_API_KEY environment variable")
    
    # Initialize pipeline with YOUR RAG pipeline
    pipeline = SupportAgenticPipeline(
        api_key=api_key,
        docs_dir="./docs",
        db_path="./db"
    )
    
    # FIRST TIME SETUP: Process PDFs and build knowledge base
    # Uncomment the line below if you need to build/rebuild the KB
    # pipeline.setup_knowledge_base()
    
    # Test avec un ticket exemple
    test_ticket = Ticket(
        id="TICKET-001",
        subject="Problème de connexion à mon compte",
        description="Bonjour, je n'arrive plus à me connecter à mon compte depuis ce matin. "
                   "J'ai essayé de réinitialiser mon mot de passe mais je ne reçois pas l'email. "
                   "C'est urgent car j'ai besoin d'accéder à mes documents.",
        client_id="CLIENT-123",
        timestamp="2024-01-15 10:30:00"
    )
    
    # Traiter le ticket
    response = pipeline.process_ticket(test_ticket)
    
    # Afficher le résultat
    print(f"\n📊 RÉSULTAT FINAL:")
    print(f"Ticket ID: {response.ticket_id}")
    print(f"Escaladé: {'OUI' if response.escalated else 'NON'}")
    print(f"Sources: {response.sources_used}")