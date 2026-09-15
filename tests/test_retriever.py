from app.services.retriever import KnowledgeRetriever


def test_loads_knowledge_base_regardless_of_working_directory():
    retriever = KnowledgeRetriever()
    assert retriever.is_ready
    assert retriever.document_count == 4


def test_rare_terms_outrank_common_ones():
    """'premium' must beat the function words shared by every document."""
    results = KnowledgeRetriever().search("How much does the Premium plan cost?")
    assert results[0]["source"] == "plans.md"


def test_generic_verbs_do_not_dominate_ranking():
    results = KnowledgeRetriever().search("What plans do you offer?")
    assert results[0]["source"] == "plans.md"


def test_policy_question_retrieves_the_policy_document():
    results = KnowledgeRetriever().search(
        "What is your policy on planned maintenance communication?"
    )
    assert results[0]["source"] == "service_policies.md"


def test_off_topic_query_retrieves_nothing():
    assert KnowledgeRetriever().search("What is the capital of Peru?") == []


def test_scores_are_descending():
    results = KnowledgeRetriever().search("billing invoice payment policy")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_missing_directory_is_not_ready_rather_than_crashing():
    retriever = KnowledgeRetriever("does_not_exist")
    assert not retriever.is_ready
    assert retriever.search("anything") == []
