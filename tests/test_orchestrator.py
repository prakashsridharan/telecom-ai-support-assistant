from app.models import ChatMessage
from app.services.orchestrator import Orchestrator


def user(text):
    return ChatMessage(role="user", content=text)


def assistant(text):
    return ChatMessage(role="assistant", content=text)


# --------------------------------------------------------------------
# Tool routing
# --------------------------------------------------------------------


def test_order_status():
    result = Orchestrator().respond("Check order ORD-10234")
    assert result["intent"] == "order_status"
    assert "ORD-10234" in result["answer"]
    assert "get_order_status" in result["tool_calls"]


def test_unknown_order():
    result = Orchestrator().respond("Check order ORD-99999")
    assert "could not find" in result["answer"].lower()


def test_order_id_without_separator():
    result = Orchestrator().respond("status of ord10077 please")
    assert "ORD-10077" in result["answer"]


def test_knowledge_question():
    result = Orchestrator().respond("How much does the Premium plan cost?")
    assert result["intent"] == "knowledge"
    assert result["sources"]
    assert result["sources"][0] == "plans.md"


def test_missing_billing_id():
    result = Orchestrator().respond("What is my billing balance?")
    assert result["intent"] == "billing"
    assert "customer id" in result["answer"].lower()


# --------------------------------------------------------------------
# Knowledge-base fallback when no identifier is present
# --------------------------------------------------------------------


def test_policy_question_answers_from_kb_instead_of_demanding_an_id():
    result = Orchestrator().respond("What is your refund policy?")
    assert result["intent"] == "billing"
    assert result["sources"][0] == "billing.md"
    assert not result["tool_calls"]
    # Still offers the lookup rather than only quoting policy.
    assert "customer id" in result["answer"].lower()


def test_missing_order_id_does_not_invent_a_status():
    result = Orchestrator().respond("Where is my order?")
    assert result["intent"] == "order_status"
    assert not result["tool_calls"]
    assert "is currently" not in result["answer"]


# --------------------------------------------------------------------
# Routing precision
# --------------------------------------------------------------------


def test_speed_complaint_is_not_an_outage_lookup():
    result = Orchestrator().respond("My data speed went down after the update")
    assert result["intent"] == "knowledge"
    assert not result["tool_calls"]


def test_network_down_with_location_is_an_outage_lookup():
    result = Orchestrator().respond("My network is down in Chennai")
    assert result["intent"] == "service_status"
    assert "get_service_status" in result["tool_calls"]


def test_network_issues_phrasing_is_an_outage_lookup():
    result = Orchestrator().respond("Any network issues in Bengaluru?")
    assert "get_service_status" in result["tool_calls"]
    assert "maintenance window" in result["answer"]


def test_unknown_location_is_not_reported_as_healthy():
    result = Orchestrator().respond("Is there an outage in Springfield?")
    assert "No active major outage" not in result["answer"]


# --------------------------------------------------------------------
# Multi-turn conversation context
# --------------------------------------------------------------------


def test_followup_inherits_order_intent_and_id():
    history = [
        user("Check order ORD-10234"),
        assistant("Order ORD-10234 is currently Shipped."),
    ]
    result = Orchestrator().respond("Is it delivered yet?", history)
    assert result["intent"] == "order_status"
    assert "ORD-10234" in result["answer"]


def test_followup_inherits_location():
    history = [
        user("I am travelling to Bengaluru tomorrow"),
        assistant("Noted."),
    ]
    result = Orchestrator().respond("Any outage there?", history)
    assert result["intent"] == "service_status"
    assert "maintenance window" in result["answer"]


def test_new_topic_does_not_inherit_previous_intent():
    history = [
        user("Check order ORD-10234"),
        assistant("Order ORD-10234 is currently Shipped."),
    ]
    result = Orchestrator().respond("What plans do you offer?", history)
    assert result["intent"] == "knowledge"
    assert not result["tool_calls"]


def test_followup_opener_does_not_override_a_fresh_topic():
    history = [
        user("What is your refund policy?"),
        assistant("Refund eligibility depends on the transaction."),
    ]
    result = Orchestrator().respond("And what plans do you offer?", history)
    assert result["intent"] == "knowledge"
    assert result["sources"][0] == "plans.md"


def test_identifiers_are_never_harvested_from_assistant_turns():
    """The assistant's own example ID must not become a real lookup."""
    history = [
        user("What is your refund policy?"),
        assistant(
            "Please share your customer ID, for example CUST-1001."
        ),
    ]
    result = Orchestrator().respond("What is my balance?", history)
    assert not result["tool_calls"]
    assert "$0.00" not in result["answer"]


def test_history_is_bounded():
    """Only recent turns are scanned, so a stale ID does not resurface."""
    history = [user("Check order ORD-10001")] + [
        user("Tell me about plans") for _ in range(20)
    ]
    result = Orchestrator().respond("Is it delivered yet?", history)
    assert "ORD-10001" not in result["answer"]
