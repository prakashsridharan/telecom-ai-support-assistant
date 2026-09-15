from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_component_readiness():
    body = client.get("/health").json()
    assert body["components"]["retriever"]["status"] == "ok"
    assert "4 knowledge documents" in body["components"]["retriever"]["detail"]
    assert body["mode"] == "demo"


def test_chat_endpoint_returns_the_decision_trail():
    response = client.post("/api/chat", json={"message": "Check order ORD-10234"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "order_status"
    assert body["tool_calls"] == ["get_order_status"]
    assert body["mode"] == "demo"


def test_chat_endpoint_threads_history():
    response = client.post(
        "/api/chat",
        json={
            "message": "Is it delivered yet?",
            "history": [
                {"role": "user", "content": "Check order ORD-10234"},
                {"role": "assistant", "content": "Order ORD-10234 is Shipped."},
            ],
        },
    )
    assert response.status_code == 200
    assert "ORD-10234" in response.json()["answer"]


def test_empty_message_is_rejected():
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
