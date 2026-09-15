"""Synthetic business APIs used by the demo agent.

These functions deliberately use in-memory data so the portfolio project is
safe to publish publicly. In a real implementation they would call secured
enterprise APIs.
"""

ORDERS = {
    "ORD-10001": {"status": "Delivered", "eta": "Delivered on 2026-09-02"},
    "ORD-10234": {"status": "Shipped", "eta": "Expected 2026-09-10"},
    "ORD-10077": {"status": "Processing", "eta": "Expected 2026-09-12"},
}

BILLING = {
    "CUST-1001": {"balance": "$0.00", "due_date": "2026-09-15"},
    "CUST-1002": {"balance": "$79.00", "due_date": "2026-09-18"},
}

SERVICE_STATUS = {
    "Chennai": "No active major outage is reported in the demo service-status system.",
    "Bengaluru": "A simulated maintenance window is scheduled for 2026-09-11 01:00-03:00 IST.",
}

# Lookup index so callers can match case-insensitively without the orchestrator
# needing its own copy of the location list.
_SERVICE_STATUS_BY_KEY = {name.lower(): name for name in SERVICE_STATUS}


def known_locations() -> tuple[str, ...]:
    """Locations the service-status system can report on, in display casing."""
    return tuple(SERVICE_STATUS)


def get_order_status(order_id: str) -> dict:
    order_id = order_id.upper()
    if order_id not in ORDERS:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order_id": order_id, **ORDERS[order_id]}


def get_billing_status(customer_id: str) -> dict:
    customer_id = customer_id.upper()
    if customer_id not in BILLING:
        return {"found": False, "customer_id": customer_id}
    return {"found": True, "customer_id": customer_id, **BILLING[customer_id]}


def get_service_status(location: str) -> dict:
    name = _SERVICE_STATUS_BY_KEY.get(location.lower().strip())
    if name is None:
        return {
            "found": False,
            "location": location,
            "status": (
                "No service-status information is available for that location."
            ),
        }
    return {"found": True, "location": name, "status": SERVICE_STATUS[name]}
