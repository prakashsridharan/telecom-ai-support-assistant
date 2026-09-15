# Conversation Flows

## Flow 1: Knowledge question

User asks about a plan.

1. Classify as `knowledge`.
2. Retrieve relevant documents.
3. Supply retrieved context to the LLM.
4. Generate a grounded response.
5. Return source documents.

## Flow 2: Order status

User asks: `Check order ORD-10234`.

1. Detect order intent.
2. Extract order ID.
3. Invoke `get_order_status`.
4. Return the verified demo result.
5. Record the tool call.

## Flow 3: Missing identifier

User asks: `Where is my order?`

1. Detect order intent only if the wording indicates order status.
2. If no order ID exists, request it.
3. Do not invent an order.

## Flow 4: Unsupported request

If retrieval produces no useful result, the assistant states that it cannot reliably answer from the available information and offers escalation.
