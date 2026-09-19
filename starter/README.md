# Customer Support AI Agent (AgentCore)

Amazon Bedrock AgentCore customer support agent with Gateway tools, Knowledge Base RAG, long-term memory, code interpreter, and browser tool.

## Submission

| Artifact                           | Location                           |
| ---------------------------------- | ---------------------------------- |
| Agent implementation               | [`main.py`](./main.py)             |
| Written reflection (200–400 words) | [`REFLECTION.md`](./REFLECTION.md) |
| Test screenshots (PNG)             | [`screenshots/`](./screenshots/)   |

### Screenshots

| Test                                   | File                                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------------------ |
| Test 1 — Order Tracking                | [`screenshots/test1_order_tracking.png`](./screenshots/test1_order_tracking.png)     |
| Test 2 — Refund Processing             | [`screenshots/test2_refund.png`](./screenshots/test2_refund.png)                     |
| Test 3 — Knowledge Base (RAG)          | [`screenshots/test3_knowledge_base.png`](./screenshots/test3_knowledge_base.png)     |
| Test 4a — Long-Term Memory (Session A) | [`screenshots/test4a_memory_intro.png`](./screenshots/test4a_memory_intro.png)       |
| Test 4b — Long-Term Memory (Session B) | [`screenshots/test4b_memory_recall.png`](./screenshots/test4b_memory_recall.png)     |
| Test 5 — Loyalty Discount Calculation  | [`screenshots/test5_loyalty_discount.png`](./screenshots/test5_loyalty_discount.png) |
| Test 6 — Browser Tool                  | [`screenshots/test6_browser.png`](./screenshots/test6_browser.png)                   |

## Project layout

```text
starter/
├── main.py                 # Agent entrypoint (TODOs implemented)
├── REFLECTION.md           # Design / challenge / production reflection
├── screenshots/            # agentcore invoke test evidence (PNG)
├── lambda/                 # order-tracker & refund-processor sources
├── product_catalog.txt     # Knowledge Base source file
├── pyproject.toml
├── .bedrock_agentcore.yaml # AgentCore deploy config
└── README.md               # this file
```

## Quick start

```bash
cd starter
uv sync
# Config values are set in main.py (GATEWAY_URL, KB_ID, REGION, MEMORY_ID)
agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```
