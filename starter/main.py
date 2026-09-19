"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in Part 1 of the INSTRUCTIONS.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-3kxuckguj9.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "FTRFS6YXKO"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-rBoClQFLXZ"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)

memory_client = MemoryClient(region_name=REGION)

_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Return a dict: { strategy["type"]: strategy["namespaces"][0] for each strategy }
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    try:
        strategies = mem_client.get_memory_strategies(memory_id)
        return {
            strategy["type"]: strategy["namespaces"][0]
            for strategy in strategies
            if strategy.get("namespaces")
        }
    except Exception as exc:
        # Fall back to the project namespaces if the runtime role cannot
        # call GetMemory (common in locked-down lab accounts).
        logger.warning("get_memory_strategies failed (%s); using defaults", exc)
        return {
            "SEMANTIC": "cs_agent/{actorId}/facts",
            "USER_PREFERENCE": "cs_agent/{actorId}/preferences",
        }


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent

def _message_text(message) -> str:
    """Extract plain text from a message content block list."""
    parts = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _is_plain_user_message(message) -> bool:
    """True when the message is a user turn without tool results."""
    if message.get("role") != "user":
        return False
    for block in message.get("content") or []:
        if isinstance(block, dict) and "toolResult" in block:
            return False
    return bool(_message_text(message))


class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return

        last_message = messages[-1]
        if not _is_plain_user_message(last_message):
            return

        user_query = _message_text(last_message)
        memory_lines = []

        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                records = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
            except Exception as exc:
                logger.warning("Memory retrieval failed for %s: %s", strategy_type, exc)
                continue

            for record in records or []:
                text = (
                    (record.get("content") or {}).get("text")
                    or record.get("text")
                    or ""
                ).strip()
                if text:
                    memory_lines.append(f"[{strategy_type}] {text}")

        if not memory_lines:
            return

        context_block = "Customer Context:\n" + "\n".join(memory_lines)
        enriched = f"{context_block}\n\n{user_query}"

        # Mutate the last user message so the model sees recalled context.
        content = last_message.get("content") or []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                block["text"] = enriched
                break
        else:
            last_message["content"] = [{"text": enriched}]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages
        if not messages:
            return

        customer_query = None
        agent_response = None

        for message in reversed(messages):
            role = message.get("role")
            if agent_response is None and role == "assistant":
                text = _message_text(message)
                if text:
                    agent_response = text
            elif customer_query is None and _is_plain_user_message(message):
                # Strip any injected customer context before saving.
                text = _message_text(message)
                if text.startswith("Customer Context:"):
                    parts = text.split("\n\n", 1)
                    text = parts[1] if len(parts) > 1 else text
                customer_query = text

            if customer_query and agent_response:
                break

        if not customer_query or not agent_response:
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (agent_response, "ASSISTANT"),
                ],
            )
        except Exception as exc:
            logger.warning("Failed to save support interaction: %s", exc)

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID.startswith("<"):
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as exc:
        logger.warning("Knowledge base retrieve failed: %s", exc)
        return f"Knowledge base search failed: {exc}"

    results = resp.get("retrievalResults") or []
    if not results:
        return "No relevant information found in the knowledge base."

    chunks = []
    for item in results:
        text = ((item.get("content") or {}).get("text") or "").strip()
        if text:
            chunks.append(text)

    if not chunks:
        return "No relevant information found in the knowledge base."

    return "\n---\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

loyalty_points = {int(loyalty_points)}
tier = {json.dumps(tier)}
order_total = {float(order_total)}
product_category = {json.dumps(product_category)}

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

# 100 points = $1. Cap redemption at 50% of the order value, floor to nearest 500.
max_redeemable_points = int(order_total * 0.5 * 100)
raw_points = min(loyalty_points, max_redeemable_points)
points_redeemed = (raw_points // 500) * 500
points_value = points_redeemed / 100.0

subtotal_after_points = order_total - points_value
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = subtotal_after_points * tier_rate
final_total = subtotal_after_points - tier_discount
total_savings = points_value + tier_discount
earn_rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned

print(json.dumps({{
    "loyalty_points": loyalty_points,
    "tier": tier,
    "order_total": order_total,
    "product_category": product_category,
    "points_redeemed": points_redeemed,
    "points_value_usd": round(points_value, 2),
    "tier_discount_rate": tier_rate,
    "tier_discount_usd": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )
            for event in response.get("stream", []):
                if "result" in event:
                    return json.dumps(event["result"])
            return json.dumps({"error": "Code Interpreter returned no result events."})

    except Exception as e:
        # Fallback: apply tier discount only when Code Interpreter is unavailable.
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_rate = tier_rates.get(tier, 0.0)
        tier_discount = float(order_total) * tier_rate
        final_total = float(order_total) - tier_discount
        return json.dumps(
            {
                "fallback": True,
                "error": str(e),
                "tier": tier,
                "tier_discount_rate": tier_rate,
                "tier_discount_usd": round(tier_discount, 2),
                "final_total": round(final_total, 2),
                "note": "Code Interpreter unavailable; applied tier discount only.",
            }
        )


SYSTEM_PROMPT = """
You are a helpful customer support assistant for an e-commerce platform.

You can:
- Track orders and look up customer profiles via Gateway tools
- Initiate refunds and generate return labels via Gateway tools
- Search the product catalog / loyalty / return policy knowledge base
- Calculate loyalty point redemptions and tier discounts with the code interpreter
- Browse the web when the customer asks about a public page

Guidelines:
- Prefer tools over guessing for order status, refunds, policies, and math
- Be clear and accurate; respect any recalled customer preferences
- When recalling long-term memory, use the Customer Context provided with the message
""".strip()


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt") or payload.get("input") or ""
        actor_id = payload.get("customer_id") or payload.get("actor_id") or "anonymous"
        session_id = payload.get("session_id") or str(uuid.uuid4())

        if not user_input:
            return "Please provide a prompt in the request payload."

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )
        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )
            # Prefer async invoke inside the AgentCore entrypoint to avoid
            # nested asyncio.run() conflicts with tools like the browser.
            result = await agent.invoke_async(user_input)

        content = (result.message or {}).get("content") or []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                return block["text"]
        return str(result)

    except Exception as exc:
        logger.exception("Agent invocation failed")
        return f"Sorry, I ran into an error while handling your request: {exc}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
