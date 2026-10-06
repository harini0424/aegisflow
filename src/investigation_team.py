"""
AegisFlow - Stage 5: 5-Agent AML Investigation Team
======================================================
This builds a team of 5 specialized AI "agents" that investigate a suspicious
case together, using LangGraph to pass work between them in sequence - like
an assembly line of virtual analysts.

IMPORTANT SAFETY RULE (matches the real AegisFlow spec): no AI agent ever
submits anything automatically. This script only produces a DRAFT report.
A real human (a compliance officer) must always review and approve it
before anything is considered final. This is built into the design on
purpose, not an afterthought.

HOW TO RUN THIS:
    python src/investigation_team.py

Make sure Ollama is running in the background first! (It usually starts
automatically after install, but if this fails with a connection error,
open a terminal and run: ollama serve)
"""

import json
from typing import TypedDict

import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

# ----------------------------------------------------------------------
# Connect to the free local AI model (Ollama)
# ----------------------------------------------------------------------
llm = ChatOllama(model="phi3", temperature=0.2)

# Load our transaction data (the agents will investigate a case from this)
transactions = pd.read_csv("data/transactions_with_features.csv", parse_dates=["timestamp"])


# ----------------------------------------------------------------------
# The shared "case file" that gets passed between agents and updated by
# each one. Think of this like a shared folder the whole team writes into.
# ----------------------------------------------------------------------
class CaseState(TypedDict):
    case_id: str
    account_ids: list
    device_ids: list
    transactions_summary: str
    link_analysis: str
    pattern_analysis: str
    media_notes: str
    draft_report: str
    qa_review: str
    qa_passed: bool


# ----------------------------------------------------------------------
# AGENT 1: Link Analyst - looks at the account/device network
# ----------------------------------------------------------------------
def link_analyst(state: CaseState) -> CaseState:
    print("\n[Agent 1/5] Link Analyst is investigating the account network...")

    prompt = f"""You are a fraud investigation Link Analyst at a bank.
Below is a summary of suspicious transactions for case {state['case_id']}.

{state['transactions_summary']}

Accounts involved: {state['account_ids']}
Devices involved: {state['device_ids']}

In 3-4 sentences, describe what the account/device network tells us.
Specifically note if multiple accounts are sharing the same device (a
classic fraud-ring signal) and how many accounts/devices are connected.
Be factual and concise."""

    response = llm.invoke(prompt)
    state["link_analysis"] = response.content
    print("Link Analyst finished.")
    return state


# ----------------------------------------------------------------------
# AGENT 2: Pattern Analyst - looks at timing/amount patterns
# ----------------------------------------------------------------------
def pattern_analyst(state: CaseState) -> CaseState:
    print("[Agent 2/5] Pattern Analyst is checking transaction timing patterns...")

    prompt = f"""You are a fraud investigation Pattern Analyst at a bank.
Below is a summary of suspicious transactions for case {state['case_id']}.

{state['transactions_summary']}

In 3-4 sentences, describe the TIMING and AMOUNT pattern. Specifically
check: are there many transactions in a short time window (a "burst")?
Are amounts similar across accounts (a possible "smurfing" / structuring
pattern, where a large amount is split into smaller pieces to avoid
detection)? Be factual and concise."""

    response = llm.invoke(prompt)
    state["pattern_analysis"] = response.content
    print("Pattern Analyst finished.")
    return state


# ----------------------------------------------------------------------
# AGENT 3: Media Researcher - simulated (no real internet search, since
# we're keeping this zero-cost and offline-friendly). In a real system,
# this agent would search news/sanctions databases.
# ----------------------------------------------------------------------
def media_researcher(state: CaseState) -> CaseState:
    print("[Agent 3/5] Media Researcher is checking for adverse media/sanctions hits...")

    # Simulated: in the full spec this searches real news articles.
    # For our free/offline build, we simulate a "no adverse media found"
    # result, and clearly label it as such so the report stays honest.
    state["media_notes"] = (
        "No adverse media or sanctions-list hits found for the accounts "
        "in this case (simulated check - a production system would search "
        "live news and sanctions databases)."
    )
    print("Media Researcher finished.")
    return state


# ----------------------------------------------------------------------
# AGENT 4: Narrative Writer - drafts the formal report
# ----------------------------------------------------------------------
def narrative_writer(state: CaseState) -> CaseState:
    print("[Agent 4/5] Narrative Writer is drafting the SAR report...")

    prompt = f"""You are a compliance officer drafting a Suspicious Activity
Report (SAR) narrative for case {state['case_id']}.

Use ONLY the findings below - do not invent any facts not mentioned here.

LINK ANALYSIS:
{state['link_analysis']}

PATTERN ANALYSIS:
{state['pattern_analysis']}

MEDIA/SANCTIONS CHECK:
{state['media_notes']}

Write a formal but concise SAR narrative (4-6 sentences) covering: what
was observed, why it is suspicious, and what accounts/entities are
involved. End with a clear recommendation (e.g., "Recommend further
investigation" or "Recommend account restriction pending review")."""

    response = llm.invoke(prompt)
    state["draft_report"] = response.content
    print("Narrative Writer finished.")
    return state


# ----------------------------------------------------------------------
# AGENT 5: QA Reviewer - checks the draft for problems before it reaches
# a human. This is the "hallucination check" the spec calls for.
# ----------------------------------------------------------------------
def qa_reviewer(state: CaseState) -> CaseState:
    print("[Agent 5/5] QA Reviewer is checking the draft report...")

    prompt = f"""You are a QA reviewer checking a draft SAR report before
it is shown to a human compliance officer.

ORIGINAL FINDINGS (the only facts allowed in the report):
Link analysis: {state['link_analysis']}
Pattern analysis: {state['pattern_analysis']}
Media notes: {state['media_notes']}

DRAFT REPORT TO CHECK:
{state['draft_report']}

Check: does the draft report introduce any claim NOT supported by the
findings above? Reply with exactly one line starting with either
"PASS:" or "FAIL:" followed by a one-sentence reason."""

    response = llm.invoke(prompt)
    state["qa_review"] = response.content
    state["qa_passed"] = response.content.strip().upper().startswith("PASS")
    print(f"QA Reviewer finished. Result: {'PASSED' if state['qa_passed'] else 'FAILED - needs human review'}")
    return state


# ----------------------------------------------------------------------
# Build the LangGraph pipeline: wire the 5 agents together in sequence
# ----------------------------------------------------------------------
def build_investigation_graph():
    graph = StateGraph(CaseState)

    graph.add_node("link_analyst", link_analyst)
    graph.add_node("pattern_analyst", pattern_analyst)
    graph.add_node("media_researcher", media_researcher)
    graph.add_node("narrative_writer", narrative_writer)
    graph.add_node("qa_reviewer", qa_reviewer)

    graph.set_entry_point("link_analyst")
    graph.add_edge("link_analyst", "pattern_analyst")
    graph.add_edge("pattern_analyst", "media_researcher")
    graph.add_edge("media_researcher", "narrative_writer")
    graph.add_edge("narrative_writer", "qa_reviewer")
    graph.add_edge("qa_reviewer", END)

    return graph.compile()


# ----------------------------------------------------------------------
# Helper: build a case summary from one of our fraud ring examples
# ----------------------------------------------------------------------
def build_case_from_ring_example():
    ring_txns = transactions[transactions["fraud_type"] == "ring"]
    if len(ring_txns) == 0:
        raise ValueError("No fraud ring examples found in the data. Run generate_transactions.py first.")

    # Take one ring's worth of transactions (same device)
    example_device = ring_txns["device_id"].iloc[0]
    case_txns = ring_txns[ring_txns["device_id"] == example_device]

    account_ids = case_txns["account_id"].unique().tolist()
    device_ids = case_txns["device_id"].unique().tolist()

    summary_lines = []
    for _, row in case_txns.iterrows():
        summary_lines.append(
            f"- Account {row['account_id'][:8]}... made a transaction of "
            f"₹{row['amount']:.2f} at {row['timestamp']} using device "
            f"{row['device_id'][:8]}..."
        )
    transactions_summary = "\n".join(summary_lines)

    return {
        "case_id": "CASE-" + example_device[:8],
        "account_ids": [a[:8] + "..." for a in account_ids],
        "device_ids": [d[:8] + "..." for d in device_ids],
        "transactions_summary": transactions_summary,
        "link_analysis": "",
        "pattern_analysis": "",
        "media_notes": "",
        "draft_report": "",
        "qa_review": "",
        "qa_passed": False,
    }


# ----------------------------------------------------------------------
# Run the investigation!
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AegisFlow - AML Investigation Team")
    print("=" * 60)

    initial_case = build_case_from_ring_example()
    print(f"\nInvestigating {initial_case['case_id']}")
    print(f"Accounts involved: {len(initial_case['account_ids'])}")
    print(f"Devices involved: {len(initial_case['device_ids'])}")

    app = build_investigation_graph()
    final_state = app.invoke(initial_case)

    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)

    print("\n--- LINK ANALYSIS ---")
    print(final_state["link_analysis"])

    print("\n--- PATTERN ANALYSIS ---")
    print(final_state["pattern_analysis"])

    print("\n--- MEDIA/SANCTIONS CHECK ---")
    print(final_state["media_notes"])

    print("\n--- DRAFT SAR REPORT ---")
    print(final_state["draft_report"])

    print("\n--- QA REVIEW ---")
    print(final_state["qa_review"])

    print("\n" + "=" * 60)
    status = "✅ Ready for human compliance officer review" if final_state["qa_passed"] else "⚠️ QA flagged an issue - needs human attention before review"
    print(status)
    print("=" * 60)
    print("\nREMINDER: This is a DRAFT only. No action is taken automatically.")
    print("A human compliance officer must review and approve before any filing.")

    # Save the full case file so we can show it in the dashboard later
    import os
    os.makedirs("cases", exist_ok=True)
    with open(f"cases/{final_state['case_id']}.json", "w") as f:
        json.dump(final_state, f, indent=2)
    print(f"\nFull case file saved to cases/{final_state['case_id']}.json")