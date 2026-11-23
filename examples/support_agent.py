"""A canned support agent for the judge-scorer example.

It answers refund questions with a fixed policy statement, and
everything else with a generic reply. The point is the *scoring*:
the suite in ``examples/support.json`` judges these free-form
answers against a rubric with the built-in stub judge, so the demo
runs offline with no model.
"""


def agent(prompt: str) -> str:
    if "refund" in prompt.lower():
        return (
            "Our refund policy gives you 30 days from purchase to request "
            "a refund. Contact support with your order number and we will "
            "process it within 5 business days."
        )
    return "Thanks for reaching out. Let me look into that for you."
