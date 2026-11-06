import os

# Tests never send traces to LangSmith.
os.environ["LANGSMITH_TRACING"] = "false"
