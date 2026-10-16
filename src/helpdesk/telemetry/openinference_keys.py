"""Span attribute keys that OpenInference writes on LangChain spans.

Verified against openinference-instrumentation-langchain 0.1.79. They are spelled out here
because openinference-semantic-conventions is only a transitive dependency.
"""

SPAN_KIND = "openinference.span.kind"
SPAN_KIND_LLM = "LLM"
OUTPUT_VALUE = "output.value"
LLM_MODEL_NAME = "llm.model_name"
LLM_TOKENS_PROMPT = "llm.token_count.prompt"
LLM_TOKENS_COMPLETION = "llm.token_count.completion"
LLM_TOKENS_CACHE_READ = "llm.token_count.prompt_details.cache_read"
LLM_TOKENS_CACHE_WRITE = "llm.token_count.prompt_details.cache_write"
