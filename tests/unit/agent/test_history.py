"""History trimming and model-input repair (CONTRACTS §5.4)."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from helpdesk.agent.history import (
    INTERRUPTED_TOOL_CALL,
    answer_orphan_tool_calls,
    model_messages,
    trim_history,
)


def _turn(n: int, size: int) -> list:
    call_id = f"c{n}"
    return [
        HumanMessage(f"question {n} " + "x" * size),
        AIMessage(
            "", tool_calls=[{"name": "search_tickets", "args": {"query": "q"}, "id": call_id}]
        ),
        ToolMessage("result " * size, tool_call_id=call_id),
        AIMessage(f"answer {n}"),
    ]


def _pairs_intact(messages: list) -> bool:
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    called = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    return answered == called


def test_trimming_keeps_tool_call_pairs_and_starts_on_a_user_message():
    history = [m for n in range(10) for m in _turn(n, 200)]
    for budget in (300, 600, 1_000, 2_500, 5_000):
        trimmed = trim_history(history, token_budget=budget)
        assert trimmed, budget
        assert isinstance(trimmed[0], HumanMessage)
        assert _pairs_intact(trimmed), budget
        assert trimmed == history[-len(trimmed) :]
    assert len(trim_history(history, token_budget=600)) < len(history)
    assert trim_history(history, token_budget=1_000_000) == history


def test_an_over_budget_turn_is_kept_whole():
    history = [*_turn(0, 10), *_turn(1, 5_000)]
    trimmed = trim_history(history, token_budget=50)
    assert trimmed == history[4:]


def test_orphan_tool_calls_get_a_result_right_after_their_ai_message():
    orphan = AIMessage("", tool_calls=[{"name": "get_ticket", "args": {}, "id": "lost"}])
    messages = [HumanMessage("hi"), orphan, HumanMessage("still there?")]
    patched = answer_orphan_tool_calls(messages)
    assert [type(m).__name__ for m in patched] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "HumanMessage",
    ]
    assert patched[2].tool_call_id == "lost"
    assert patched[2].content == INTERRUPTED_TOOL_CALL
    assert answer_orphan_tool_calls(_turn(0, 1)) == _turn(0, 1)


def test_model_messages_put_the_system_prompt_first():
    system = SystemMessage("rules", id="sys")
    messages = [HumanMessage("one"), system, AIMessage("two")]
    assert model_messages(messages, token_budget=1_000) == [system, messages[0], messages[2]]


def test_orphan_invalid_tool_calls_get_a_result_too():
    # langchain-anthropic sends an invalid call (with id and name) as a tool_use block.
    orphan = AIMessage(
        "",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "name": "search_tickets",
                "args": '{"q',
                "id": "cut",
                "error": "not valid JSON",
            }
        ],
    )
    patched = answer_orphan_tool_calls([HumanMessage("hi"), orphan, HumanMessage("again")])
    assert isinstance(patched[2], ToolMessage)
    assert patched[2].tool_call_id == "cut"
    assert patched[2].status == "error"


def test_blank_replies_are_left_out_of_the_model_input():
    blank = AIMessage("  \n")
    blank_blocks = AIMessage(["  ", {"type": "text", "text": " "}])
    blank_with_call = AIMessage(" ", tool_calls=[{"name": "get_ticket", "args": {}, "id": "c1"}])
    result = ToolMessage("{}", tool_call_id="c1")
    messages = [HumanMessage("a"), blank, blank_blocks, HumanMessage("b"), blank_with_call, result]

    shaped = model_messages(messages, token_budget=1_000)

    assert [type(m).__name__ for m in shaped] == [
        "HumanMessage",
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert shaped[2].content == ""
    assert shaped[2].tool_calls == blank_with_call.tool_calls
    # The stored messages are untouched.
    assert blank_with_call.content == " "
