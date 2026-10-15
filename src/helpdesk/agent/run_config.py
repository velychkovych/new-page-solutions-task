"""The run config: the one trusted source of `user_id` inside the graph (CONTRACTS §5.3)."""

from langchain_core.runnables import RunnableConfig

from helpdesk.shared.ids import is_principal_id, thread_id


class MissingUserIdError(RuntimeError):
    """A graph run started without a valid request user id in its config."""


def build_run_config(user_id: str, session_id: str, *, recursion_limit: int) -> RunnableConfig:
    # recursion_limit is always explicit: the installed LangGraph defaults it to 10007.
    return RunnableConfig(
        configurable={"thread_id": thread_id(user_id, session_id), "user_id": user_id},
        recursion_limit=recursion_limit,
    )


def user_id_from_config(config: RunnableConfig) -> str:
    """The request's user id. Never read a user id from model output instead."""
    user_id = config.get("configurable", {}).get("user_id")
    if not isinstance(user_id, str) or not is_principal_id(user_id):
        raise MissingUserIdError("run config has no valid configurable.user_id")
    return user_id
