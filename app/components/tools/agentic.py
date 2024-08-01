from langchain_core.pydantic_v1 import BaseModel, Field


class EscalateRouter(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = Field(
        default=True, description="Whether to cancel the current task and escalate control to the main assistant.")
    reason: str = Field(
        description="The reason for cancelling or not cancelling the current task and escalating control to the main assistant.",
    )

    class Config:
        schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }
