from enum import StrEnum


class Category(StrEnum):
    DAMAGED = "damaged"
    LATE = "late"
    LOST = "lost"
    WRONG_ITEM = "wrong_item"
    REFUND_REQUEST = "refund_request"
    OTHER = "other"
    NOT_A_CLAIM = "not_a_claim"


CLAIM_CATEGORIES = {
    Category.DAMAGED,
    Category.LATE,
    Category.LOST,
    Category.WRONG_ITEM,
    Category.REFUND_REQUEST,
}


class Urgency(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class RequestedAction(StrEnum):
    REFUND = "refund"
    REPLACEMENT = "replacement"
    REDELIVERY = "redelivery"
    INFORMATION = "information"
    NONE = "none"


class EmailStatus(StrEnum):
    RECEIVED = "received"
    TRIAGED = "triaged"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class RunStatus(StrEnum):
    SUCCESS = "success"
    INVALID_OUTPUT = "invalid_output"
    ERROR = "error"  # the provider did not answer, a retry may work (network, quota, 5xx)
    REJECTED = "rejected"  # the provider refused the request (key, model): a retry would not


class JobKind(StrEnum):
    TRIAGE = "triage"
    AGENT = "agent"  # queued after a successful triage of a claim


class AgentRunStatus(StrEnum):
    COMPLETED = "completed"  # the agent finished and wrote its note for the reviewer
    STEP_LIMIT = "step_limit"  # stopped after the maximum number of model calls
    ERROR = "error"  # the provider did not answer, a retry may work
    REJECTED = "rejected"  # the provider refused the request, a retry would not


class ActionKind(StrEnum):
    REFUND = "refund"
    REPLACEMENT = "replacement"
    CARRIER_CLAIM = "carrier_claim"
    REPLY = "reply"


class ActionStatus(StrEnum):
    PROPOSED = "proposed"  # written by the agent, waiting for a person
    APPROVED = "approved"  # approved by a person, being executed
    EXECUTED = "executed"
    REJECTED = "rejected"  # refused by a person


class JobStatus(StrEnum):
    QUEUED = "queued"  # waiting for a worker (until run_after when it is a retry)
    RUNNING = "running"  # claimed by a worker
    DONE = "done"
    FAILED = "failed"  # no attempt left, or an error that a retry would not fix
