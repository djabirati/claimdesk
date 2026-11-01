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


class JobStatus(StrEnum):
    QUEUED = "queued"  # waiting for a worker (until run_after when it is a retry)
    RUNNING = "running"  # claimed by a worker
    DONE = "done"
    FAILED = "failed"  # no attempt left, or an error that a retry would not fix
