from __future__ import annotations


class PlanError(ValueError):
    """A config the plan cannot represent. Match on ``code`` and the fields, not the text."""

    def __init__(
        self,
        code: str,
        *,
        key: str | None,
        value: object,
        allowed: tuple[str, ...] | None = None,
        hint: str | None = None,
    ) -> None:
        self.code = code
        self.key = key
        self.value = value
        self.allowed = allowed
        self.hint = hint
        message = f"[{code}] {key}={value!r}" if key else f"[{code}] {value!r}"
        if allowed:
            message += f" (allowed: {', '.join(allowed)})"
        if hint:
            message += f": {hint}"
        super().__init__(message)
