"""Errors raised by the guards. Each one names what was refused and why."""

from __future__ import annotations


class AccbenchError(Exception):
    """Base class for every refusal raised by accbench."""


class ProvenanceError(AccbenchError):
    """An artifact's recorded provenance does not hold."""


class MissingProvenance(ProvenanceError):
    """The artifact carries no provenance stamp, so nothing about it can be checked."""


class StaleInput(ProvenanceError):
    """An input recorded in the stamp has changed or gone since the artifact was written."""


class CodeStateMismatch(ProvenanceError):
    """The artifact was written by different code from the code it is checked against."""


class ConstructError(AccbenchError):
    """The construct statement is missing or does not pass its schema check."""

    def __init__(self, path: object, problems: list[str]):
        self.path = path
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"construct statement {path} is not valid:\n{lines}")


class FixtureError(AccbenchError):
    """The fixture spec is invalid, or the data cannot be frozen under it."""


class ChannelError(AccbenchError):
    """The channel registry is invalid or does not match the data."""

    def __init__(self, message: str, problems: list[str] | None = None):
        self.problems = list(problems or [])
        if self.problems:
            message += ":\n" + "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(message)


class UnregisteredColumn(ChannelError):
    """A column would reach a model without being registered to a channel."""


class TrainOnlyViolation(ChannelError):
    """A featurizer was asked to fit on rows outside the training split."""


class LogError(AccbenchError):
    """An append-only log was edited, or an entry was refused."""
