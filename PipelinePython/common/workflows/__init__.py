"""Canonical end-to-end workflow composition for the biosignal pipeline.

The signal, feature, training, and export implementations remain in
``PipelinePython.common``.  This package owns only their durable stage order and keeps
the three operator-facing flows in one place.
"""

from .runner import CanonicalWorkflowRunner, WorkflowName

__all__ = ["CanonicalWorkflowRunner", "WorkflowName"]
