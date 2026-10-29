"""
Strictly Typed Multi-Agent Swarm State Machine (INV-13, #6).

Implements a deterministic LangGraph-style typed state machine:
    RESEARCH -> CODE -> REVIEW -> APPROVED | REJECTED
with bounded REVIEW -> CODE revision cycles, where ResearcherAgent, CoderAgent,
and ReviewerAgent exchange strictly validated, frozen Pydantic v2 models and
reject raw string/bytes handoffs with TypeError.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
import uuid
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class SwarmPhase(str, Enum):
    """Deterministic lifecycle phases for the multi-agent coding swarm."""

    RESEARCH = "RESEARCH"
    CODE = "CODE"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class _SwarmEdgeMap(dict):  # type: ignore[type-arg]
    """Adjacency map supporting both ``edges[src]`` and ``(src, dst) in edges`` checks."""

    def __contains__(self, item: object) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            src, dst = item
            try:
                src_phase = SwarmPhase(src)
                dst_phase = SwarmPhase(dst)
            except ValueError:
                return False
            allowed = super().get(src_phase)
            return allowed is not None and dst_phase in allowed
        return super().__contains__(item)


VALID_SWARM_EDGES: Dict[SwarmPhase, FrozenSet[SwarmPhase]] = _SwarmEdgeMap(
    {
        SwarmPhase.RESEARCH: frozenset({SwarmPhase.CODE}),
        SwarmPhase.CODE: frozenset({SwarmPhase.REVIEW}),
        SwarmPhase.REVIEW: frozenset(
            {SwarmPhase.APPROVED, SwarmPhase.CODE, SwarmPhase.REJECTED}
        ),
        SwarmPhase.APPROVED: frozenset(),
        SwarmPhase.REJECTED: frozenset(),
    }
)


class EvidenceFinding(BaseModel):
    """Single grounded research evidence item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchBrief(BaseModel):
    """Frozen output artifact produced by ResearcherAgent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: str
    constraints: List[str]
    findings: List[EvidenceFinding]
    target_files: List[str]
    acceptance_criteria: List[str]


class CodeArtifact(BaseModel):
    """Frozen code implementation + unit test artifact produced by CoderAgent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(ge=1)
    file_path: str
    language: str
    implementation: str
    unit_test_code: str
    addressed_findings: List[str] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    """Actionable review defect emitted by ReviewerAgent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Literal["P0", "P1", "P2"]
    location: str
    description: str
    remediation: str


class ReviewVerdict(BaseModel):
    """Frozen review gate decision produced by ReviewerAgent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool
    revision_evaluated: int
    issues: List[ReviewIssue] = Field(default_factory=list)
    rationale: str


class SwarmTransitionRecord(BaseModel):
    """Immutable audit log entry for every legal state transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_phase: SwarmPhase
    to_phase: SwarmPhase
    actor: str
    payload_type: str
    timestamp_ms: int


class SwarmState(BaseModel):
    """Typed state container threaded across swarm nodes."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    objective: str
    phase: SwarmPhase = SwarmPhase.RESEARCH
    max_revisions: int = 2
    revision_count: int = 0
    research: Optional[ResearchBrief] = None
    artifact: Optional[CodeArtifact] = None
    verdict: Optional[ReviewVerdict] = None
    transitions: List[SwarmTransitionRecord] = Field(default_factory=list)


def assert_typed_handoff(payload: Any, expected_cls: Type[ModelT]) -> ModelT:
    """Validate an inter-agent handoff payload and forbid raw text messages."""
    if not (isinstance(expected_cls, type) and issubclass(expected_cls, BaseModel)):
        raise TypeError("expected_cls must be a Pydantic BaseModel subclass.")

    if isinstance(payload, (str, bytes, bytearray)):
        raise TypeError(
            "Raw text handoffs are forbidden in TypedMultiAgentSwarm; pass a typed Pydantic model."
        )

    if isinstance(payload, expected_cls):
        return payload

    if isinstance(payload, dict):
        return expected_cls.model_validate(payload)

    raise TypeError(
        f"Invalid handoff payload of type {type(payload).__name__}; "
        f"expected {expected_cls.__name__} or dict."
    )


def validate_swarm_transition(
    from_phase: SwarmPhase,
    to_phase: SwarmPhase,
    *,
    approved: Optional[bool] = None,
    revision_count: int = 0,
    max_revisions: int = 2,
) -> None:
    """Enforce legal edges and conditional REVIEW branch invariants."""
    allowed_targets = VALID_SWARM_EDGES.get(from_phase, frozenset())
    if to_phase not in allowed_targets:
        raise ValueError(
            f"Illegal swarm state transition: {from_phase.value} -> {to_phase.value}"
        )

    if from_phase == SwarmPhase.REVIEW:
        if approved is None:
            raise ValueError("Transitions out of REVIEW require an explicit 'approved' flag.")
        if to_phase == SwarmPhase.APPROVED and not approved:
            raise ValueError("Cannot transition REVIEW -> APPROVED when verdict.approved is False.")
        if to_phase == SwarmPhase.CODE:
            if approved:
                raise ValueError("Cannot transition REVIEW -> CODE when verdict.approved is True.")
            if revision_count >= max_revisions:
                raise ValueError(
                    f"Cannot transition REVIEW -> CODE: revision_count ({revision_count}) "
                    f">= max_revisions ({max_revisions})."
                )
        if to_phase == SwarmPhase.REJECTED:
            if approved:
                raise ValueError(
                    "Cannot transition REVIEW -> REJECTED when verdict.approved is True."
                )
            if revision_count < max_revisions:
                raise ValueError(
                    f"Cannot transition REVIEW -> REJECTED while revision budget remains "
                    f"({revision_count} < {max_revisions})."
                )


def _slugify_symbol(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if not tokens:
        return "generated_module"
    return "_".join(tokens[:4])


class ResearcherAgent:
    """Default deterministic researcher producing a strictly typed ResearchBrief."""

    def __call__(self, state: SwarmState) -> ResearchBrief:
        if isinstance(state, (str, bytes, bytearray)):
            raise TypeError(
                "Raw text handoffs are forbidden in TypedMultiAgentSwarm; pass a typed Pydantic model."
            )
        if not isinstance(state, SwarmState):
            raise TypeError(f"ResearcherAgent expected SwarmState, got {type(state).__name__}.")

        slug = _slugify_symbol(state.objective)
        return ResearchBrief(
            objective=state.objective,
            constraints=[
                "Strict Pydantic v2 schema validation with extra='forbid'",
                "Deterministic execution with zero raw-string handoffs",
            ],
            findings=[
                EvidenceFinding(
                    source="spec:INV-13",
                    summary=f"Synthesized typed interface requirements for '{state.objective}'.",
                    confidence=0.95,
                )
            ],
            target_files=[f"src/{slug}.py"],
            acceptance_criteria=[
                "Implementation must be non-empty and free of placeholder markers",
                "Unit test suite must include explicit assertions",
            ],
        )


class CoderAgent:
    """Default deterministic coder consuming ResearchBrief (+ optional ReviewVerdict)."""

    def __call__(self, state: SwarmState) -> CodeArtifact:
        if isinstance(state, (str, bytes, bytearray)):
            raise TypeError(
                "Raw text handoffs are forbidden in TypedMultiAgentSwarm; pass a typed Pydantic model."
            )
        if not isinstance(state, SwarmState):
            raise TypeError(f"CoderAgent expected SwarmState, got {type(state).__name__}.")

        brief = assert_typed_handoff(state.research, ResearchBrief)
        next_revision = state.revision_count + 1
        if state.artifact is not None and state.artifact.revision >= next_revision:
            next_revision = state.artifact.revision + 1

        addressed: List[str] = []
        if state.verdict is not None:
            prior_verdict = assert_typed_handoff(state.verdict, ReviewVerdict)
            addressed = [
                f"{issue.severity}:{issue.location} -> {issue.remediation}"
                for issue in prior_verdict.issues
            ]

        slug = _slugify_symbol(brief.objective)
        target_file = brief.target_files[0] if brief.target_files else f"src/{slug}.py"
        implementation = (
            f"def {slug}(payload: dict) -> dict:\n"
            f'    """Deterministic implementation for: {brief.objective}"""\n'
            f"    if not isinstance(payload, dict):\n"
            f'        raise TypeError("payload must be a dict")\n'
            f'    return {{"ok": True, "revision": {next_revision}, "data": dict(payload)}}\n'
        )
        unit_test_code = (
            f"def test_{slug}() -> None:\n"
            f'    result = {slug}({{"sample": 1}})\n'
            f'    assert result["ok"] is True\n'
            f'    assert result["revision"] == {next_revision}\n'
        )
        return CodeArtifact(
            revision=next_revision,
            file_path=target_file,
            language="python",
            implementation=implementation,
            unit_test_code=unit_test_code,
            addressed_findings=addressed,
        )


class ReviewerAgent:
    """Default deterministic reviewer auditing CodeArtifact against ResearchBrief."""

    _PLACEHOLDER_MARKERS: Tuple[str, ...] = ("TODO", "FIXME", "NotImplementedError")

    def __call__(self, state: SwarmState) -> ReviewVerdict:
        if isinstance(state, (str, bytes, bytearray)):
            raise TypeError(
                "Raw text handoffs are forbidden in TypedMultiAgentSwarm; pass a typed Pydantic model."
            )
        if not isinstance(state, SwarmState):
            raise TypeError(f"ReviewerAgent expected SwarmState, got {type(state).__name__}.")

        assert_typed_handoff(state.research, ResearchBrief)
        artifact = assert_typed_handoff(state.artifact, CodeArtifact)

        issues: List[ReviewIssue] = []
        impl = artifact.implementation.strip()
        tests = artifact.unit_test_code.strip()

        if not impl:
            issues.append(
                ReviewIssue(
                    severity="P0",
                    location=artifact.file_path,
                    description="Implementation body is empty.",
                    remediation="Provide a concrete module implementation.",
                )
            )

        for marker in self._PLACEHOLDER_MARKERS:
            if marker in impl:
                issues.append(
                    ReviewIssue(
                        severity="P1",
                        location=artifact.file_path,
                        description=f"Unresolved placeholder '{marker}' in implementation.",
                        remediation=f"Replace '{marker}' with production logic.",
                    )
                )

        if not tests or "assert" not in tests:
            issues.append(
                ReviewIssue(
                    severity="P1",
                    location=f"tests/{artifact.file_path.rsplit('/', 1)[-1]}",
                    description="Unit test suite is empty or missing explicit assertions.",
                    remediation="Add deterministic unit tests containing 'assert' statements.",
                )
            )

        blocking = [issue for issue in issues if issue.severity in ("P0", "P1")]
        approved = len(blocking) == 0
        rationale = (
            f"Revision {artifact.revision} passed all structural and test gates."
            if approved
            else f"Revision {artifact.revision} blocked by {len(blocking)} P0/P1 issue(s)."
        )
        return ReviewVerdict(
            approved=approved,
            revision_evaluated=artifact.revision,
            issues=issues,
            rationale=rationale,
        )


async def _invoke_callable(fn: Callable[..., Any], *args: Any) -> Any:
    result = fn(*args)
    if inspect.isawaitable(result):
        return await result
    return result


class TypedMultiAgentSwarm:
    """
    Deterministic typed state-machine swarm enforcing frozen Pydantic v2 handoffs.

    Validates every inter-agent output via ``assert_typed_handoff`` and restricts phase
    progression to ``VALID_SWARM_EDGES``.
    """

    def __init__(
        self,
        researcher_fn: Optional[Callable[..., Any]] = None,
        coder_fn: Optional[Callable[..., Any]] = None,
        reviewer_fn: Optional[Callable[..., Any]] = None,
        max_revisions: int = 2,
    ) -> None:
        if max_revisions < 0:
            raise ValueError("max_revisions must be >= 0.")
        self.researcher_fn: Callable[..., Any] = researcher_fn or ResearcherAgent()
        self.coder_fn: Callable[..., Any] = coder_fn or CoderAgent()
        self.reviewer_fn: Callable[..., Any] = reviewer_fn or ReviewerAgent()
        self.max_revisions: int = max_revisions

    @staticmethod
    def _positional_params(fn: Callable[..., Any]) -> List[inspect.Parameter]:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return []
        params: List[inspect.Parameter] = []
        for param in sig.parameters.values():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                params.append(param)
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                return [param]
        return params

    async def _call_researcher(self, state: SwarmState) -> Any:
        params = self._positional_params(self.researcher_fn)
        if not params:
            return await _invoke_callable(self.researcher_fn)
        if len(params) == 1 and params[0].kind != inspect.Parameter.VAR_POSITIONAL:
            p0 = params[0]
            if p0.annotation in (str, "str") or p0.name in (
                "objective",
                "query",
                "goal",
                "prompt",
            ):
                return await _invoke_callable(self.researcher_fn, state.objective)
            return await _invoke_callable(self.researcher_fn, state)
        if params[0].kind == inspect.Parameter.VAR_POSITIONAL:
            return await _invoke_callable(self.researcher_fn, state)
        return await _invoke_callable(self.researcher_fn, state.objective, state)

    async def _call_coder(self, state: SwarmState) -> Any:
        params = self._positional_params(self.coder_fn)
        if not params:
            return await _invoke_callable(self.coder_fn)
        if len(params) == 1 and params[0].kind != inspect.Parameter.VAR_POSITIONAL:
            p0 = params[0]
            if p0.annotation in (ResearchBrief, "ResearchBrief") or p0.name in (
                "research",
                "brief",
                "research_brief",
            ):
                return await _invoke_callable(self.coder_fn, state.research)
            return await _invoke_callable(self.coder_fn, state)
        if params[0].kind == inspect.Parameter.VAR_POSITIONAL:
            return await _invoke_callable(self.coder_fn, state)
        if len(params) == 2:
            if params[0].name in ("state", "swarm_state"):
                return await _invoke_callable(self.coder_fn, state, state.verdict)
            return await _invoke_callable(self.coder_fn, state.research, state.verdict)
        return await _invoke_callable(
            self.coder_fn, state.research, state.verdict, state.revision_count + 1
        )

    async def _call_reviewer(self, state: SwarmState) -> Any:
        params = self._positional_params(self.reviewer_fn)
        if not params:
            return await _invoke_callable(self.reviewer_fn)
        if len(params) == 1 and params[0].kind != inspect.Parameter.VAR_POSITIONAL:
            p0 = params[0]
            if p0.annotation in (CodeArtifact, "CodeArtifact") or p0.name in (
                "artifact",
                "code",
                "code_artifact",
            ):
                return await _invoke_callable(self.reviewer_fn, state.artifact)
            return await _invoke_callable(self.reviewer_fn, state)
        if params[0].kind == inspect.Parameter.VAR_POSITIONAL:
            return await _invoke_callable(self.reviewer_fn, state)
        p0_name = params[0].name
        if p0_name in ("research", "brief", "research_brief") or params[0].annotation in (
            ResearchBrief,
            "ResearchBrief",
        ):
            return await _invoke_callable(self.reviewer_fn, state.research, state.artifact)
        if p0_name in ("state", "swarm_state"):
            return await _invoke_callable(self.reviewer_fn, state, state.artifact)
        return await _invoke_callable(self.reviewer_fn, state.artifact, state.research)

    @staticmethod
    def _record_transition(
        state: SwarmState,
        to_phase: SwarmPhase,
        *,
        actor: str,
        payload_type: str,
        approved: Optional[bool] = None,
    ) -> None:
        validate_swarm_transition(
            state.phase,
            to_phase,
            approved=approved,
            revision_count=state.revision_count,
            max_revisions=state.max_revisions,
        )
        record = SwarmTransitionRecord(
            from_phase=state.phase,
            to_phase=to_phase,
            actor=actor,
            payload_type=payload_type,
            timestamp_ms=int(time.time() * 1000),
        )
        state.transitions.append(record)
        state.phase = to_phase

    async def run(self, objective: str, task_id: Optional[str] = None) -> SwarmState:
        """Execute the RESEARCH -> CODE -> REVIEW state machine to completion."""
        if not isinstance(objective, str):
            raise TypeError("objective must be a string.")
        cleaned_objective = objective.strip()
        if not cleaned_objective:
            raise ValueError("objective must be a non-empty string.")

        resolved_task_id = task_id or f"swarm-{uuid.uuid4().hex[:12]}"
        state = SwarmState(
            task_id=resolved_task_id,
            objective=cleaned_objective,
            phase=SwarmPhase.RESEARCH,
            max_revisions=self.max_revisions,
            revision_count=0,
        )

        # Phase 1: RESEARCH -> CODE
        raw_research = await self._call_researcher(state)
        brief = assert_typed_handoff(raw_research, ResearchBrief)
        state.research = brief
        self._record_transition(
            state,
            SwarmPhase.CODE,
            actor="ResearcherAgent",
            payload_type=type(brief).__name__,
        )

        # Bounded CODE <-> REVIEW loop
        max_steps = (self.max_revisions + 1) * 2 + 2
        steps = 0
        while state.phase not in (SwarmPhase.APPROVED, SwarmPhase.REJECTED):
            steps += 1
            if steps > max_steps:
                raise RuntimeError(
                    f"Swarm exceeded bounded step guard ({max_steps}) for task {state.task_id}."
                )

            if state.phase == SwarmPhase.CODE:
                assert_typed_handoff(state.research, ResearchBrief)
                raw_artifact = await self._call_coder(state)
                artifact = assert_typed_handoff(raw_artifact, CodeArtifact)
                state.artifact = artifact
                self._record_transition(
                    state,
                    SwarmPhase.REVIEW,
                    actor="CoderAgent",
                    payload_type=type(artifact).__name__,
                )
            elif state.phase == SwarmPhase.REVIEW:
                assert_typed_handoff(state.artifact, CodeArtifact)
                raw_verdict = await self._call_reviewer(state)
                verdict = assert_typed_handoff(raw_verdict, ReviewVerdict)
                state.verdict = verdict

                if verdict.approved:
                    self._record_transition(
                        state,
                        SwarmPhase.APPROVED,
                        actor="ReviewerAgent",
                        payload_type=type(verdict).__name__,
                        approved=True,
                    )
                elif state.revision_count < state.max_revisions:
                    self._record_transition(
                        state,
                        SwarmPhase.CODE,
                        actor="ReviewerAgent",
                        payload_type=type(verdict).__name__,
                        approved=False,
                    )
                    state.revision_count += 1
                else:
                    self._record_transition(
                        state,
                        SwarmPhase.REJECTED,
                        actor="ReviewerAgent",
                        payload_type=type(verdict).__name__,
                        approved=False,
                    )

        return state


__all__ = (
    "CodeArtifact",
    "CoderAgent",
    "EvidenceFinding",
    "ResearchBrief",
    "ResearcherAgent",
    "ReviewIssue",
    "ReviewVerdict",
    "ReviewerAgent",
    "SwarmPhase",
    "SwarmState",
    "SwarmTransitionRecord",
    "TypedMultiAgentSwarm",
    "VALID_SWARM_EDGES",
    "assert_typed_handoff",
    "validate_swarm_transition",
)
