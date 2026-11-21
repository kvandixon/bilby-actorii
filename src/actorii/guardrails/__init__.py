"""Guardrails Subsystem for ActorII (`actorii.guardrails`).

Provides:
- `ActorLinter`, `LintReport`, `LintIssue`, `RiskTier`: Static AST/bytecode verification & risk classification.
- `ActorCodeJudge`, `ZeroVramSystemOneJudge`, `JudgeVerdict`: Deterministic + zero-VRAM CPU System 1 (`:11434` / `:8082`) policy gate.
- `ActorFileOutManager`: Automatic `.act` source code export for Git reproducibility.
- `ActorGuardrailError`, `ProtectedClassError`, `OpcodeBudgetExceededError`: Guardrail exceptions.
"""

from actorii.guardrails.fileout import ActorFileOutManager
from actorii.guardrails.judge import (
    SPECIALIST_CPU_8082_URL,
    SYSTEMONE_CPU_URL,
    ActorCodeJudge,
    JudgeVerdict,
    ZeroVramSystemOneJudge,
)
from actorii.guardrails.linter import (
    PROTECTED_POOL_SINGLETONS,
    PROTECTED_SYSTEM_CLASSES,
    RING1_MUTABLE_HARNESS_SELECTORS,
    SYNTHESIZABLE_TWIN_CLASSES,
    ActorGuardrailError,
    ActorLinter,
    LintIssue,
    LintReport,
    OpcodeBudgetExceededError,
    ProtectedClassError,
    RiskTier,
)

__all__ = [
    "PROTECTED_POOL_SINGLETONS",
    "PROTECTED_SYSTEM_CLASSES",
    "RING1_MUTABLE_HARNESS_SELECTORS",
    "SPECIALIST_CPU_8082_URL",
    "SYNTHESIZABLE_TWIN_CLASSES",
    "SYSTEMONE_CPU_URL",
    "ActorCodeJudge",
    "ActorFileOutManager",
    "ActorGuardrailError",
    "ActorLinter",
    "JudgeVerdict",
    "LintIssue",
    "LintReport",
    "OpcodeBudgetExceededError",
    "ProtectedClassError",
    "RiskTier",
    "ZeroVramSystemOneJudge",
]
