"""Automatic `.act` Source File-Out Manager (`actorii.guardrails.fileout`).

Solves the classic Smalltalk/Actor "Image Rot" problem where dynamically
evaluated or JIT-synthesized methods live only inside the binary/JSON `.img`
snapshot and become invisible to Git version control.

Whenever a class or method is admitted into the live `ActorVM`, `ActorFileOutManager`
exports a clean, deterministic, Git-diffable `.act` source file for that class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from actorii.vm import CompiledMethod


class ActorFileOutManager:
    """Exports live `ActorClass` definitions and `CompiledMethod` sources to `.act` files."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_class_source(self, vm: Any, class_name: str) -> str:
        """Generate a complete, standalone 1993 `.act` source file for `class_name`."""
        cls = vm.hierarchy.get(class_name)
        if cls is None:
            raise KeyError(f"Class '{class_name}' not found in ActorVM hierarchy")

        super_name = cls._bases[0] if getattr(cls, "_bases", None) else "Object"
        ivars: List[str] = list(getattr(cls, "ivars", []))
        ivar_literals = " ".join(f"#{iv}" for iv in ivars)

        lines: List[str] = [
            f"/* =========================================================================",
            f" * FILE-OUT: {class_name}.act",
            f" * Auto-exported from live ActorII Image Hierarchy",
            f" * Superclass: {super_name} | Instance Variables ({len(ivars)}): {ivars}",
            f" * ========================================================================= */",
            "",
            f"inherit({super_name}, #{class_name}, #({ivar_literals}), {len(ivars)}, nil)!!",
            f"now({class_name})!!",
            "",
        ]

        methods_dict = getattr(cls, "_class_methods", {})
        for sel in sorted(methods_dict.keys()):
            if sel == "new":
                continue
            m = methods_dict[sel]
            if isinstance(m, CompiledMethod) and getattr(m, "source", None):
                src = m.source.strip()
                if not src.endswith("!!"):
                    src = src.rstrip("!") + "!!"
                tag = "Synthesized (JIT)" if getattr(m, "synthesized", False) else "Compiled"
                tier = getattr(m, "risk_tier", None)
                tier_str = f" | RiskTier={int(tier)}" if tier is not None else ""
                lines.append(f"/* Method: {class_name}>>{sel} [{tag}{tier_str}] */")
                lines.append(src)
                lines.append("")

        actor_pool = vm.global_vars.get("Actor", {})
        inst = actor_pool.get(class_name)
        if inst is not None and hasattr(inst, "_fields") and isinstance(inst._fields, dict):
            lines.append(f"Actor[#{class_name}] := new({class_name})!!")
            for fk, fv in inst._fields.items():
                if fv is not None and isinstance(fv, (int, float, str, bool)):
                    if isinstance(fv, bool):
                        val_lit = "true" if fv else "false"
                    elif isinstance(fv, str):
                        esc = fv.replace("\\", "\\\\").replace('"', '\\"')
                        val_lit = f'"{esc}"'
                    else:
                        val_lit = repr(fv)
                    lines.append(f"Actor[#{class_name}].{fk} := {val_lit}!!")

        return "\n".join(lines).rstrip() + "\n"

    def file_out_class(self, vm: Any, class_name: str) -> Path:
        """Write `<output_dir>/<class_name>.act` atomically and return its path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.output_dir / f"{class_name}.act"
        tmp_path = self.output_dir / f".{class_name}.act.tmp"
        content = self.format_class_source(vm, class_name)
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(target_path)
        return target_path
