"""First-Class OOP `Agent` Class for the 1993 Whitewater Actor 4.1 Hierarchy.

Integrates AI reasoning, semantic/episodic memory, object-graph introspection,
tool invocation, multi-agent message delegation, and live JIT bytecode method
synthesis directly into Actor's single-root `Object` hierarchy.

Supports:
- Native instantiation: `new(Agent)!!` or `new(Agent, "Planner", "gpt-oss:20b")!!`
- Subclassing in 1993 Actor 4.1 syntax:
  `inherit(Agent, #SecurityAgent, #(#severity #domain), 2, nil)!!`
- Persistent semantic memory (`Dictionary`) and episodic conversation history
  (`ActorOrderedCollection`) that survive `.img` image snapshots (`ACTOR.IMG`).
- Live object inspection (`inspectObject`), tool registration (`registerTool` / `useTool`),
  multi-agent delegation (`delegateTo`), and live method synthesis (`synthesizeMethod`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from actorii.ai import (
    DEFAULT_VALKYRIE_ENDPOINTS,
    ActorMethodSynthesizer,
    ValkyrieLLMClient,
    format_inspection_for_prompt,
    inspect_actor_object,
    resolve_default_endpoint_for_model,
)
from actorii.classes.collections import ActorOrderedCollection
from actorii.classes.dictionary import Dictionary
from actorii.core import ActorClass, ActorObject, Symbol


def _norm_key(key: Any) -> str:
    if isinstance(key, Symbol):
        return key.name
    s = str(key)
    return s[1:] if s.startswith("#") else s


class Agent(ActorObject):
    """First-class autonomous AI Agent in the ActorII object hierarchy."""

    def __init__(
        self,
        name: str = "Agent",
        model: str = "qwen3.8:latest",
        system_prompt: str = "You are a helpful object-oriented AI Agent living inside the 1993 ActorII virtual machine.",
        endpoint: Optional[str] = None,
        auto_synthesize: bool = True,
        obj_id: Optional[str] = None,
        class_name: str = "Agent",
        vm: Optional[Any] = None,
    ) -> None:
        super().__init__(obj_id=obj_id)
        self._class_name = class_name
        self.vm = vm

        # Native Actor collections for semantic memory, episodic history, and tools
        self._memory = Dictionary(32)
        self._history = ActorOrderedCollection()
        self._tools = Dictionary(16)

        self._fields["name"] = str(name)
        self._fields["model"] = str(model)
        self._fields["endpoint"] = resolve_default_endpoint_for_model(str(model), endpoint)
        self._fields["systemPrompt"] = str(system_prompt)
        self._fields["memory"] = self._memory
        self._fields["history"] = self._history
        self._fields["tools"] = self._tools
        self._fields["lastResponse"] = None
        self._fields["autoSynthesize"] = bool(auto_synthesize)

        self.custom_provider: Optional[Callable[..., str]] = None

    # =========================================================================
    # Synchronized Properties
    # =========================================================================

    @property
    def name(self) -> str:
        return str(self._fields.get("name", "Agent"))

    @name.setter
    def name(self, val: str) -> None:
        self._fields["name"] = str(val)

    @property
    def model(self) -> str:
        return str(self._fields.get("model", "qwen3.8:latest"))

    @model.setter
    def model(self, val: str) -> None:
        self._fields["model"] = str(val)
        self._fields["endpoint"] = resolve_default_endpoint_for_model(str(val))

    @property
    def endpoint(self) -> str:
        return str(self._fields.get("endpoint", DEFAULT_VALKYRIE_ENDPOINTS[0]))

    @endpoint.setter
    def endpoint(self, val: str) -> None:
        self._fields["endpoint"] = str(val)

    @property
    def system_prompt(self) -> str:
        return str(self._fields.get("systemPrompt", ""))

    @system_prompt.setter
    def system_prompt(self, val: str) -> None:
        self._fields["systemPrompt"] = str(val)

    @property
    def auto_synthesize(self) -> bool:
        return bool(self._fields.get("autoSynthesize", True))

    @auto_synthesize.setter
    def auto_synthesize(self, val: bool) -> None:
        self._fields["autoSynthesize"] = bool(val)

    @property
    def last_response(self) -> Optional[str]:
        val = self._fields.get("lastResponse")
        return str(val) if val is not None else None

    @last_response.setter
    def last_response(self, val: Optional[str]) -> None:
        self._fields["lastResponse"] = val

    # =========================================================================
    # 1993 Actor Message Protocol Methods
    # =========================================================================

    def getName(self) -> str:
        return self.name

    def setName(self, name: str) -> "Agent":
        self.name = str(name)
        return self

    def getModel(self) -> str:
        return self.model

    def setModel(self, model: str) -> "Agent":
        self.model = str(model)
        return self

    def getEndpoint(self) -> str:
        return self.endpoint

    def setEndpoint(self, endpoint: str) -> "Agent":
        self.endpoint = str(endpoint)
        return self

    def systemPrompt(self) -> str:
        return self.system_prompt

    def setSystemPrompt(self, prompt: str) -> "Agent":
        self.system_prompt = str(prompt)
        return self

    def autoSynthesize(self) -> bool:
        return self.auto_synthesize

    def setAutoSynthesize(self, enabled: Any) -> "Agent":
        self.auto_synthesize = bool(enabled)
        return self

    def lastResponse(self) -> Optional[str]:
        return self.last_response

    def memory(self) -> Dictionary:
        mem = self._fields.get("memory")
        if isinstance(mem, Dictionary):
            self._memory = mem
        return self._memory

    def history(self) -> ActorOrderedCollection:
        hist = self._fields.get("history")
        if isinstance(hist, ActorOrderedCollection):
            self._history = hist
        return self._history

    def tools(self) -> Dictionary:
        tls = self._fields.get("tools")
        if isinstance(tls, Dictionary):
            self._tools = tls
        return self._tools

    def remember(self, key: Any, value: Any) -> Any:
        """Store a semantic fact or object reference in the Agent's persistent Dictionary."""
        norm = _norm_key(key)
        mem = self.memory()
        mem.atPut(norm, value)
        mem.atPut(Symbol(norm), value)
        return value

    def recall(self, key: Any) -> Any:
        """Retrieve a value from the Agent's semantic memory by key or Symbol."""
        norm = _norm_key(key)
        mem = self.memory()
        val = mem.at(norm)
        if val is None:
            val = mem.at(Symbol(norm))
        if val is None:
            val = mem.at(f"#{norm}")
        return val

    def forget(self, key: Any) -> Any:
        """Remove a key from the Agent's semantic memory."""
        norm = _norm_key(key)
        mem = self.memory()
        removed = mem.removeKey(norm)
        mem.removeKey(Symbol(norm))
        return removed

    def clearHistory(self) -> "Agent":
        """Clear episodic conversation history."""
        self.history().clear()
        return self

    def registerTool(self, tool_name: Any, target: Any) -> "Agent":
        """Register an ActorObject, BlockClosure, or callable tool under `tool_name`."""
        norm = _norm_key(tool_name)
        tls = self.tools()
        tls.atPut(norm, target)
        tls.atPut(Symbol(norm), target)
        return self

    def useTool(self, tool_name: Any, *args: Any) -> Any:
        """Invoke a registered tool (BlockClosure, ActorObject, or Python callable)."""
        from actorii.vm import ActorVM, BlockClosure

        norm = _norm_key(tool_name)
        tls = self.tools()
        target = tls.at(norm)
        if target is None:
            target = tls.at(Symbol(norm))
        if target is None:
            raise KeyError(f"Agent '{self.name}' has no registered tool '{norm}'")

        active_vm = self.vm or getattr(ActorVM, "_active_vm", None)

        if isinstance(target, BlockClosure):
            if active_vm is not None:
                return active_vm._call_block_closure(target, list(args))
        if isinstance(target, ActorObject):
            if active_vm is not None:
                if target.has_method(norm) or active_vm.hierarchy.resolve_method(target.class_name, norm):
                    return active_vm.dispatch_message(target, norm, list(args))
                if target.has_method("execute") or active_vm.hierarchy.resolve_method(target.class_name, "execute"):
                    return active_vm.dispatch_message(target, "execute", list(args))
                if target.has_method("run") or active_vm.hierarchy.resolve_method(target.class_name, "run"):
                    return active_vm.dispatch_message(target, "run", list(args))
        if callable(target):
            return target(*args)
        return target

    def invokeTool(self, tool_name: Any, *args: Any) -> Any:
        return self.useTool(tool_name, *args)

    def inspectObject(self, target: Any) -> str:
        """Introspect any live Actor object or class and return a structured JSON summary."""
        from actorii.vm import ActorVM

        active_vm = self.vm or getattr(ActorVM, "_active_vm", None)
        hierarchy = active_vm.hierarchy if active_vm is not None else None
        return format_inspection_for_prompt(target, hierarchy=hierarchy)

    def buildContextPrompt(self) -> str:
        """Build the system prompt enriched with the Agent's live fields, memory, and tools."""
        lines = [self.system_prompt]
        lines.append(f"\n[Agent Identity]: name={self.name!r}, class={self.class_name!r}, model={self.model!r}")

        # Include custom subclass fields (excluding volatile per-step AgentHarness fields so system prompt stays KV-cache invariant)
        standard_keys = {
            "name",
            "model",
            "endpoint",
            "systemPrompt",
            "memory",
            "history",
            "tools",
            "lastResponse",
            "autoSynthesize",
            "sandbox",
            "goal",
            "status",
            "stepCount",
            "maxSteps",
            "failures",
            "maxFailures",
            "scratchpad",
            "stepLog",
            "lastAction",
            "lastResult",
            "lastError",
            "gasLimit",
            "totalGasUsed",
            "chatLog",
            "taskContract",
            "lastErrorContext",
            "tokenBudget",
        }
        custom_fields = {
            k: v for k, v in self._fields.items() if k not in standard_keys
        }
        if custom_fields:
            lines.append(f"[Agent Instance Fields]: {custom_fields}")

        # Include semantic memory entries only for non-AgentHarness objects (AgentHarness formats scratchpad at the end of user prompt)
        if self.class_name != "AgentHarness":
            mem_dict: Dict[str, Any] = {}
            for k, v in self.memory()._entries.items():
                mem_dict[_norm_key(k)] = v
            if mem_dict:
                lines.append(f"[Agent Semantic Memory]: {mem_dict}")

        # Include registered tool names
        tool_names = sorted({_norm_key(k) for k in self.tools()._entries.keys()})
        if tool_names:
            lines.append(f"[Registered Tools]: {tool_names}")

        return "\n".join(lines)

    def ask(self, prompt: Any, context_obj: Any = None) -> str:
        """Send a prompt to the Agent, incorporating live object state, memory, and history."""
        from actorii.vm import ActorVM

        active_vm = self.vm or getattr(ActorVM, "_active_vm", None)
        user_text = str(prompt)
        if context_obj is not None:
            inspection = self.inspectObject(context_obj)
            user_text = f"{user_text}\n\n[Live Object Context]:\n{inspection}"

        sys_prompt = self.buildContextPrompt()
        history_list: List[Dict[str, str]] = []
        for item in self.history():
            if isinstance(item, dict) and "role" in item and "content" in item:
                history_list.append({"role": str(item["role"]), "content": str(item["content"])})

        provider = self.custom_provider or (
            getattr(active_vm, "ai_provider", None) if active_vm is not None else None
        )

        metadata = {
            "mode": "agent_ask",
            "agent": self,
            "agent_name": self.name,
            "class_name": self.class_name,
            "memory": { _norm_key(k): v for k, v in self.memory()._entries.items() },
            "context_obj": context_obj,
        }

        client = ValkyrieLLMClient(
            endpoint=self.endpoint,
            model=self.model,
            custom_provider=provider,
        )

        try:
            reply = client.complete(
                prompt=user_text,
                system_prompt=sys_prompt,
                history=history_list,
                metadata=metadata,
            )
        except Exception as exc:
            # Deterministic offline fallback summarizing agent state + prompt
            mem_summary = { _norm_key(k): v for k, v in self.memory()._entries.items() }
            reply = (
                f"[{self.name} ({self.class_name})] Processed: {prompt} "
                f"(memory_keys={sorted(mem_summary.keys())}, error={exc})"
            )

        self.history().add({"role": "user", "content": str(prompt)})
        self.history().add({"role": "assistant", "content": str(reply)})
        self.last_response = str(reply)
        return str(reply)

    def think(self, prompt: Any, context_obj: Any = None) -> str:
        """Alias for `ask`."""
        return self.ask(prompt, context_obj=context_obj)

    def delegateTo(self, other_agent: "Agent", task_prompt: Any) -> str:
        """Delegate a task to another `Agent` via OOP message passing and record the result."""
        if not isinstance(other_agent, Agent):
            raise TypeError(f"delegateTo expects an Agent receiver, got {type(other_agent).__name__}")
        framed_prompt = f"[Delegated from {self.name} ({self.class_name})]: {task_prompt}"
        response = other_agent.ask(framed_prompt)
        self.remember(f"delegated_{other_agent.name}", response)
        self.history().add(
            {
                "role": "assistant",
                "content": f"Delegated to {other_agent.name}: {task_prompt} -> {response}",
            }
        )
        self.last_response = response
        return response

    def synthesizeMethod(
        self,
        target: Any,
        selector: Any,
        spec_prompt: str = "",
        arity: int = 0,
    ) -> Any:
        """Synthesize a new 1993 Actor bytecode method and attach it to `target`'s class."""
        from actorii.vm import ActorVM

        active_vm = self.vm or getattr(ActorVM, "_active_vm", None)
        if active_vm is None:
            raise RuntimeError("synthesizeMethod requires an active ActorVM")

        sel_name = _norm_key(selector)
        if isinstance(target, ActorClass):
            cls_name = target.name
            receiver_obj = target._default_constructor()
        elif isinstance(target, (str, Symbol)):
            cls_name = _norm_key(target)
            cls_obj = active_vm.hierarchy.get(cls_name)
            receiver_obj = cls_obj._default_constructor() if cls_obj else ActorObject()
        else:
            receiver_obj = target
            cls_name = active_vm._get_class_name(target)

        dummy_args = [0] * int(arity)
        synthesizer = ActorMethodSynthesizer(active_vm)
        cm = synthesizer.synthesize_method(
            receiver=receiver_obj,
            class_name=cls_name,
            selector=sel_name,
            args=dummy_args,
            spec_prompt=str(spec_prompt or ""),
            use_llm=True if (self.custom_provider or getattr(active_vm, "ai_provider", None) or getattr(active_vm, "use_live_llm_for_jit", False)) else False,
        )
        self.remember(f"synthesized_{cls_name}_{sel_name}", getattr(cm, "source", ""))
        return getattr(cm, "source", cm)

    def asString(self) -> str:
        return f"<{self.class_name} name={self.name!r} model={self.model!r}>"

    def __repr__(self) -> str:
        return self.asString()
