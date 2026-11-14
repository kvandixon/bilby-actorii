"""ActorII AI + OOP Bridge: JIT Method Synthesis & Local Valkyrie LLM Integration.

Combines 1993 Whitewater Actor 4.1 pure object-oriented message passing and live
image persistence with modern LLM reasoning and dynamic bytecode synthesis:

1. `doesNotUnderstand` Dynamic JIT Method Synthesis (`ActorMethodSynthesizer`):
   When an object receives an unknown message selector, inspects the live object's
   class hierarchy, instance variables (`_fields`), existing methods, and runtime
   arguments, synthesizes valid 1993 Actor `Def selector(self, ...) { ... }!!` source,
   compiles it into a `CompiledMethod` bytecode object, attaches it permanently to
   the `ActorClass` (persisting in `.img` snapshots), and executes it immediately.
2. Live Object Graph Introspection (`inspect_actor_object`):
   Serializes any live Actor object, collection, dictionary, class, or GUI window
   into a structured schema for `Agent` reasoning.
3. Local Valkyrie LLM Client (`ValkyrieLLMClient`):
   Zero-dependency HTTP client supporting local `valkyrie` OpenAI-compatible
   endpoints (`:8080` GPU primary, `:8082` zero-VRAM CPU specialist slot) and
   Ollama (`:11434`), plus pluggable custom Python provider callables.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from actorii.core import ActorClass, ActorInheritanceHierarchy, ActorObject, Symbol

_AI_LOCK = threading.RLock()

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>[\s\S]*$")
_FENCE_EXTRACT_RE = re.compile(
    r"```(?:actor|act|smalltalk|c|javascript|python)?\s*(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)
_STEP_NUM_RE = re.compile(
    r"^step:\s*(\d+)(?:\s+of\s+\d+|\s+\(unlimited\))?",
    flags=re.MULTILINE,
)
_DEF_CHUNK_RE = re.compile(
    r"(Def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*?\)\s*\{.*?\}\s*(?:!!)?)",
    flags=re.DOTALL,
)
_RETURN_STMT_RE = re.compile(r"\breturn\s+([^;]+);")
_LOCAL_VAR_DECL_RE = re.compile(r"\b(?:local|var|let)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:=")
_DEF_HEADER_RE = re.compile(r"^Def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")

DEFAULT_VALKYRIE_ENDPOINTS = (
    os.environ.get("ACTORII_LLM_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions"),
    os.environ.get("ACTORII_SPECIALIST_ENDPOINT", "http://127.0.0.1:8082/v1/chat/completions"),
)

PROVIDER_DEFAULT_ENDPOINTS: Dict[str, str] = {
    "llamacpp": "http://127.0.0.1:8080/v1/chat/completions",
    "ollama": "http://127.0.0.1:11434/v1/chat/completions",
    "vllm": "http://127.0.0.1:8000/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


def resolve_llm_api_key() -> str:
    """Resolve optional API key for cloud or authenticated LLM endpoints."""
    for env_var in (
        "ACTORII_LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
    ):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
    return ""


def _is_local_llamacpp_or_vllm(url: str) -> bool:
    """Return True when `url` targets a local llama.cpp or vLLM OpenAI-compatible server that accepts `chat_template_kwargs` / `force_no_think`."""
    prov = os.environ.get("ACTORII_LLM_PROVIDER", "").strip().lower()
    if prov in ("openai", "openrouter", "gemini", "anthropic"):
        return False
    if prov in ("llamacpp", "vllm"):
        return True
    u = str(url or "").lower()
    if any(cloud in u for cloud in ("api.openai.com", "openrouter.ai", "googleapis.com", "anthropic.com")):
        return False
    return any(loc in u for loc in ("127.0.0.1", "localhost", "host.docker.internal", ":8080", ":8082", ":18088"))

ACTOR_SYNTHESIS_SYSTEM_PROMPT = """You are an expert 1993 Whitewater Actor 4.1 compiler and method synthesizer.
Your job is to write a single valid 1993 Actor 4.1 method definition (`Def ... !!`) for the requested class and message selector.

STRICT 1993 ACTOR 4.1 SYNTAX RULES:
1. Method header syntax:
   `Def methodName(self, param1, param2 | local1, local2)`
   (Omit `| local1` if no local variables are needed: `Def methodName(self, param1)`).
2. Method body is enclosed in `{ ... }!!` and MUST end with `!!`.
3. Every statement inside `{ ... }` ends with a semicolon `;`.
4. Return statements use a caret `^`:
   `^expression;`
5. Variable assignment uses `:=`:
   `local1 := param1 + 10;`
   `self.fieldName := newValue;`
6. Instance variables / fields on `self` MUST be accessed as `self.fieldName`.
7. Conditionals use:
   `if condition then stmt1; else stmt2; endif;`
8. Loops use:
   `loop while condition begin stmt1; endLoop;`
9. All method calls use prefix functional notation:
   `selector(receiver, arg1, arg2)`
    Built-in math primitives: `max(a, b)`, `min(a, b)`, `abs(x)`, `sqrt(x)`, `inc(x)`, `dec(x)`, `asInt(x)`.

Output ONLY the raw 1993 Actor method definition (`Def ... { ... }!!`) with no markdown fences or explanation."""


def _strip_thinking_only(text: str) -> str:
    """Remove closed `<think>...</think>` and truncated unclosed `<think>...` blocks without touching Markdown fences."""
    if not text:
        return ""
    if "<think>" not in text:
        return text.strip()
    cleaned = _THINK_CLOSED_RE.sub("", text)
    if "<think>" in cleaned:
        cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    return cleaned.strip()


def _strip_thinking_and_fences(text: str) -> str:
    """Remove <think>...</think> blocks and markdown code fences from raw code synthesis output."""
    cleaned = _strip_thinking_only(text)
    if not cleaned:
        return ""
    if "```" in cleaned:
        fence_match = _FENCE_EXTRACT_RE.search(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()
    return cleaned


def resolve_default_endpoint_for_model(model: str, explicit_endpoint: Optional[str] = None) -> str:
    """Route CPU/Ollama specialist models (`qwen2.5-coder*`, `*-cpu*`) to `:11434/api/chat` and GPU/cloud models to `ACTORII_LLM_ENDPOINT` or `ACTORII_LLM_PROVIDER`."""
    if explicit_endpoint:
        return str(explicit_endpoint)
    env_ep = os.environ.get("ACTORII_LLM_ENDPOINT", "").strip()
    if env_ep:
        return env_ep
    prov = os.environ.get("ACTORII_LLM_PROVIDER", "").strip().lower()
    if prov in PROVIDER_DEFAULT_ENDPOINTS:
        return PROVIDER_DEFAULT_ENDPOINTS[prov]
    m = str(model or "").lower()
    if any(k in m for k in ("-cpu", "nomic-embed")):
        return os.environ.get("ACTORII_OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
    if "8082" in m:
        return os.environ.get("ACTORII_SPECIALIST_ENDPOINT", "http://127.0.0.1:8082/v1/chat/completions")
    if m.startswith(("gpt-4", "gpt-5", "o1-", "o3-")) and os.environ.get("OPENAI_API_KEY"):
        return PROVIDER_DEFAULT_ENDPOINTS["openai"]
    if "/" in m and os.environ.get("OPENROUTER_API_KEY"):
        return PROVIDER_DEFAULT_ENDPOINTS["openrouter"]
    if m.startswith("gemini-") and os.environ.get("GEMINI_API_KEY"):
        return PROVIDER_DEFAULT_ENDPOINTS["gemini"]
    return DEFAULT_VALKYRIE_ENDPOINTS[0]


class ValkyrieLLMClient:
    """Client for local `valkyrie` LLM endpoints (:8080, :8082, :11434) or custom callables."""

    GLOBAL_PROMPT_TOKENS: int = 0
    GLOBAL_COMPLETION_TOKENS: int = 0
    GLOBAL_TOTAL_TOKENS: int = 0
    GLOBAL_API_CALLS: int = 0

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "default",
        timeout: float = 360.0,
        custom_provider: Optional[Callable[..., str]] = None,
    ) -> None:
        self._explicit_endpoint = endpoint
        self.endpoint = resolve_default_endpoint_for_model(model, endpoint)
        self.model = model
        self.timeout = timeout
        self.custom_provider = custom_provider
        self._lock = threading.RLock()
        self._local = threading.local()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_api_calls = 0
        self.last_call_stats: Dict[str, Any] = {}

    def _record_usage_and_stats(
        self,
        p_tok: int,
        c_tok: int,
        t_tok: int,
        calls: int = 1,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self.total_prompt_tokens += p_tok
            self.total_completion_tokens += c_tok
            self.total_tokens += t_tok
            self.total_api_calls += calls
            if stats is not None:
                self.last_call_stats = dict(stats)
        with _AI_LOCK:
            ValkyrieLLMClient.GLOBAL_PROMPT_TOKENS += p_tok
            ValkyrieLLMClient.GLOBAL_COMPLETION_TOKENS += c_tok
            ValkyrieLLMClient.GLOBAL_TOTAL_TOKENS += t_tok
            ValkyrieLLMClient.GLOBAL_API_CALLS += calls
        if stats is not None:
            self._local.last_call_stats = dict(stats)

    def get_thread_last_call_stats(self) -> Dict[str, Any]:
        """Return the most recent call stats for the current thread (or client fallback)."""
        local_stats = getattr(self._local, "last_call_stats", None)
        if isinstance(local_stats, dict):
            return dict(local_stats)
        with self._lock:
            return dict(self.last_call_stats)

    def complete_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1600,
        temperature: float = 0.2,
    ) -> str:
        """Execute a chat completion using an explicit KV-cache-aligned `messages` array."""
        endpoints_to_try = [self.endpoint]
        if not self._explicit_endpoint and _is_local_llamacpp_or_vllm(self.endpoint):
            for ep in DEFAULT_VALKYRIE_ENDPOINTS:
                if ep not in endpoints_to_try:
                    endpoints_to_try.append(ep)

        last_err: Optional[Exception] = None
        for ep in endpoints_to_try:
            try:
                return self._call_http_endpoint(
                    ep,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(
            f"ValkyrieLLMClient failed to reach LLM endpoints {endpoints_to_try}: {last_err}"
        )

    def complete(
        self,
        prompt: str,
        system_prompt: str = "You are an intelligent Actor OOP Agent running inside the ActorII VM.",
        history: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1600,
        temperature: float = 0.1,
    ) -> str:
        """Execute a chat completion via custom provider, local HTTP endpoint, or fallback."""
        if self.custom_provider is not None:
            sig_meta = metadata or {}
            try:
                return str(self.custom_provider(prompt, system_prompt, sig_meta))
            except TypeError:
                try:
                    return str(self.custom_provider(prompt, system_prompt))
                except TypeError:
                    return str(self.custom_provider(prompt))

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for turn in history[-48:]:
                role = str(turn.get("role", "user"))
                content = str(turn.get("content", ""))
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        return self.complete_messages(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _call_http_endpoint(
        self,
        url: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1600,
        temperature: float = 0.1,
    ) -> str:
        t0 = time.perf_counter()
        is_local_engine = _is_local_llamacpp_or_vllm(url)
        sys_txt = (messages[0].get("content", "") if messages and messages[0].get("role") == "system" else "").lower()
        is_raw_synthesis = (
            "method synthesizer" in sys_txt
            or "output only the raw 1993 actor method" in sys_txt
            or "output only a valid 1993 actor" in sys_txt
        )
        if "/api/chat" in url:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
        else:
            last_user_msg = (messages[-1].get("content", "") if messages else "")
            last_user_lower = last_user_msg.lower()
            step_m = _STEP_NUM_RE.search(last_user_lower)
            step_num = int(step_m.group(1)) if step_m else (2 if len(messages) >= 4 else 1)
            last_res_tail = last_user_lower.split("last result:")[-1] if "last result:" in last_user_lower else last_user_lower[-1200:]
            already_verified_pass = (
                step_num >= 2
                and " passed" in last_res_tail
                and " failed" not in last_res_tail
                and "error" not in last_res_tail
                and "traceback" not in last_res_tail
            )
            needs_deep_reasoning = (
                step_num in (2, 3)
                or "[previous step failed" in last_user_lower
                or any(k in last_res_tail for k in ("failed", "error", "traceback", "timeoutexpired", "exitcode -> 1", "exitcode -> 2"))
            )
            env_thinking = (
                os.environ.get("ACTORII_ENABLE_THINKING") == "1"
                and step_num >= 2
                and needs_deep_reasoning
                and not already_verified_pass
            )
            is_thinking = "cascade" in str(self.model).lower() or env_thinking
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 600 if is_thinking else max_tokens,
                "temperature": 0.6 if is_thinking else temperature,
                "stream": False,
            }
            if is_raw_synthesis:
                payload["stop"] = ["\nExplanation:", "\nNote:"]
            if is_local_engine:
                payload["chat_template_kwargs"] = {"enable_thinking": True if is_thinking else False}
                if is_thinking:
                    payload["top_p"] = 0.95
                else:
                    payload["force_no_think"] = True
                    payload["reasoning_effort"] = "low"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        api_key = resolve_llm_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)

        usage = parsed.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        t_tok = int(usage.get("total_tokens") or (p_tok + c_tok))
        self._record_usage_and_stats(p_tok, c_tok, t_tok, calls=1)

        clean_fn = _strip_thinking_and_fences if is_raw_synthesis else _strip_thinking_only

        if "choices" in parsed and parsed["choices"]:
            choice0 = parsed["choices"][0]
            msg = choice0.get("message", {})
            finish_reason = choice0.get("finish_reason")
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            wants_json = "valid json" in sys_txt or "json object" in sys_txt
            if wants_json:
                needs_continuation = (not content) or (finish_reason == "length" and not content.rstrip().endswith("}"))
                cont_instruction = "Now emit ONLY the valid JSON object (no markdown fences, no prose):"
            else:
                needs_continuation = (not content) or (
                    finish_reason == "length" and not content.endswith("!!") and not content.upper().startswith("COMPLETE:")
                )
                cont_instruction = (
                    "Now emit ONLY the single raw 1993 Actor 4.1 expression ending in `!!` "
                    "or `COMPLETE: <your answer>` (no chain-of-thought):"
                )
            if needs_continuation and reasoning and "/api/chat" not in url:
                # Harmony / gpt-oss:20b or bounded Qwen3.8 thinking pass stopped after `reasoning_content`
                # before finishing `content`. Perform one non-thinking continuation pass to emit the output.
                followup_msgs = [dict(m) for m in messages]
                if followup_msgs and followup_msgs[-1].get("role") == "user":
                    followup_msgs[-1]["content"] = (
                        f"{followup_msgs[-1].get('content', '')}\n\n"
                        f"[Draft Analysis from Thinking Phase]:\n{reasoning}\n\n"
                        f"{cont_instruction}"
                    )
                else:
                    followup_msgs.append(
                        {
                            "role": "user",
                            "content": (
                                f"[Draft Analysis from Thinking Phase]:\n{reasoning}\n\n"
                                f"{cont_instruction}"
                            ),
                        }
                    )
                followup_payload: Dict[str, Any] = {
                    "model": self.model,
                    "messages": followup_msgs,
                    "max_tokens": 2048,
                    "temperature": 0.2,
                    "stream": False,
                }
                if is_local_engine:
                    followup_payload["chat_template_kwargs"] = {"enable_thinking": False}
                    followup_payload["force_no_think"] = True
                try:
                    req2 = urllib.request.Request(
                        url,
                        data=json.dumps(followup_payload).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req2, timeout=self.timeout) as resp2:
                        parsed2 = json.loads(resp2.read().decode("utf-8", errors="replace"))
                        u2 = parsed2.get("usage") or {}
                        p2 = int(u2.get("prompt_tokens") or 0)
                        c2 = int(u2.get("completion_tokens") or 0)
                        t2 = int(u2.get("total_tokens") or (p2 + c2))
                        p_tok += p2
                        c_tok += c2
                        t_tok += t2
                        self._record_usage_and_stats(p2, c2, t2, calls=1)
                        msg2 = parsed2.get("choices", [{}])[0].get("message", {})
                        content = (msg2.get("content") or "").strip()
                except Exception:
                    pass
                # Never leak raw `reasoning_content` chain-of-thought as user-facing content!

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            call_stats = {
                "endpoint": url,
                "model": self.model,
                "message_count": len(messages),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
                "latency_ms": round(elapsed_ms, 1),
            }
            self._record_usage_and_stats(0, 0, 0, calls=0, stats=call_stats)
            return clean_fn(str(content))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        call_stats = {
            "endpoint": url,
            "model": self.model,
            "message_count": len(messages),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "latency_ms": round(elapsed_ms, 1),
        }
        self._record_usage_and_stats(0, 0, 0, calls=0, stats=call_stats)
        if "message" in parsed and isinstance(parsed["message"], dict):
            return clean_fn(str(parsed["message"].get("content", "")))

        return clean_fn(raw)


# =============================================================================
# 2. Live Object Graph Introspection
# =============================================================================


def inspect_actor_object(
    obj: Any, hierarchy: Optional[ActorInheritanceHierarchy] = None
) -> Dict[str, Any]:
    """Return a structured dictionary describing any live Actor object or class."""
    if obj is None:
        return {"class": "Nil", "value": None}

    if isinstance(obj, ActorClass):
        cls_name = obj.name
        lineage = _get_class_lineage(cls_name, hierarchy)
        methods = {}
        for m_name, m_fn in list(getattr(obj, "_class_methods", {}).items()):
            if hasattr(m_fn, "params"):
                methods[m_name] = {
                    "params": list(getattr(m_fn, "params", [])),
                    "synthesized": bool(getattr(m_fn, "synthesized", False)),
                    "source": getattr(m_fn, "source", None),
                }
            else:
                methods[m_name] = {"type": "native"}
        return {
            "kind": "ActorClass",
            "class": cls_name,
            "lineage": lineage,
            "ivars": list(getattr(obj, "ivars", [])),
            "jit_synthesis": bool(getattr(obj, "jit_synthesis", False)),
            "methods": methods,
        }

    cls_name = getattr(obj, "class_name", obj.__class__.__name__)
    if callable(cls_name):
        cls_name = cls_name()
    cls_name = str(cls_name)
    lineage = _get_class_lineage(cls_name, hierarchy)

    fields_summary: Dict[str, Any] = {}
    if hasattr(obj, "_fields") and isinstance(obj._fields, dict):
        for k, v in list(obj._fields.items()):
            fields_summary[str(k)] = _summarize_field_value(v)

    cls_ivars: List[str] = []
    cls_ivars_set = set()
    cls_methods: List[str] = []
    cls_methods_set = set()
    if hierarchy is not None:
        curr = cls_name
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            c_obj = hierarchy.get(curr)
            if c_obj is not None:
                for iv in list(getattr(c_obj, "ivars", [])):
                    if iv not in cls_ivars_set:
                        cls_ivars_set.add(iv)
                        cls_ivars.append(iv)
                for mn in list(getattr(c_obj, "_class_methods", {}).keys()):
                    if mn not in cls_methods_set:
                        cls_methods_set.add(mn)
                        cls_methods.append(mn)
                curr = c_obj._bases[0] if c_obj._bases else ""
            else:
                break

    info: Dict[str, Any] = {
        "kind": "ActorObject",
        "class": cls_name,
        "obj_id": str(getattr(obj, "obj_id", id(obj))),
        "lineage": lineage,
        "ivars": cls_ivars,
        "fields": fields_summary,
        "methods": cls_methods,
    }

    if hasattr(obj, "title") and hasattr(obj, "rect"):
        info["title"] = getattr(obj, "title", "")
        r = getattr(obj, "rect", None)
        if r is not None and hasattr(r, "left"):
            info["rect"] = [r.left, r.top, r.right, r.bottom]
    if hasattr(obj, "children") and isinstance( getattr(obj, "children", None), list):
        info["children_count"] = len(obj.children)

    return info


def format_inspection_for_prompt(
    obj: Any, hierarchy: Optional[ActorInheritanceHierarchy] = None
) -> str:
    """Format `inspect_actor_object` into a concise human/LLM-readable summary string."""
    info = inspect_actor_object(obj, hierarchy=hierarchy)
    return json.dumps(info, indent=2, default=str)


def _get_class_lineage(
    class_name: str, hierarchy: Optional[ActorInheritanceHierarchy] = None
) -> List[str]:
    lineage = [class_name]
    if hierarchy is None:
        return lineage
    curr = class_name
    visited = {curr}
    while curr:
        cls = hierarchy.get(curr)
        if cls and cls._bases:
            parent = cls._bases[0]
            if parent not in visited:
                lineage.append(parent)
                visited.add(parent)
                curr = parent
            else:
                break
        else:
            break
    return lineage


def _summarize_field_value(val: Any) -> Any:
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Symbol):
        return f"#{val.name}"
    if hasattr(val, "_entries") and isinstance(val._entries, dict):
        return {str(k): _summarize_field_value(v) for k, v in list(val._entries.items())[:16]}
    if isinstance(val, (list, tuple)):
        return [_summarize_field_value(x) for x in list(val)[:16]]
    if hasattr(val, "class_name"):
        cn = val.class_name() if callable(val.class_name) else val.class_name
        return f"<{cn} {getattr(val, 'obj_id', id(val))}>"
    return repr(val)


# =============================================================================
# 3. 1993 Actor Method Source Extraction & Normalization
# =============================================================================


def extract_and_normalize_actor_method(
    raw_text: str, selector: str, param_names: List[str]
) -> str:
    """Clean and normalize LLM output into a valid 1993 Actor `Def ... !!` chunk."""
    cleaned = _strip_thinking_and_fences(raw_text).strip()

    # Locate `Def <selector>` or any `Def ` header
    def_match = _DEF_CHUNK_RE.search(cleaned)
    if def_match:
        cleaned = def_match.group(1).strip()

    # Replace `return expr;` with `^expr;`
    if "return" in cleaned:
        cleaned = _RETURN_STMT_RE.sub(r"^\1;", cleaned)
    # Strip C/JS-style `local x :=` or `var x :=` inside method body
    cleaned = _LOCAL_VAR_DECL_RE.sub(r"\1 :=", cleaned)

    # Ensure method name matches `selector`
    cleaned = _DEF_HEADER_RE.sub(
        f"Def {selector}(",
        cleaned,
        count=1,
    )

    # If the output didn't start with `Def `, wrap it as a method body
    if not cleaned.startswith("Def "):
        sig_params = ", ".join(["self"] + param_names)
        body = cleaned.rstrip("!").strip()
        if not body.startswith("{"):
            if not body.endswith(";"):
                body += ";"
            if "^" not in body:
                body = f"^{body}"
            body = "{ " + body + " }"
        cleaned = f"Def {selector}({sig_params}) {body}!!"

    if not cleaned.endswith("!!"):
        cleaned = cleaned.rstrip("!") + "!!"

    return cleaned


# =============================================================================
# 4. Semantic Heuristic Synthesis Fallback (Deterministic 1993 Actor Generator)
# =============================================================================


def synthesize_heuristic_actor_method(
    receiver: Any,
    class_name: str,
    selector: str,
    args: List[Any],
    hierarchy: Optional[ActorInheritanceHierarchy] = None,
    spec_prompt: str = "",
) -> str:
    """Synthesize a valid 1993 Actor `Def` method deterministically from live object schema & selector semantics.

    Used when no external LLM is configured or as a guaranteed-valid structural synthesizer.
    """
    arity = len(args)
    param_names = [f"arg{i + 1}" for i in range(arity)]
    sig_params = ", ".join(["self"] + param_names)

    info = inspect_actor_object(receiver, hierarchy=hierarchy)
    fields: Dict[str, Any] = info.get("fields", {})
    ivars: List[str] = info.get("ivars", [])
    all_field_names = list(dict.fromkeys(list(fields.keys()) + ivars))

    numeric_fields = [
        k for k in all_field_names if isinstance(fields.get(k), (int, float)) and not isinstance(fields.get(k), bool)
    ]
    string_fields = [
        k for k in all_field_names if isinstance(fields.get(k), str)
    ]
    if not numeric_fields and all_field_names:
        numeric_fields = [k for k in all_field_names if k not in string_fields]

    sel_lower = selector.lower()
    spec_lower = (spec_prompt or "").lower()

    # 1. Explicit Getter: get<Field> or exact field name match (arity == 0)
    if arity == 0:
        for fname in all_field_names:
            if selector == fname or sel_lower == f"get{fname.lower()}":
                return f"Def {selector}(self) {{\n    ^self.{fname};\n}}!!"

    # 2. Explicit Setter: set<Field>(self, val) (arity == 1)
    if arity == 1:
        for fname in all_field_names:
            if sel_lower == f"set{fname.lower()}":
                return (
                    f"Def {selector}(self, {param_names[0]}) {{\n"
                    f"    self.{fname} := {param_names[0]};\n"
                    f"    ^self;\n"
                    f"}}!!"
                )

    # 3. Tax / Discount / Invoice calculations (e.g. totalWithTax, applyTax, discountedTotal)
    if "tax" in sel_lower or "tax" in spec_lower:
        base_field = None
        tax_field = None
        for f in all_field_names:
            fl = f.lower()
            if fl in ("subtotal", "amount", "price", "total", "base", "cost"):
                base_field = f
            elif "tax" in fl or "rate" in fl:
                tax_field = f
        if base_field is None and numeric_fields:
            base_field = numeric_fields[0]
        if base_field and tax_field and arity == 0:
            return (
                f"Def {selector}(self) {{\n"
                f"    ^self.{base_field} + (self.{base_field} * self.{tax_field} / 100);\n"
                f"}}!!"
            )
        if base_field and arity == 1:
            return (
                f"Def {selector}(self, {param_names[0]}) {{\n"
                f"    ^self.{base_field} + (self.{base_field} * {param_names[0]} / 100);\n"
                f"}}!!"
            )

    if "discount" in sel_lower or "discount" in spec_lower:
        base_field = numeric_fields[0] if numeric_fields else (all_field_names[0] if all_field_names else None)
        if base_field and arity == 1:
            return (
                f"Def {selector}(self, {param_names[0]}) {{\n"
                f"    ^self.{base_field} - (self.{base_field} * {param_names[0]} / 100);\n"
                f"}}!!"
            )

    # 4. Area / Perimeter / Volume / Geometry calculations
    if sel_lower in ("area", "computearea", "getarea"):
        w_f = next((f for f in all_field_names if f.lower() in ("width", "w", "base", "radius", "side")), None)
        h_f = next((f for f in all_field_names if f.lower() in ("height", "h", "length")), None)
        if w_f and h_f:
            return f"Def {selector}(self) {{\n    ^self.{w_f} * self.{h_f};\n}}!!"
        if w_f:
            return f"Def {selector}(self) {{\n    ^self.{w_f} * self.{w_f};\n}}!!"
        if len(numeric_fields) >= 2:
            return f"Def {selector}(self) {{\n    ^self.{numeric_fields[0]} * self.{numeric_fields[1]};\n}}!!"

    if sel_lower in ("perimeter", "computeperimeter", "circumference"):
        w_f = next((f for f in all_field_names if f.lower() in ("width", "w", "side")), None)
        h_f = next((f for f in all_field_names if f.lower() in ("height", "h", "length")), None)
        if w_f and h_f:
            return f"Def {selector}(self) {{\n    ^(self.{w_f} + self.{h_f}) * 2;\n}}!!"
        if len(numeric_fields) >= 2:
            return f"Def {selector}(self) {{\n    ^(self.{numeric_fields[0]} + self.{numeric_fields[1]}) * 2;\n}}!!"

    # 5. Sum / Total / Product / Average across numeric fields or arguments
    if sel_lower in ("sum", "total", "computetotal", "sumfields"):
        if arity == 0 and numeric_fields:
            expr = " + ".join(f"self.{f}" for f in numeric_fields)
            return f"Def {selector}(self) {{\n    ^{expr};\n}}!!"
        if arity >= 1:
            terms = [f"self.{f}" for f in numeric_fields] + param_names
            expr = " + ".join(terms)
            return f"Def {selector}({sig_params}) {{\n    ^{expr};\n}}!!"

    if sel_lower in ("product", "multiply"):
        if arity == 0 and len(numeric_fields) >= 2:
            expr = " * ".join(f"self.{f}" for f in numeric_fields)
            return f"Def {selector}(self) {{\n    ^{expr};\n}}!!"
        if arity == 1 and numeric_fields:
            return f"Def {selector}(self, {param_names[0]}) {{\n    ^self.{numeric_fields[0]} * {param_names[0]};\n}}!!"
        if arity == 2:
            return f"Def {selector}({sig_params}) {{\n    ^{param_names[0]} * {param_names[1]};\n}}!!"

    if sel_lower in ("doubleplusone", "twiceplusone"):
        if arity == 1:
            return f"Def {selector}(self, {param_names[0]}) {{\n    ^({param_names[0]} * 2) + 1;\n}}!!"
        if numeric_fields:
            return f"Def {selector}(self) {{\n    ^(self.{numeric_fields[0]} * 2) + 1;\n}}!!"

    if sel_lower in ("double", "twice"):
        if arity == 1:
            return f"Def {selector}(self, {param_names[0]}) {{\n    ^{param_names[0]} * 2;\n}}!!"
        if numeric_fields:
            return f"Def {selector}(self) {{\n    ^self.{numeric_fields[0]} * 2;\n}}!!"

    if sel_lower in ("triple", "thrice"):
        if arity == 1:
            return f"Def {selector}(self, {param_names[0]}) {{\n    ^{param_names[0]} * 3;\n}}!!"
        if numeric_fields:
            return f"Def {selector}(self) {{\n    ^self.{numeric_fields[0]} * 3;\n}}!!"

    if sel_lower in ("square", "squared"):
        if arity == 1:
            return f"Def {selector}(self, {param_names[0]}) {{\n    ^{param_names[0]} * {param_names[0]};\n}}!!"
        if numeric_fields:
            f0 = numeric_fields[0]
            return f"Def {selector}(self) {{\n    ^self.{f0} * self.{f0};\n}}!!"

    if sel_lower in ("clamp", "clampvalue") and arity == 2 and numeric_fields:
        f0 = numeric_fields[0]
        return (
            f"Def {selector}(self, {param_names[0]}, {param_names[1]}) {{\n"
            f"    ^min(max(self.{f0}, {param_names[0]}), {param_names[1]});\n"
            f"}}!!"
        )
    if sel_lower in ("clamp", "clampvalue") and arity == 3:
        return (
            f"Def {selector}({sig_params}) {{\n"
            f"    ^min(max({param_names[0]}, {param_names[1]}), {param_names[2]});\n"
            f"}}!!"
        )

    # 6. Increment / Decrement / Add / Deposit / Withdraw on primary numeric field
    if ("increment" in sel_lower or "deposit" in sel_lower or "addto" in sel_lower) and numeric_fields:
        f0 = numeric_fields[0]
        for f in numeric_fields:
            if f.lower() in sel_lower or (f.lower() in ("balance", "count", "counter", "score", "value")):
                f0 = f
                break
        if arity == 0:
            return f"Def {selector}(self) {{\n    self.{f0} := self.{f0} + 1;\n    ^self.{f0};\n}}!!"
        if arity == 1:
            return (
                f"Def {selector}(self, {param_names[0]}) {{\n"
                f"    self.{f0} := self.{f0} + {param_names[0]};\n"
                f"    ^self.{f0};\n"
                f"}}!!"
            )

    if ("decrement" in sel_lower or "withdraw" in sel_lower or "subtract" in sel_lower) and numeric_fields:
        f0 = numeric_fields[0]
        for f in numeric_fields:
            if f.lower() in sel_lower or (f.lower() in ("balance", "count", "counter", "score", "value")):
                f0 = f
                break
        if arity == 0:
            return f"Def {selector}(self) {{\n    self.{f0} := self.{f0} - 1;\n    ^self.{f0};\n}}!!"
        if arity == 1:
            return (
                f"Def {selector}(self, {param_names[0]}) {{\n"
                f"    self.{f0} := self.{f0} - {param_names[0]};\n"
                f"    ^self.{f0};\n"
                f"}}!!"
            )

    # 7. String / Name / Summary / Greeting methods
    if sel_lower in ("fullname", "getfullname"):
        first_f = next((f for f in all_field_names if "first" in f.lower()), None)
        last_f = next((f for f in all_field_names if "last" in f.lower()), None)
        if first_f and last_f:
            return f'Def {selector}(self) {{\n    ^self.{first_f} + " " + self.{last_f};\n}}!!'
        if len(string_fields) >= 2:
            return f'Def {selector}(self) {{\n    ^self.{string_fields[0]} + " " + self.{string_fields[1]};\n}}!!'

    if sel_lower in ("greet", "greeting", "sayhello"):
        name_f = next((f for f in all_field_names if "name" in f.lower()), None)
        if arity == 1 and name_f:
            return f'Def {selector}(self, {param_names[0]}) {{\n    ^"Hello " + {param_names[0]} + " from " + self.{name_f};\n}}!!'
        if arity == 1:
            return f'Def {selector}(self, {param_names[0]}) {{\n    ^"Hello, " + {param_names[0]} + "!";\n}}!!'
        if name_f:
            return f'Def {selector}(self) {{\n    ^"Hello from " + self.{name_f};\n}}!!'

    if sel_lower in ("describe", "summary", "summarize", "tojson", "info"):
        if string_fields:
            expr = ' + " | " + '.join(f"self.{f}" for f in string_fields[:3])
            return f'Def {selector}(self) {{\n    ^"{class_name}: " + {expr};\n}}!!'
        return f'Def {selector}(self) {{\n    ^"<{class_name}>";\n}}!!'

    # 8. General fallback based on arguments & fields
    if arity == 0 and numeric_fields:
        expr = " + ".join(f"self.{f}" for f in numeric_fields)
        return f"Def {selector}(self) {{\n    ^{expr};\n}}!!"
    if arity == 1 and numeric_fields and isinstance(args[0], (int, float)):
        return f"Def {selector}(self, {param_names[0]}) {{\n    ^self.{numeric_fields[0]} + {param_names[0]};\n}}!!"
    if arity >= 2 and all(isinstance(a, (int, float)) for a in args):
        expr = " + ".join(param_names)
        return f"Def {selector}({sig_params}) {{\n    ^{expr};\n}}!!"
    if arity == 1:
        return f"Def {selector}(self, {param_names[0]}) {{\n    ^{param_names[0]};\n}}!!"

    return f"Def {selector}({sig_params}) {{\n    ^self;\n}}!!"


# =============================================================================
# 5. ActorMethodSynthesizer (`doesNotUnderstand` -> CompiledMethod Bytecode)
# =============================================================================


class ActorMethodSynthesizer:
    """Synthesizes, compiles, and attaches 1993 Actor bytecode methods on the fly."""

    def __init__(self, vm: Any) -> None:
        self.vm = vm

    def build_synthesis_prompt(
        self,
        receiver: Any,
        class_name: str,
        selector: str,
        args: List[Any],
        spec_prompt: str = "",
    ) -> str:
        """Construct the user prompt with full live object/class introspection."""
        info = inspect_actor_object(receiver, hierarchy=self.vm.hierarchy)
        param_names = [f"arg{i + 1}" for i in range(len(args))]
        sig = ", ".join(["self"] + param_names)
        arg_summary = [_summarize_field_value(a) for a in args]

        lines = [
            f"Target Class: {class_name}",
            f"Inheritance Lineage: {' -> '.join(info.get('lineage', [class_name]))}",
            f"Declared Instance Variables (ivars): {info.get('ivars', [])}",
            f"Live Receiver Instance Fields (access via `self.<field>`): {json.dumps(info.get('fields', {}), default=str)}",
            f"Existing Class Methods: {info.get('methods', [])}",
            f"Missing Message Selector: {selector}",
            f"Required Method Signature: Def {selector}({sig})",
            f"Runtime Arguments Passed: {json.dumps(arg_summary, default=str)}",
        ]
        if spec_prompt:
            lines.append(f"Additional Specification: {spec_prompt}")
        lines.append(
            f"\nWrite ONLY the 1993 Whitewater Actor 4.1 method `Def {selector}({sig}) {{ ... }}!!`."
        )
        return "\n".join(lines)

    def synthesize_method(
        self,
        receiver: Any,
        class_name: str,
        selector: str,
        args: List[Any],
        spec_prompt: str = "",
        use_llm: Optional[bool] = None,
    ) -> Any:
        """Synthesize a `CompiledMethod` for `class_name>>selector`, attach it to the class, and return it."""
        from actorii.parser import MethodDefNode, parse_actor

        param_names = [f"arg{i + 1}" for i in range(len(args))]
        prompt = self.build_synthesis_prompt(
            receiver=receiver,
            class_name=class_name,
            selector=selector,
            args=args,
            spec_prompt=spec_prompt,
        )

        candidate_source: Optional[str] = None
        metadata = {
            "mode": "method_synthesis",
            "class_name": class_name,
            "selector": selector,
            "arity": len(args),
            "args": args,
            "receiver": receiver,
            "spec_prompt": spec_prompt,
        }

        # 1. Try custom AI provider or live LLM if enabled
        ai_provider = getattr(self.vm, "ai_provider", None)
        should_call_llm = (
            use_llm
            if use_llm is not None
            else (ai_provider is not None or bool(getattr(self.vm, "use_live_llm_for_jit", False)))
        )

        compiled_method = None
        final_source = ""
        guardrail_rejection_reasons: List[str] = []

        if should_call_llm:
            client = ValkyrieLLMClient(
                endpoint=getattr(self.vm, "ai_endpoint", None),
                model=getattr(self.vm, "ai_model", "default"),
                custom_provider=ai_provider,
            )
            active_prompt = prompt
            for attempt in range(2):
                try:
                    meta_attempt = dict(metadata)
                    meta_attempt["attempt"] = attempt + 1
                    if guardrail_rejection_reasons:
                        meta_attempt["rejection_reasons"] = list(guardrail_rejection_reasons)
                    raw_resp = client.complete(
                        prompt=active_prompt,
                        system_prompt=ACTOR_SYNTHESIS_SYSTEM_PROMPT,
                        metadata=meta_attempt,
                        max_tokens=1024,
                        temperature=0.1,
                    )
                    if not raw_resp or not raw_resp.strip():
                        break
                    candidate_source = extract_and_normalize_actor_method(
                        raw_resp, selector=selector, param_names=param_names
                    )
                except Exception:
                    candidate_source = None
                    break

                if not candidate_source:
                    break

                cand_cm = self._try_compile_method_source(
                    candidate_source, class_name=class_name, selector=selector
                )
                if cand_cm is None:
                    guardrail_rejection_reasons = [
                        f"Syntax/compile error in candidate: {candidate_source}"
                    ]
                    active_prompt = (
                        prompt
                        + "\n\nPREVIOUS ATTEMPT FAILED TO COMPILE:\n"
                        + "\n".join(guardrail_rejection_reasons)
                        + "\nFix the 1993 Actor syntax and try again."
                    )
                    continue

                # Run static linter & judge check on candidate before accepting
                from actorii.guardrails.judge import ActorCodeJudge
                from actorii.guardrails.linter import ActorLinter, RiskTier

                linter = ActorLinter(
                    self.vm,
                    strict_ivars=bool(getattr(self.vm, "strict_ivars", True)),
                    protect_system_classes=bool(
                        getattr(self.vm, "protect_system_classes", False)
                    ),
                )
                report = linter.lint_method(
                    cand_cm, class_name=class_name, receiver=receiver
                )
                judge = ActorCodeJudge(
                    self.vm,
                    max_allowed_tier=RiskTier(
                        int(getattr(self.vm, "max_allowed_tier", 2))
                    ),
                    custom_judge=getattr(self.vm, "custom_judge", None),
                )
                verdict = judge.evaluate(cand_cm, report, is_synthesized=True)
                if verdict.approved:
                    compiled_method = cand_cm
                    final_source = candidate_source
                    guardrail_rejection_reasons = []
                    break

                guardrail_rejection_reasons = list(verdict.reasons)
                active_prompt = (
                    prompt
                    + "\n\nPREVIOUS ATTEMPT BLOCKED BY GUARDRAIL:\n"
                    + "\n".join(guardrail_rejection_reasons)
                    + "\nRewrite the method using ONLY declared instance variables and valid risk tier."
                )

        # If guardrails are explicitly enabled and LLM synthesis was rejected by guardrails, raise
        if (
            compiled_method is None
            and guardrail_rejection_reasons
            and bool(getattr(self.vm, "guardrails_enabled", False))
        ):
            from actorii.guardrails.linter import ActorGuardrailError

            raise ActorGuardrailError(
                f"Guardrail blocked synthesized method {class_name}>>{selector}: "
                f"{'; '.join(guardrail_rejection_reasons)}"
            )

        # 2. Fall back to deterministic heuristic synthesizer if needed
        if compiled_method is None:
            heuristic_source = synthesize_heuristic_actor_method(
                receiver=receiver,
                class_name=class_name,
                selector=selector,
                args=args,
                hierarchy=self.vm.hierarchy,
                spec_prompt=spec_prompt,
            )
            compiled_method = self._try_compile_method_source(
                heuristic_source, class_name=class_name, selector=selector
            )
            final_source = heuristic_source

        if compiled_method is None:
            raise RuntimeError(
                f"ActorMethodSynthesizer failed to compile synthesized method {class_name}>>{selector}"
            )

        compiled_method.source = final_source
        compiled_method.synthesized = True

        # 3. Permanently attach the CompiledMethod to the ActorClass in the hierarchy
        with _AI_LOCK:
            cls = self.vm.hierarchy.get(class_name)
            if cls is None and class_name:
                cls = ActorClass(class_name, bases=("Object",))
                self.vm.hierarchy.register(cls)
                self.vm.global_vars[class_name] = cls
                if "Actor" in self.vm.global_vars and isinstance(self.vm.global_vars["Actor"], dict):
                    self.vm.global_vars["Actor"][class_name] = cls

            if cls is not None:
                if hasattr(self.vm, "_validate_and_attach_method"):
                    self.vm._validate_and_attach_method(
                        cls, compiled_method, receiver=receiver, is_synthesized=True
                    )
                else:
                    cls.add_method(selector, compiled_method)
                if hasattr(self.vm, "_file_out_class_if_enabled"):
                    self.vm._file_out_class_if_enabled(class_name)

            if isinstance(receiver, ActorObject):
                receiver.set_method(selector, compiled_method)

            # Record synthesis event on the VM
            if not hasattr(self.vm, "synthesized_methods"):
                self.vm.synthesized_methods = []
            self.vm.synthesized_methods.append(
                {
                    "class_name": class_name,
                    "selector": selector,
                    "arity": len(compiled_method.params),
                    "source": final_source,
                    "risk_tier": int(getattr(compiled_method, "risk_tier", 0)),
                    "timestamp": time.time(),
                }
            )

        return compiled_method

    def _try_compile_method_source(
        self, source: str, class_name: str, selector: str
    ) -> Optional[Any]:
        from actorii.parser import MethodDefNode, parse_actor

        try:
            chunks = [c.strip() for c in source.split("!!") if c.strip()]
            for chunk in chunks:
                nodes = parse_actor(chunk)
                for node in nodes:
                    if isinstance(node, MethodDefNode):
                        node.name = selector
                        cm = self.vm.compile_method_node(node, class_name=class_name)
                        return cm
        except Exception:
            return None
        return None
