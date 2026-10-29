"""
Krusch Agentic Proxy — Strict Pydantic v2 Tool Router & Validation Retry Loop (INV-12).

Provides:
  - Dynamic compilation of OpenAI / MCP JSON Schemas into strict Pydantic v2 BaseModels
    (`ConfigDict(extra="forbid")`, `StrictInt`, `StrictFloat`, `StrictStr`, `StrictBool`,
    `Literal` enums, numeric/length bounds, and nested objects/arrays).
  - Deterministic LLM repair trace generation from `ValidationError` and `json.JSONDecodeError`.
  - Bounded `execute_with_retry` loop (`max_retries=3` default) stripping `<think>` blocks and
    markdown fences, validating arguments strictly, and offloading synchronous handlers via
    `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import operator
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    create_model,
)

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)
_IDENT_CLEAN_RE = re.compile(r"[^a-zA-Z0-9_]")

__all__ = (
    "StrictPydanticToolRouter",
    "ToolExecutionResult",
    "format_validation_error_trace",
    "json_schema_to_pydantic_model",
)


class ToolExecutionResult(BaseModel):
    """Structured outcome returned by StrictPydanticToolRouter.execute_with_retry."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    tool_name: str
    validated_args: dict[str, Any] | None = None
    output: Any | None = None
    attempts: int = Field(ge=0)
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def _strip_thinking_and_fences(raw_text: str) -> str:
    """Remove `<think>...</think>` blocks and markdown code fences prior to JSON parsing."""
    if not raw_text:
        return ""
    cleaned = _THINK_RE.sub("", str(raw_text)).strip()
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        return fence_match.group(1).strip()
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines()[1:] if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()
    return cleaned


def _sanitize_model_name(name: str) -> str:
    cleaned = _IDENT_CLEAN_RE.sub("_", name or "ToolArgs").strip("_")
    if not cleaned:
        return "ToolArgs"
    if cleaned[0].isdigit():
        return f"Tool_{cleaned}"
    return cleaned


def _extract_field_constraints(prop_schema: dict[str, Any]) -> dict[str, Any]:
    """Map JSON Schema numeric, string, and array bounds onto Pydantic v2 Field kwargs."""
    kwargs: dict[str, Any] = {}
    if "description" in prop_schema and isinstance(prop_schema["description"], str):
        kwargs["description"] = prop_schema["description"]

    schema_type = prop_schema.get("type")
    if schema_type in ("integer", "number") or any(
        k in prop_schema for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
    ):
        if "minimum" in prop_schema:
            kwargs["ge"] = prop_schema["minimum"]
        if "maximum" in prop_schema:
            kwargs["le"] = prop_schema["maximum"]
        if "exclusiveMinimum" in prop_schema:
            kwargs["gt"] = prop_schema["exclusiveMinimum"]
        if "exclusiveMaximum" in prop_schema:
            kwargs["lt"] = prop_schema["exclusiveMaximum"]
        if "multipleOf" in prop_schema:
            kwargs["multiple_of"] = prop_schema["multipleOf"]

    if schema_type == "string" or any(
        k in prop_schema for k in ("minLength", "maxLength", "pattern")
    ):
        if "minLength" in prop_schema:
            kwargs["min_length"] = int(prop_schema["minLength"])
        if "maxLength" in prop_schema:
            kwargs["max_length"] = int(prop_schema["maxLength"])
        if "pattern" in prop_schema:
            kwargs["pattern"] = str(prop_schema["pattern"])

    if schema_type == "array" or any(k in prop_schema for k in ("minItems", "maxItems")):
        if "minItems" in prop_schema:
            kwargs["min_length"] = int(prop_schema["minItems"])
        if "maxItems" in prop_schema:
            kwargs["max_length"] = int(prop_schema["maxItems"])

    return kwargs


def _compile_schema_type(model_prefix: str, prop_schema: dict[str, Any]) -> Any:
    """Recursively compile a JSON Schema property node into a strict Python/Pydantic type."""
    if not isinstance(prop_schema, dict):
        return Any

    if "enum" in prop_schema and isinstance(prop_schema["enum"], list) and prop_schema["enum"]:
        enum_vals = tuple(prop_schema["enum"])
        non_null_vals = tuple(v for v in enum_vals if v is not None)
        has_null = any(v is None for v in enum_vals)
        if non_null_vals:
            lit_type = Literal[non_null_vals]  # type: ignore[valid-type]
            return (lit_type | None) if has_null else lit_type
        return Any | None

    for union_key in ("anyOf", "oneOf"):
        variants = prop_schema.get(union_key)
        if isinstance(variants, list) and variants:
            compiled_variants: list[Any] = []
            has_null = False
            for idx, sub in enumerate(variants):
                if isinstance(sub, dict) and sub.get("type") == "null":
                    has_null = True
                else:
                    compiled_variants.append(
                        _compile_annotated_type(f"{model_prefix}_V{idx}", sub)
                    )
            if not compiled_variants:
                return Any | None
            union_t = functools.reduce(operator.or_, compiled_variants)
            return (union_t | None) if has_null else union_t

    raw_type = prop_schema.get("type")
    nullable = bool(prop_schema.get("nullable", False))
    if isinstance(raw_type, list):
        non_null_types = [t for t in raw_type if t != "null"]
        if "null" in raw_type:
            nullable = True
        raw_type = non_null_types[0] if len(non_null_types) == 1 else non_null_types

    if isinstance(raw_type, list):
        sub_types = [
            _compile_schema_type(f"{model_prefix}_{t}", {**prop_schema, "type": t})
            for t in raw_type
        ]
        base_t = functools.reduce(operator.or_, sub_types) if sub_types else Any
        return (base_t | None) if nullable else base_t

    if raw_type == "string":
        base: Any = StrictStr
    elif raw_type == "integer":
        base = StrictInt
    elif raw_type == "number":
        base = StrictFloat
    elif raw_type == "boolean":
        base = StrictBool
    elif raw_type == "null":
        return None
    elif raw_type == "array":
        items_schema = prop_schema.get("items")
        if isinstance(items_schema, dict):
            item_t = _compile_annotated_type(f"{model_prefix}_Item", items_schema)
        else:
            item_t = Any
        base = list[item_t]  # type: ignore[valid-type]
    elif raw_type == "object" or "properties" in prop_schema:
        if "properties" in prop_schema or prop_schema.get("additionalProperties") is False:
            base = json_schema_to_pydantic_model(model_prefix, prop_schema)
        elif isinstance(prop_schema.get("additionalProperties"), dict):
            val_t = _compile_annotated_type(
                f"{model_prefix}_Val", prop_schema["additionalProperties"]
            )
            base = dict[StrictStr, val_t]  # type: ignore[valid-type]
        else:
            base = dict[str, Any]
    else:
        base = Any

    return (base | None) if nullable else base


def _compile_annotated_type(model_prefix: str, prop_schema: dict[str, Any]) -> Any:
    """Compile a nested item/union schema including inline Field constraints via Annotated."""
    base_t = _compile_schema_type(model_prefix, prop_schema)
    constraints = _extract_field_constraints(prop_schema)
    if constraints:
        return Annotated[base_t, Field(**constraints)]
    return base_t


def json_schema_to_pydantic_model(
    name: str,
    parameters_schema: dict[str, Any],
) -> type[BaseModel]:
    """
    Dynamically compile an OpenAI or MCP tool JSON Schema (`properties`, `required`, `enum`,
    `minimum`, `maximum`, `minLength`, `maxLength`, nested `object`/`array`) into a strict
    Pydantic v2 `BaseModel` with `ConfigDict(extra="forbid")`.
    """
    if not isinstance(parameters_schema, dict):
        raise TypeError("parameters_schema must be a dict")

    # Unwrap full OpenAI tool dict or MCP inputSchema wrapper if passed directly
    schema = parameters_schema
    if "function" in schema and isinstance(schema["function"], dict):
        schema = schema["function"].get("parameters", schema["function"])
    elif "parameters" in schema and "properties" not in schema and isinstance(schema["parameters"], dict):
        schema = schema["parameters"]
    elif "inputSchema" in schema and "properties" not in schema and isinstance(schema["inputSchema"], dict):
        schema = schema["inputSchema"]

    model_name = _sanitize_model_name(name)
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        raise TypeError(f"Invalid 'properties' in schema for '{name}'")

    required_fields = set(schema.get("required") or [])
    field_definitions: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            prop_schema = {}

        sub_prefix = f"{model_name}_{_sanitize_model_name(prop_name)}"
        field_type = _compile_schema_type(sub_prefix, prop_schema)
        constraints = _extract_field_constraints(prop_schema)
        is_required = prop_name in required_fields

        if is_required:
            field_definitions[prop_name] = (field_type, Field(..., **constraints))
        else:
            default_val = prop_schema.get("default", None)
            annotated_type = (field_type | None) if default_val is None else field_type
            field_definitions[prop_name] = (
                annotated_type,
                Field(default=default_val, **constraints),
            )

    extra_policy: Literal["forbid", "allow"] = (
        "allow" if schema.get("additionalProperties") is True else "forbid"
    )
    return create_model(
        model_name,
        __config__=ConfigDict(extra=extra_policy),
        **field_definitions,
    )


def _json_safe(value: Any) -> Any:
    """Coerce arbitrary values from ValidationError.errors() into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return repr(value)


def format_validation_error_trace(
    exc: Exception,
    raw_payload: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Format `pydantic.ValidationError` or `json.JSONDecodeError` (or routing errors) into a
    concise, deterministic repair trace string and structured error dictionary list.
    """
    preview = (raw_payload or "").strip()
    if len(preview) > 400:
        preview = preview[:397] + "..."

    if isinstance(exc, ValidationError):
        raw_errs = exc.errors(include_url=False)
        structured: list[dict[str, Any]] = []
        bullet_lines: list[str] = []
        for err in raw_errs:
            loc_list = [str(p) for p in err.get("loc", ())]
            path_str = ".".join(loc_list) if loc_list else "<root>"
            err_type = str(err.get("type", "validation_error"))
            msg = str(err.get("msg", "Invalid value"))
            inp = _json_safe(err.get("input"))
            structured.append(
                {
                    "loc": loc_list,
                    "path": path_str,
                    "type": err_type,
                    "msg": msg,
                    "input": inp,
                }
            )
            bullet_lines.append(f"- field `{path_str}` ({err_type}): {msg} [received={inp!r}]")

        trace = (
            f"TOOL_VALIDATION_ERROR ({len(structured)} issue(s)):\n"
            + "\n".join(bullet_lines)
            + f"\nRaw payload: {preview}\n"
            + "Repair instruction: Respond ONLY with valid JSON matching the exact tool schema "
            "(all required fields present, strict types, no extra keys)."
        )
        return trace, structured

    if isinstance(exc, json.JSONDecodeError):
        structured = [
            {
                "loc": ["__json__"],
                "path": "__json__",
                "type": "json_invalid",
                "msg": exc.msg,
                "lineno": exc.lineno,
                "colno": exc.colno,
                "pos": exc.pos,
            }
        ]
        trace = (
            f"TOOL_JSON_DECODE_ERROR at line {exc.lineno}, column {exc.colno} (char {exc.pos}): "
            f"{exc.msg}.\n"
            f"Raw payload: {preview}\n"
            "Repair instruction: Respond ONLY with a valid JSON object (no prose or trailing commas)."
        )
        return trace, structured

    structured = [
        {
            "loc": ["__tool__"],
            "path": "__tool__",
            "type": type(exc).__name__,
            "msg": str(exc),
        }
    ]
    trace = (
        f"TOOL_ROUTING_ERROR ({type(exc).__name__}): {exc}\n"
        f"Raw payload: {preview}\n"
        "Repair instruction: Respond ONLY with valid JSON targeting a registered tool."
    )
    return trace, structured


@dataclass
class _RegisteredTool:
    name: str
    model: type[BaseModel]
    handler: Callable[..., Any]
    description: str


def _is_async_handler(fn: Callable[..., Any]) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    call_attr = getattr(fn, "__call__", None)  # noqa: B004
    return bool(call_attr and inspect.iscoroutinefunction(call_attr))


class StrictPydanticToolRouter:
    """
    Strict Pydantic v2 tool registry, validator, and self-healing retry executor.
    """

    def __init__(self, max_retries: int = 3) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = int(max_retries)
        self._tools: dict[str, _RegisteredTool] = {}

    def register_tool(
        self,
        name: str,
        schema: type[BaseModel] | dict[str, Any],
        handler: Callable[..., Any],
        description: str = "",
    ) -> None:
        """Register a tool handler backed by either a Pydantic v2 BaseModel or JSON Schema dict."""
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("Tool name must be a non-empty string")
        if not callable(handler):
            raise TypeError(f"Handler for tool '{clean_name}' must be callable")

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            if schema.model_config.get("extra") == "forbid":
                model_cls: type[BaseModel] = schema
            else:
                merged_cfg = ConfigDict(**{**dict(schema.model_config), "extra": "forbid"})
                model_cls = create_model(
                    f"{schema.__name__}Strict",
                    __base__=schema,
                    __config__=merged_cfg,
                )
        elif isinstance(schema, dict):
            model_cls = json_schema_to_pydantic_model(clean_name, schema)
        else:
            raise TypeError(
                f"schema for '{clean_name}' must be a BaseModel subclass or JSON Schema dict"
            )

        self._tools[clean_name] = _RegisteredTool(
            name=clean_name,
            model=model_cls,
            handler=handler,
            description=description,
        )

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Export registered tools as OpenAI-compatible function schemas."""
        exported: list[dict[str, Any]] = []
        for entry in self._tools.values():
            exported.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry.name,
                        "description": entry.description,
                        "parameters": entry.model.model_json_schema(),
                    },
                }
            )
        return exported

    def validate_call(
        self,
        tool_name: str,
        raw_arguments: str | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate `raw_arguments` (JSON string or dict) against `tool_name`'s strict BaseModel.
        Returns normalized `model_dump()` dict or raises KeyError / JSONDecodeError / ValidationError.
        """
        if tool_name not in self._tools:
            raise KeyError(
                f"Unknown tool '{tool_name}'. Available tools: {sorted(self._tools.keys())}"
            )

        entry = self._tools[tool_name]
        if isinstance(raw_arguments, str):
            cleaned = _strip_thinking_and_fences(raw_arguments)
            parsed = json.loads(cleaned)
        elif isinstance(raw_arguments, dict):
            parsed = raw_arguments
        else:
            parsed = raw_arguments  # Let Pydantic raise canonical model_type ValidationError

        validated = entry.model.model_validate(parsed)
        return validated.model_dump()

    def _extract_tool_and_args(
        self,
        parsed_json: Any,
        expected_tool: str | None,
    ) -> tuple[str, Any]:
        """
        Normalize parsed JSON payload into `(tool_name, raw_arguments)`.
        Supports:
          - `{"name": "<tool>", "arguments": {...}}`
          - `{"type": "function", "function": {"name": "<tool>", "arguments": {...}}}`
          - `{"tool_calls": [{"name": "<tool>", "arguments": {...}}]}`
          - Direct argument dictionary `{...}` when `expected_tool` is specified.
        """
        if isinstance(parsed_json, list):
            if not parsed_json:
                raise ValueError("Tool call array is empty")
            parsed_json = parsed_json[0]

        if not isinstance(parsed_json, dict):
            raise TypeError(f"Expected JSON object for tool call, got {type(parsed_json).__name__}")

        if isinstance(parsed_json.get("tool_calls"), list) and parsed_json["tool_calls"]:
            first_call = parsed_json["tool_calls"][0]
            if not isinstance(first_call, dict):
                raise TypeError("Invalid entry in 'tool_calls'")
            parsed_json = first_call

        if isinstance(parsed_json.get("function"), dict):
            fn_obj = parsed_json["function"]
            t_name = str(fn_obj.get("name") or "").strip()
            raw_args = fn_obj.get("arguments", fn_obj.get("parameters", {}))
            if not t_name:
                raise ValueError("Missing tool 'name' inside 'function' block")
            if expected_tool and t_name != expected_tool:
                raise ValueError(f"Expected tool '{expected_tool}', but model called '{t_name}'")
            return t_name, raw_args

        has_wrapper_args = "arguments" in parsed_json or "parameters" in parsed_json
        candidate_name = parsed_json.get("name")

        if has_wrapper_args and isinstance(candidate_name, str) and candidate_name.strip():
            t_name = candidate_name.strip()
            if expected_tool and t_name != expected_tool:
                raise ValueError(f"Expected tool '{expected_tool}', but model called '{t_name}'")
            raw_args = parsed_json.get("arguments", parsed_json.get("parameters", {}))
            return t_name, raw_args

        if expected_tool:
            # Handle zero-arg wrapper `{"name": expected_tool}` when `name` is not a schema field
            tool_entry = self._tools.get(expected_tool)
            model_fields = set(tool_entry.model.model_fields.keys()) if tool_entry else set()
            if (
                set(parsed_json.keys()) == {"name"}
                and candidate_name == expected_tool
                and "name" not in model_fields
            ):
                return expected_tool, {}
            return expected_tool, parsed_json

        if isinstance(candidate_name, str) and candidate_name.strip():
            t_name = candidate_name.strip()
            extra_keys = set(parsed_json.keys()) - {"name"}
            if not extra_keys:
                return t_name, {}
            # Model emitted flat `{"name": "<tool>", ...inline_args}`
            if t_name in self._tools:
                inline_args = {k: v for k, v in parsed_json.items() if k != "name"}
                return t_name, inline_args
            return t_name, {}

        if len(self._tools) == 1:
            sole_tool = next(iter(self._tools.keys()))
            return sole_tool, parsed_json

        raise ValueError(
            f"Missing 'name' and 'arguments' in tool call payload; registered tools: "
            f"{sorted(self._tools.keys())}"
        )

    async def _invoke_handler(
        self,
        entry: _RegisteredTool,
        validated_args: dict[str, Any],
    ) -> Any:
        """Invoke synchronous or asynchronous tool handler safely without blocking the event loop."""
        handler = entry.handler
        call_args: tuple[Any, ...] = ()
        call_kwargs: dict[str, Any] = dict(validated_args)

        try:
            sig = inspect.signature(handler)
            params = list(sig.parameters.values())
            pos_params = [
                p
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

            if len(pos_params) == 1 and not has_var_kw:
                p0 = pos_params[0]
                if p0.name not in validated_args:
                    ann = p0.annotation
                    if isinstance(ann, type) and issubclass(ann, BaseModel):
                        call_args = (entry.model.model_validate(validated_args),)
                        call_kwargs = {}
                    elif ann in (dict, dict[str, Any], "dict", "dict[str, Any]", "Dict[str, Any]"):
                        call_args = (validated_args,)
                        call_kwargs = {}
        except (TypeError, ValueError):
            pass

        if _is_async_handler(handler):
            return await handler(*call_args, **call_kwargs)

        res = await asyncio.to_thread(handler, *call_args, **call_kwargs)
        if inspect.isawaitable(res):
            return await res
        return res

    async def execute_with_retry(
        self,
        *,
        prompt: str,
        llm_generate_fn: Callable[[str], Awaitable[str]],
        expected_tool: str | None = None,
    ) -> ToolExecutionResult:
        """
        Generate, parse, strictly validate, and execute a tool call.
        On JSONDecodeError, ValidationError, or unknown-tool routing failure, appends a
        deterministic repair trace to the prompt and retries up to `self.max_retries`
        (`total attempts <= 1 + self.max_retries`).
        """
        if expected_tool and expected_tool not in self._tools:
            return ToolExecutionResult(
                ok=False,
                tool_name=expected_tool,
                validated_args=None,
                output=None,
                attempts=0,
                validation_errors=[
                    {
                        "loc": ["__tool__"],
                        "path": "__tool__",
                        "type": "unknown_tool",
                        "msg": f"Unknown expected_tool '{expected_tool}'",
                    }
                ],
                error=f"Unknown expected_tool '{expected_tool}'",
            )

        max_attempts = 1 + self.max_retries
        current_prompt = prompt
        last_tool_name = expected_tool or ""
        last_validation_errors: list[dict[str, Any]] = []
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            raw_response_maybe = llm_generate_fn(current_prompt)
            raw_response = (
                await raw_response_maybe
                if inspect.isawaitable(raw_response_maybe)
                else str(raw_response_maybe)
            )
            cleaned = _strip_thinking_and_fences(raw_response)

            try:
                parsed = json.loads(cleaned)
                resolved_tool, candidate_args = self._extract_tool_and_args(parsed, expected_tool)
                last_tool_name = resolved_tool
                validated_args = self.validate_call(resolved_tool, candidate_args)
            except (json.JSONDecodeError, ValidationError, KeyError, ValueError, TypeError) as exc:
                trace_str, structured_errs = format_validation_error_trace(exc, raw_response)
                last_error = trace_str
                last_validation_errors = structured_errs
                logger.debug(
                    "Tool validation failed on attempt %d/%d for '%s': %s",
                    attempt,
                    max_attempts,
                    last_tool_name or "<unresolved>",
                    exc,
                )
                if attempt < max_attempts:
                    current_prompt = (
                        f"{current_prompt}\n\n"
                        f"[Validation Retry {attempt}/{self.max_retries}]\n"
                        f"{trace_str}"
                    )
                    continue
                return ToolExecutionResult(
                    ok=False,
                    tool_name=last_tool_name or (expected_tool or "unknown"),
                    validated_args=None,
                    output=None,
                    attempts=attempt,
                    validation_errors=last_validation_errors,
                    error=last_error,
                )

            # Arguments strictly validated — execute registered handler
            entry = self._tools[resolved_tool]
            try:
                output = await self._invoke_handler(entry, validated_args)
            except Exception as handler_exc:  # noqa: BLE001 — isolate runtime tool handler exceptions
                return ToolExecutionResult(
                    ok=False,
                    tool_name=resolved_tool,
                    validated_args=validated_args,
                    output=None,
                    attempts=attempt,
                    validation_errors=[],
                    error=f"Tool handler {type(handler_exc).__name__}: {handler_exc}",
                )

            return ToolExecutionResult(
                ok=True,
                tool_name=resolved_tool,
                validated_args=validated_args,
                output=output,
                attempts=attempt,
                validation_errors=[],
                error=None,
            )

        return ToolExecutionResult(  # pragma: no cover - defensive fallback
            ok=False,
            tool_name=last_tool_name or "unknown",
            validated_args=None,
            output=None,
            attempts=max_attempts,
            validation_errors=last_validation_errors,
            error=last_error or "Exhausted retries",
        )
