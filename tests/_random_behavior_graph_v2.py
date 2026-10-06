"""Identity-driven bounded behavior graph for RNG-independent entry audits."""

from __future__ import annotations

import dis
import inspect
import re
from collections.abc import Callable
from functools import partial
from types import (
    BuiltinFunctionType,
    ClassMethodDescriptorType,
    CodeType,
    FunctionType,
    GenericAlias,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    MethodDescriptorType,
    MethodType,
    MethodWrapperType,
    ModuleType,
    UnionType,
    WrapperDescriptorType,
)
from typing import NamedTuple
from weakref import ReferenceType

import numpy as np

_MISSING = object()
_CAPSULE_OPERATION_NAMES = (
    "create_stream",
    "seed_digest",
    "randbelow",
    "seed_digest_for_replicate",
)
_KNOWN_NATIVE_CALLABLE_TYPES = (
    BuiltinFunctionType,
    MethodDescriptorType,
    WrapperDescriptorType,
    ClassMethodDescriptorType,
)
_DEFAULT_NUMPY_RANDOM_MODULES = (np.random,)
_DEFAULT_NUMPY_ARRAY_TYPE = np.ndarray
_DEFAULT_SAFE_NUMPY_CALLABLE_TERMINALS = (
    np.all,
    np.asarray,
    np.broadcast_to,
    np.concatenate,
    np.diff,
    np.empty_like,
    np.floor,
    np.isfinite,
    np.linspace,
    np.logical_and,
    np.max,
    np.maximum,
    np.min,
    np.minimum,
    np.moveaxis,
    np.ndim,
    np.not_equal,
    np.subtract,
)
_KNOWN_CALLABLE_METADATA_TYPES = (type(Callable[..., object]),)
_KNOWN_SAFE_PATTERN_TYPE = type(re.compile(""))


class BehaviorCaptureGraph(NamedTuple):
    """All scheduled paths plus opaque behavior carriers in a bounded graph."""

    occurrences: tuple[tuple[str, object], ...]
    opaque_callables: tuple[tuple[str, object], ...]
    opaque_carriers: tuple[tuple[str, object], ...]


def _static_attribute(candidate: object, name: str) -> object:
    if type(candidate) is ModuleType:
        return vars(candidate).get(name, _MISSING)
    if isinstance(candidate, tuple):
        fields = getattr(type(candidate), "_fields", ())
        if type(fields) is tuple and name in fields:
            return tuple.__getitem__(candidate, fields.index(name))
    try:
        return inspect.getattr_static(candidate, name)
    except (AttributeError, TypeError):
        return _MISSING


def _referenced_attribute_chains(
    function: FunctionType,
    closure: inspect.ClosureVars,
) -> tuple[tuple[str, object], ...]:
    """Resolve only direct LOAD_* -> LOAD_ATTR/METHOD provider chains."""

    bindings = {**closure.nonlocals, **closure.globals}
    instructions = tuple(dis.get_instructions(function))
    resolved: list[tuple[str, object]] = []
    load_opnames = {"LOAD_GLOBAL", "LOAD_DEREF", "LOAD_CLASSDEREF"}
    attribute_opnames = {"LOAD_ATTR", "LOAD_METHOD"}
    for index, instruction in enumerate(instructions):
        if instruction.opname not in load_opnames:
            continue
        root_name = instruction.argval
        if type(root_name) is not str or root_name not in bindings:
            continue
        candidate = bindings[root_name]
        chain = root_name
        for follower in instructions[index + 1 :]:
            if follower.opname not in attribute_opnames:
                break
            attribute_name = follower.argval
            if type(attribute_name) is not str:
                break
            candidate = _static_attribute(candidate, attribute_name)
            if candidate is _MISSING:
                break
            chain = f"{chain}.{attribute_name}"
            resolved.append((chain, candidate))
    return tuple(resolved)


def _referenced_bound_attribute_chains(
    function: FunctionType,
    bindings: dict[str, object],
) -> tuple[tuple[str, object], ...]:
    """Resolve direct LOAD_FAST -> LOAD_ATTR/METHOD chains from known objects."""

    instructions = tuple(dis.get_instructions(function))
    resolved: list[tuple[str, object]] = []
    for index, instruction in enumerate(instructions):
        if instruction.opname != "LOAD_FAST":
            continue
        root_name = instruction.argval
        if type(root_name) is not str or root_name not in bindings:
            continue
        candidate = bindings[root_name]
        chain = root_name
        for follower in instructions[index + 1 :]:
            if follower.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                break
            attribute_name = follower.argval
            if type(attribute_name) is not str:
                break
            candidate = _static_attribute(candidate, attribute_name)
            if candidate is _MISSING:
                break
            chain = f"{chain}.{attribute_name}"
            resolved.append((chain, candidate))
    return tuple(resolved)


def _function_default_bindings(function: FunctionType) -> dict[str, object]:
    positional_count = function.__code__.co_argcount
    positional_names = function.__code__.co_varnames[:positional_count]
    defaults = function.__defaults__ or ()
    bindings = (
        dict(zip(positional_names[-len(defaults) :], defaults, strict=True))
        if defaults
        else {}
    )
    bindings.update(function.__kwdefaults__ or {})
    return bindings


def _referenced_exact_getattrs(
    function: FunctionType,
    closure: inspect.ClosureVars,
) -> tuple[tuple[str, object], ...]:
    """Resolve exact ``getattr(bound_object, constant_name)`` call sites."""

    exact_getattr = closure.builtins.get("getattr", _MISSING)
    if exact_getattr is not getattr:
        return ()
    bindings = {
        **closure.nonlocals,
        **closure.globals,
        **_function_default_bindings(function),
    }
    instructions = tuple(dis.get_instructions(function))
    resolved: list[tuple[str, object]] = []
    for index, instruction in enumerate(instructions):
        if instruction.opname not in {"LOAD_GLOBAL", "LOAD_DEREF"}:
            continue
        if instruction.argval != "getattr":
            continue
        cursor = index + 1
        if cursor + 2 >= len(instructions):
            continue
        root_load = instructions[cursor]
        name_load = instructions[cursor + 1]
        cursor += 2
        if instructions[cursor].opname == "PRECALL":
            if instructions[cursor].argval != 2:
                continue
            cursor += 1
        if cursor >= len(instructions):
            continue
        call = instructions[cursor]
        if root_load.opname not in {
            "LOAD_GLOBAL",
            "LOAD_DEREF",
            "LOAD_CLASSDEREF",
            "LOAD_FAST",
        } or name_load.opname != "LOAD_CONST" or call.opname != "CALL":
            continue
        if call.argval != 2:
            continue
        root_name = root_load.argval
        attribute_name = name_load.argval
        if (
            type(root_name) is not str
            or root_name not in bindings
            or type(attribute_name) is not str
        ):
            continue
        value = _static_attribute(bindings[root_name], attribute_name)
        if value is not _MISSING:
            resolved.append((f"getattr({root_name}.{attribute_name})", value))
    return tuple(resolved)


def _slot_values(candidate: object) -> tuple[tuple[str, object], ...]:
    values: list[tuple[str, object]] = []
    for owner in type(candidate).__mro__:
        declared = vars(owner).get("__slots__", ())
        slot_names = (declared,) if type(declared) is str else declared
        if type(slot_names) not in {tuple, list}:
            continue
        for name in slot_names:
            if type(name) is not str or name in {"__dict__", "__weakref__"}:
                continue
            descriptor = vars(owner).get(name, _MISSING)
            if type(descriptor) is not MemberDescriptorType:
                continue
            try:
                value = descriptor.__get__(candidate, type(candidate))
            except AttributeError:
                continue
            values.append((name, value))
    return tuple(values)


def _instance_dict(candidate: object) -> dict[str, object] | None:
    try:
        state = object.__getattribute__(candidate, "__dict__")
    except (AttributeError, TypeError):
        return None
    return state if type(state) is dict else None


_ATOMIC_VALUE_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
    range,
    slice,
    object,
    GenericAlias,
    UnionType,
    MemberDescriptorType,
    GetSetDescriptorType,
    *_KNOWN_CALLABLE_METADATA_TYPES,
    _KNOWN_SAFE_PATTERN_TYPE,
)


def behavior_capture_graph(
    root: object,
    *,
    follow_function_globals: bool = True,
    terminal_identities: tuple[object, ...] = (),
    numpy_array_type: type[object] | None = None,
    node_limit: int = 2048,
    edge_limit: int = 8192,
    depth_limit: int = 24,
) -> BehaviorCaptureGraph:
    """Walk behavior-reachable Python values without sweeping module namespaces."""

    pending: list[tuple[str, object, int]] = [("root", root, 0)]
    expanded: set[int] = set()
    occurrences: list[tuple[str, object]] = []
    opaque_callables: list[tuple[str, object]] = []
    opaque_carriers: list[tuple[str, object]] = []
    edge_count = 0

    def schedule(path: str, value: object, depth: int) -> None:
        nonlocal edge_count
        edge_count += 1
        if edge_count > edge_limit:
            raise AssertionError(f"behavior graph exceeded edge bound at {path}")
        pending.append((path, value, depth))

    while pending:
        path, candidate, depth = pending.pop()
        if depth > depth_limit:
            raise AssertionError(f"behavior graph exceeded depth bound at {path}")
        occurrences.append((path, candidate))
        if id(candidate) in expanded:
            continue
        expanded.add(id(candidate))
        if len(expanded) > node_limit:
            raise AssertionError(f"behavior graph exceeded node bound at {path}")
        if any(candidate is terminal for terminal in terminal_identities):
            continue

        if type(candidate) is FunctionType:
            closure = inspect.getclosurevars(candidate)
            schedule(f"{path}.code", candidate.__code__, depth + 1)
            for name, value in closure.nonlocals.items():
                schedule(f"{path}.nonlocal[{name}]", value, depth + 1)
            if follow_function_globals:
                for name, value in closure.globals.items():
                    schedule(f"{path}.global[{name}]", value, depth + 1)
                for chain, value in _referenced_attribute_chains(candidate, closure):
                    schedule(f"{path}.provider[{chain}]", value, depth + 1)
                for chain, value in _referenced_exact_getattrs(candidate, closure):
                    schedule(f"{path}.provider[{chain}]", value, depth + 1)
            for name, value in closure.builtins.items():
                schedule(f"{path}.builtin[{name}]", value, depth + 1)
            for index, value in enumerate(candidate.__defaults__ or ()):
                schedule(f"{path}.default[{index}]", value, depth + 1)
            for name, value in (candidate.__kwdefaults__ or {}).items():
                schedule(f"{path}.kwdefault[{name}]", value, depth + 1)
            for chain, value in _referenced_bound_attribute_chains(
                candidate,
                _function_default_bindings(candidate),
            ):
                schedule(f"{path}.argument_provider[{chain}]", value, depth + 1)
            continue

        if type(candidate) is CodeType:
            for index, constant in enumerate(candidate.co_consts):
                if type(constant) is CodeType:
                    schedule(f"{path}.nested_code[{index}]", constant, depth + 1)
            continue

        if type(candidate) is partial:
            schedule(f"{path}.partial.func", candidate.func, depth + 1)
            schedule(f"{path}.partial.args", candidate.args, depth + 1)
            if candidate.keywords is not None:
                schedule(f"{path}.partial.keywords", candidate.keywords, depth + 1)
            state = _instance_dict(candidate)
            if state is not None:
                for name, value in state.items():
                    schedule(f"{path}.state[{name}]", value, depth + 1)
            continue

        if type(candidate) is MethodType:
            schedule(f"{path}.method.func", candidate.__func__, depth + 1)
            schedule(f"{path}.method.self", candidate.__self__, depth + 1)
            method_self = candidate.__self__
            state = _instance_dict(method_self)
            if state is not None:
                for name, value in state.items():
                    schedule(
                        f"{path}.method.self.state[{name}]",
                        value,
                        depth + 2,
                    )
            for name, value in _slot_values(method_self):
                schedule(
                    f"{path}.method.self.slot[{name}]",
                    value,
                    depth + 2,
                )
            continue

        if type(candidate) is BuiltinFunctionType:
            bound_self = candidate.__self__
            if bound_self is not None and type(bound_self) is not ModuleType:
                schedule(f"{path}.builtin.self", bound_self, depth + 1)
                method_name = candidate.__name__
                provider = _static_attribute(type(bound_self), method_name)
                if provider is not _MISSING:
                    schedule(f"{path}.builtin.provider[{method_name}]", provider, depth + 1)
            continue

        if type(candidate) is MethodWrapperType:
            bound_self = candidate.__self__
            schedule(f"{path}.method_wrapper.self", bound_self, depth + 1)
            method_name = candidate.__name__
            provider = _static_attribute(type(bound_self), method_name)
            if provider is not _MISSING:
                schedule(
                    f"{path}.method_wrapper.provider[{method_name}]",
                    provider,
                    depth + 1,
                )
            continue

        if type(candidate) is ReferenceType:
            referent = candidate()
            if referent is not None:
                schedule(f"{path}.weakref.referent", referent, depth + 1)
            continue

        if type(candidate) in {
            MethodDescriptorType,
            WrapperDescriptorType,
            ClassMethodDescriptorType,
        }:
            continue

        if type(candidate) is property:
            for name, value in (
                ("fget", candidate.fget),
                ("fset", candidate.fset),
                ("fdel", candidate.fdel),
            ):
                if value is not None:
                    schedule(f"{path}.property.{name}", value, depth + 1)
            continue

        if type(candidate) in {staticmethod, classmethod}:
            schedule(f"{path}.descriptor.func", candidate.__func__, depth + 1)
            continue

        if type(candidate) is tuple or (
            isinstance(candidate, tuple)
            and type(getattr(type(candidate), "_fields", ())) is tuple
        ):
            fields = getattr(type(candidate), "_fields", ())
            for index in range(len(candidate)):
                label = fields[index] if index < len(fields) else index
                schedule(
                    f"{path}.tuple[{label}]",
                    tuple.__getitem__(candidate, index),
                    depth + 1,
                )
            continue

        if type(candidate) is list:
            for index, value in enumerate(candidate):
                schedule(f"{path}.list[{index}]", value, depth + 1)
            continue

        if type(candidate) in {dict, MappingProxyType}:
            for index, (key, value) in enumerate(candidate.items()):
                schedule(f"{path}.dict.key[{index}]", key, depth + 1)
                schedule(f"{path}.dict.value[{index}]", value, depth + 1)
            continue

        if type(candidate) in {set, frozenset}:
            for index, value in enumerate(candidate):
                schedule(f"{path}.{type(candidate).__name__}[{index}]", value, depth + 1)
            continue

        if numpy_array_type is not None and type(candidate) is numpy_array_type:
            try:
                contains_objects = bool(candidate.dtype.hasobject)  # type: ignore[attr-defined]
            except (AttributeError, TypeError, ValueError):
                opaque_carriers.append((path, candidate))
                continue
            if contains_objects:
                try:
                    values = tuple(candidate.flat)  # type: ignore[attr-defined]
                except (AttributeError, TypeError, ValueError):
                    opaque_carriers.append((path, candidate))
                    continue
                for index, value in enumerate(values):
                    schedule(f"{path}.object_array[{index}]", value, depth + 1)
            continue

        if type(candidate) in _ATOMIC_VALUE_TYPES:
            continue

        if type(candidate) is ModuleType:
            continue

        if isinstance(candidate, type):
            if type(candidate) is not type:
                call_descriptor = _static_attribute(type(candidate), "__call__")
                if call_descriptor is _MISSING:
                    opaque_callables.append((path, candidate))
                else:
                    schedule(f"{path}.metaclass.__call__", call_descriptor, depth + 1)
                    call_function = call_descriptor
                    if type(call_descriptor) in {staticmethod, classmethod}:
                        call_function = call_descriptor.__func__
                    if (
                        type(call_function) is FunctionType
                        and call_function.__code__.co_argcount
                    ):
                        cls_name = call_function.__code__.co_varnames[0]
                        for chain, value in _referenced_bound_attribute_chains(
                            call_function,
                            {cls_name: candidate},
                        ):
                            schedule(
                                f"{path}.class_provider[{chain}]",
                                value,
                                depth + 1,
                            )
            continue

        if callable(candidate):
            call_descriptor = _static_attribute(type(candidate), "__call__")
            inspectable_call = type(call_descriptor) in {
                FunctionType,
                MethodType,
                staticmethod,
                classmethod,
                partial,
            }
            if call_descriptor is not _MISSING:
                schedule(f"{path}.__call__", call_descriptor, depth + 1)

            call_function = call_descriptor
            if type(call_descriptor) in {staticmethod, classmethod}:
                call_function = call_descriptor.__func__
            if type(call_function) is FunctionType and call_function.__code__.co_argcount:
                self_name = call_function.__code__.co_varnames[0]
                for chain, value in _referenced_bound_attribute_chains(
                    call_function,
                    {self_name: candidate},
                ):
                    schedule(f"{path}.instance_provider[{chain}]", value, depth + 1)

            state = _instance_dict(candidate)
            if state is not None:
                for name, value in state.items():
                    schedule(f"{path}.state[{name}]", value, depth + 1)
            for name, value in _slot_values(candidate):
                schedule(f"{path}.slot[{name}]", value, depth + 1)

            wrapped = state.get("__wrapped__", _MISSING) if state is not None else _MISSING
            inspectable_wrapper = type(wrapped) in {FunctionType, MethodType}
            if (
                not inspectable_call
                and not inspectable_wrapper
                and type(candidate) not in _KNOWN_NATIVE_CALLABLE_TYPES
            ):
                opaque_callables.append((path, candidate))
            continue

        state = _instance_dict(candidate)
        slots = _slot_values(candidate)
        if state is not None:
            for name, value in state.items():
                schedule(f"{path}.state[{name}]", value, depth + 1)
            for name, value in slots:
                schedule(f"{path}.slot[{name}]", value, depth + 1)
            continue
        if slots:
            for name, value in slots:
                schedule(f"{path}.slot[{name}]", value, depth + 1)
            continue
        opaque_carriers.append((path, candidate))

    return BehaviorCaptureGraph(
        occurrences=tuple(occurrences),
        opaque_callables=tuple(opaque_callables),
        opaque_carriers=tuple(opaque_carriers),
    )


def random_behavior_reachability_findings(
    root: Callable[..., object],
    *,
    capsule_type: type[tuple[object, ...]],
    canonical_capsule: tuple[object, ...],
    public_random_source_type: type[object],
    pcg64_type: type[object],
    pcg64_raw_descriptor: object,
    numpy_random_modules: tuple[ModuleType, ...] = _DEFAULT_NUMPY_RANDOM_MODULES,
    numpy_array_type: type[object] | None = _DEFAULT_NUMPY_ARRAY_TYPE,
    safe_terminal_identities: tuple[object, ...] = _DEFAULT_SAFE_NUMPY_CALLABLE_TERMINALS,
    node_limit: int = 2048,
    edge_limit: int = 8192,
    depth_limit: int = 24,
) -> tuple[str, ...]:
    """Report random providers reachable from one entrypoint by exact identity."""

    tuple_getitem = tuple.__getitem__
    canonical_operations = tuple(
        tuple_getitem(canonical_capsule, index) for index in range(4)
    )
    canonical_codes = {
        operation.__code__: name
        for name, operation in zip(
            _CAPSULE_OPERATION_NAMES,
            canonical_operations,
            strict=True,
        )
    }
    public_descriptors = tuple(
        inspect.getattr_static(public_random_source_type, name)
        for name in (
            "__init__",
            "_next_raw64",
            "randbelow",
            "seed_digest_sha256",
        )
    )
    forbidden_identities = (
        capsule_type,
        canonical_capsule,
        public_random_source_type,
        pcg64_type,
        pcg64_raw_descriptor,
        *canonical_operations,
        *public_descriptors,
    )
    graph = behavior_capture_graph(
        root,
        terminal_identities=safe_terminal_identities,
        numpy_array_type=numpy_array_type,
        node_limit=node_limit,
        edge_limit=edge_limit,
        depth_limit=depth_limit,
    )
    findings: list[str] = []
    random_module_names = tuple(
        name
        for module in numpy_random_modules
        for name in (vars(module).get("__name__"),)
        if type(name) is str
    )
    dangerous_dynamic_resolvers = (__import__, eval, exec, compile)
    for path, candidate in graph.occurrences:
        if any(candidate is forbidden for forbidden in forbidden_identities):
            findings.append(f"{path}:forbidden-identity:{type(candidate).__name__}")
        if type(candidate) is capsule_type:
            findings.append(f"{path}:random-capsule-instance")
        if type(candidate) is public_random_source_type:
            findings.append(f"{path}:public-random-source-instance")
        if type(candidate) is pcg64_type:
            findings.append(f"{path}:pcg64-instance")
        if type(candidate) is FunctionType and candidate.__code__ in canonical_codes:
            operation_name = canonical_codes[candidate.__code__]
            findings.append(f"{path}:random-capsule-operation:{operation_name}")
            findings.append(
                f"{path}:random-capsule-operation-code:{operation_name}"
            )
        if type(candidate) is CodeType and candidate in canonical_codes:
            findings.append(
                f"{path}:random-capsule-operation-code:{canonical_codes[candidate]}"
            )
        if any(candidate is module for module in numpy_random_modules) or any(
            candidate is exported
            for module in numpy_random_modules
            for exported in vars(module).values()
        ):
            findings.append(f"{path}:numpy-random-provider:exact-module-export")
        if any(candidate is resolver for resolver in dangerous_dynamic_resolvers):
            findings.append(f"{path}:dynamic-resolver:{candidate.__name__}")
        if type(candidate) is CodeType:
            for instruction in dis.get_instructions(candidate):
                if instruction.opname != "IMPORT_NAME":
                    continue
                import_name = instruction.argval
                if type(import_name) is str and any(
                    import_name == module_name
                    or import_name.startswith(f"{module_name}.")
                    for module_name in random_module_names
                ):
                    findings.append(
                        f"{path}:numpy-random-provider:import-instruction"
                    )
    for path, candidate in graph.opaque_callables:
        findings.append(
            f"{path}:opaque-callable:{type(candidate).__module__}."
            f"{type(candidate).__qualname__}"
        )
    for path, candidate in graph.opaque_carriers:
        findings.append(
            f"{path}:opaque-carrier:{type(candidate).__module__}."
            f"{type(candidate).__qualname__}"
        )
    return tuple(sorted(findings))
