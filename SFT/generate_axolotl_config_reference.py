"""Generate a readable reference from the pinned Axolotl configuration schema.

This intentionally avoids importing PyTorch, Transformers, and Accelerate. The
schema modules only need a few runtime capability symbols from those packages,
so lightweight stubs are sufficient for inspecting field declarations.
"""

from __future__ import annotations

import inspect
import logging
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import get_args

import yaml
from pydantic import BaseModel


SFT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SFT_DIR.parent
AXOLOTL_DIR = SFT_DIR / "axolotl"
SOURCE_ROOT = AXOLOTL_DIR / "src"
OUTPUT_PATH = SFT_DIR / "AXOLOTL_CONFIG_REFERENCE.yaml"
PINNED_AXOLOTL_COMMIT = "2f5cb9da62a0fe763a1ddeb7798fc9acb2f4a417"


def install_import_stubs() -> None:
    """Provide only the external symbols needed to declare the config schema."""

    torch = types.ModuleType("torch")
    torch.int4 = "int4"
    torch.int8 = "int8"
    torch.float8_e4m3fn = "float8_e4m3fn"
    torch.__version__ = "2.12.0"
    sys.modules["torch"] = torch

    accelerate = types.ModuleType("accelerate")
    accelerate_utils = types.ModuleType("accelerate.utils")
    accelerate_utils.is_fp8_available = lambda: False
    accelerate.utils = accelerate_utils
    sys.modules["accelerate"] = accelerate
    sys.modules["accelerate.utils"] = accelerate_utils

    transformers = types.ModuleType("transformers")
    transformers_utils = types.ModuleType("transformers.utils")
    transformers_import_utils = types.ModuleType("transformers.utils.import_utils")
    transformers_import_utils.is_torch_npu_available = lambda: False
    transformers_utils.import_utils = transformers_import_utils
    transformers.utils = transformers_utils
    sys.modules["transformers"] = transformers
    sys.modules["transformers.utils"] = transformers_utils
    sys.modules["transformers.utils.import_utils"] = transformers_import_utils

    axolotl_logging = types.ModuleType("axolotl.utils.logging")
    axolotl_logging.get_logger = logging.getLogger
    sys.modules["axolotl.utils.logging"] = axolotl_logging


def unwrap_optional(annotation):
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return args[0] if len(args) == 1 else annotation


def readable_annotation(annotation) -> str:
    text = str(annotation)
    return (
        text.removeprefix("<class '")
        .removesuffix("'>")
        .replace("typing.", "")
    )


def defining_class(model_class, field_name):
    for cls in model_class.__mro__:
        if field_name in getattr(cls, "__annotations__", {}):
            return cls
    return model_class


def source_location(model_class, field_name: str) -> str | None:
    cls = defining_class(model_class, field_name)
    source_file = inspect.getsourcefile(cls)
    if not source_file:
        return None

    path = Path(source_file).resolve()
    try:
        display_path = path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        display_path = path.as_posix()

    try:
        lines, start_line = inspect.getsourcelines(cls)
        pattern = re.compile(rf"^\s*{re.escape(field_name)}\s*:")
        for offset, line in enumerate(lines):
            if pattern.match(line):
                return f"{display_path}:{start_line + offset}"
    except (OSError, TypeError):
        pass

    return display_path


def field_record(
    model_class,
    field_name: str,
    yaml_name: str,
    field_info,
    property_schema: dict,
    required_names: set[str],
) -> dict:
    reserved = {"title", "description", "default"}
    record = {
        "type": readable_annotation(field_info.annotation),
        "required": yaml_name in required_names,
        "description": property_schema.get("description"),
    }

    if "default" in property_schema:
        record["default"] = property_schema["default"]
    elif yaml_name in required_names:
        record["default"] = "NO DEFAULT — REQUIRED"
    elif field_info.default_factory is not None:
        record["default"] = "COMPUTED BY DEFAULT FACTORY"
    else:
        record["default"] = None

    if field_info.alias and field_info.alias != field_name:
        record["python_field_name"] = field_name

    record["source"] = source_location(model_class, field_name)
    accepted = {
        key: value
        for key, value in property_schema.items()
        if key not in reserved
    }
    if accepted:
        record["accepted_schema"] = accepted
    return record


def get_axolotl_commit() -> str:
    """Return the vendored Axolotl commit, with or without nested Git metadata."""
    if not (AXOLOTL_DIR / ".git").exists():
        return PINNED_AXOLOTL_COMMIT

    result = subprocess.run(
        ["git", "-C", str(AXOLOTL_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    install_import_stubs()
    sys.path.insert(0, str(SOURCE_ROOT))

    from axolotl.utils.schemas.config import AxolotlInputConfig

    main_schema = AxolotlInputConfig.model_json_schema()
    main_required = set(main_schema.get("required", []))
    top_level_fields = {}
    nested_sections = {}
    missing_descriptions = []

    for name, field_info in AxolotlInputConfig.model_fields.items():
        yaml_name = field_info.alias or name
        property_schema = main_schema["properties"][yaml_name]
        top_level_fields[yaml_name] = field_record(
            AxolotlInputConfig,
            name,
            yaml_name,
            field_info,
            property_schema,
            main_required,
        )
        if not property_schema.get("description"):
            missing_descriptions.append(yaml_name)

        nested_model = unwrap_optional(field_info.annotation)
        try:
            is_nested_model = isinstance(nested_model, type) and issubclass(
                nested_model, BaseModel
            )
        except TypeError:
            is_nested_model = False
        if not is_nested_model:
            continue

        nested_schema = nested_model.model_json_schema()
        nested_required = set(nested_schema.get("required", []))
        nested_fields = {}
        for nested_name, nested_info in nested_model.model_fields.items():
            nested_yaml_name = nested_info.alias or nested_name
            nested_property = nested_schema["properties"][nested_yaml_name]
            dotted_name = f"{yaml_name}.{nested_yaml_name}"
            nested_fields[nested_yaml_name] = field_record(
                nested_model,
                nested_name,
                nested_yaml_name,
                nested_info,
                nested_property,
                nested_required,
            )
            if not nested_property.get("description"):
                missing_descriptions.append(dotted_name)
        nested_sections[yaml_name] = nested_fields

    reference = {
        "metadata": {
            "axolotl_version": (AXOLOTL_DIR / "VERSION")
            .read_text(encoding="utf-8")
            .strip(),
            "axolotl_commit": get_axolotl_commit(),
            "source_schema": (
                "SFT/axolotl/src/axolotl/utils/schemas/"
                "config.py::AxolotlInputConfig"
            ),
            "top_level_field_count": len(top_level_fields),
            "nested_section_count": len(nested_sections),
            "nested_field_count": sum(
                len(fields) for fields in nested_sections.values()
            ),
            "fields_without_schema_descriptions": len(missing_descriptions),
            "notes": [
                "This is a reference, not a runnable Axolotl configuration.",
                (
                    "A null description means Axolotl v0.18.0 supplies no "
                    "schema explanation; use the listed source location."
                ),
                (
                    "accepted_schema is the exact Pydantic JSON-schema "
                    "fragment for the field."
                ),
                (
                    "Plugin-provided fields are not included because no "
                    "plugins were loaded."
                ),
            ],
        },
        "fields_without_schema_descriptions": missing_descriptions,
        "top_level_fields": top_level_fields,
        "nested_sections": nested_sections,
        "schema_definitions": main_schema.get("$defs", {}),
    }

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output:
        output.write(
            "# Generated from the pinned Axolotl v0.18.0 configuration schema.\n"
        )
        output.write("# Do not use this file as a training configuration.\n")
        yaml.safe_dump(
            reference,
            output,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
