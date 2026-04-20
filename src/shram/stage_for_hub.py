"""Produce a Hub-compatible flat staging directory from a model source directory.

HuggingFace's dynamic_module_utils resolves relative imports by appending ".py"
to the dotted import path without converting dots to path separators. This means
any file in a subdirectory is never found at load time. The staging system works
around this by flattening the directory tree and rewriting imports to match.

Subdirectory files are renamed using a double-underscore prefix convention:

    cache/shram_cache.py      ->  __cache__shram_cache.py
    attention/mosrah.py       ->  __attention__mosrah.py

All imports — relative and absolute — that refer to files within the model
directory are rewritten to use the flat staged names:

    from .cache.shram_cache import X      ->  from .__cache__shram_cache import X
    from .attention.mosrah import X       ->  from .__attention__mosrah import X
    from src.shram.model.attention.router import X  ->  from .__attention__router import X

__init__.py files inside subdirectories are dropped. No meaningful logic should
live in subdirectory __init__.py files — they will not be present on the Hub.
"""

import shutil
from pathlib import Path
from typing import Optional

import libcst as cst


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def compute_flat_name(rel: Path) -> Optional[str]:
    """Return the flat staged filename stem for a path relative to the source root.

    This is the single authority for staged file naming. Both the file-saving
    loop and the import rewriter call this function, so they always agree.

    Args:
        rel: Path relative to the source model directory, e.g. Path("cache/shram_cache.py").

    Returns:
        Flat filename stem (no extension), or None if the file should be dropped.

    Raises:
        ValueError: If the path is nested more than one level deep.
    """
    parts = rel.parts

    if len(parts) == 1:
        return rel.stem

    if len(parts) == 2:
        subfolder, filename = parts
        if filename == "__init__.py":
            return None
        return f"__{subfolder}__{Path(filename).stem}"

    raise ValueError(
        f"Unexpected path depth in model directory: {rel}. "
        f"Only one level of subdirectory is supported by the staging system."
    )


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------

def _resolve_to_rel(
    source_dir: Path,
    importing_dir: Path,
    level: int,
    dotted_module: str,
    abs_package: str,
) -> Optional[Path]:
    """Resolve an import to a path relative to source_dir.

    For relative imports (level > 0): navigates (level - 1) directories up from
    importing_dir, then appends the dotted module parts as a path, and looks for
    a matching .py file within source_dir.

    For absolute imports (level == 0): if dotted_module starts with abs_package,
    strips that prefix and resolves the remainder from source_dir root.

    Args:
        source_dir: Absolute path to the model source directory.
        importing_dir: Absolute path of the directory containing the importing file.
        level: Number of leading dots in the import (0 for absolute).
        dotted_module: Dotted module string, e.g. "cache.shram_cache" or
            "src.shram.model.attention.router".
        abs_package: The absolute import prefix corresponding to source_dir root,
            e.g. "src.shram.model".

    Returns:
        Path relative to source_dir, or None if the import does not resolve to
        a file within source_dir (i.e. it is an external import).
    """
    if level == 0:
        # Absolute import — only handle if it starts with the known package prefix.
        prefix = abs_package + "."
        if not dotted_module.startswith(prefix):
            return None
        remainder = dotted_module[len(prefix):]
        parts = remainder.split(".")
        candidate = source_dir.joinpath(*parts).with_suffix(".py")
        if candidate.is_file():
            return candidate.relative_to(source_dir)
        return None

    # Relative import — navigate up (level - 1) directories from importing_dir.
    base = importing_dir
    for _ in range(level - 1):
        base = base.parent

    parts = dotted_module.split(".") if dotted_module else []
    candidate = base.joinpath(*parts).with_suffix(".py") if parts else None

    if candidate is not None and candidate.is_file():
        try:
            return candidate.relative_to(source_dir)
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Import rewriting
# ---------------------------------------------------------------------------

def _extract_dotted_parts(node: cst.BaseExpression) -> Optional[list[str]]:
    """Recursively extract dotted name parts from a Name or Attribute node.

    Args:
        node: A libcst expression node.

    Returns:
        List of name segments in left-to-right order, or None if the node is
        not a pure dotted name expression.
    """
    if isinstance(node, cst.Name):
        return [node.value]
    if isinstance(node, cst.Attribute):
        left = _extract_dotted_parts(node.value)
        if left is None:
            return None
        if not isinstance(node.attr, cst.Name):
            return None
        return left + [node.attr.value]
    return None


class _ImportRewriter(cst.CSTTransformer):
    """Rewrite local imports to use flat staged module names.

    Handles both relative imports (leading dots) and absolute imports that
    start with the known package prefix. For each matched import, resolves
    the module to a source file path and rewrites the module name using
    compute_flat_name.
    """

    def __init__(
        self,
        source_dir: Path,
        importing_dir: Path,
        abs_package: str,
    ) -> None:
        super().__init__()
        self._source_dir = source_dir
        self._importing_dir = importing_dir
        self._abs_package = abs_package

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        """Rewrite the module path of local imports to use flat staged names.

        Args:
            original_node: The original ImportFrom node before transformation.
            updated_node: The updated ImportFrom node to potentially modify.

        Returns:
            Modified ImportFrom node, or the unchanged node if no rewrite is needed.
        """
        level = len(updated_node.relative)
        module = updated_node.module

        if level == 0 and module is None:
            return updated_node

        # Extract the dotted module string.
        if module is None:
            dotted_module = ""
        else:
            parts = _extract_dotted_parts(module)
            if parts is None:
                return updated_node
            dotted_module = ".".join(parts)

        rel = _resolve_to_rel(
            self._source_dir,
            self._importing_dir,
            level,
            dotted_module,
            self._abs_package,
        )
        if rel is None:
            return updated_node

        flat_name = compute_flat_name(rel)
        if flat_name is None:
            # Resolves to a dropped file (subdirectory __init__.py). This should
            # not occur after the manual __init__.py fixes; raise to surface it.
            raise ValueError(
                f"Import resolves to a dropped file: {rel}. "
                f"Fix the import in the source before staging."
            )

        # Rewrite as a relative import (single dot) with the flat module name.
        return updated_node.with_changes(
            relative=(cst.Dot(),),
            module=cst.Name(flat_name),
        )


def _rewrite_imports(
    source: str,
    source_dir: Path,
    file_rel: Path,
    abs_package: str = "src.shram.model",
) -> str:
    """Parse Python source, rewrite all local imports to flat staged names.

    Args:
        source: Python source code as a string.
        source_dir: Absolute path to the model source directory.
        file_rel: Path of the file being processed, relative to source_dir.
        abs_package: Absolute import prefix corresponding to source_dir root.

    Returns:
        Modified Python source with all local imports rewritten.
    """
    importing_dir = (source_dir / file_rel).parent
    tree = cst.parse_module(source)
    rewritten = tree.visit(_ImportRewriter(source_dir, importing_dir, abs_package))
    return rewritten.code


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def stage(
    source_dir: Path,
    dest_dir: Path,
    abs_package: str = "src.shram.model",
) -> None:
    """Flatten source_dir into dest_dir for Hub-compatible upload.

    Copies all files from source_dir to dest_dir, renaming subdirectory files
    using the double-underscore prefix convention and rewriting all local imports
    in Python files to match. dest_dir must already exist and should be empty.

    Files excluded from staging:
    - __pycache__ directories and .pyc files
    - __init__.py files inside subdirectories

    Args:
        source_dir: Source model directory (e.g. src/shram/model/).
        dest_dir: Destination staging directory to write flattened files into.
        abs_package: Absolute import prefix that maps to source_dir root.
            Defaults to "src.shram.model".
    """
    for src_path in sorted(source_dir.rglob("*")):
        if not src_path.is_file():
            continue

        rel = src_path.relative_to(source_dir)

        if any(part == "__pycache__" for part in rel.parts):
            continue
        if src_path.suffix == ".pyc":
            continue

        flat_name = compute_flat_name(rel)
        if flat_name is None:
            continue

        dst_path = dest_dir / (flat_name + src_path.suffix)

        if src_path.suffix == ".py":
            source = src_path.read_text(encoding="utf-8")
            rewritten = _rewrite_imports(source, source_dir, rel, abs_package)
            dst_path.write_text(rewritten, encoding="utf-8")
        else:
            shutil.copy2(src_path, dst_path)
