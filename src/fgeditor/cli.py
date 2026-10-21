"""Command line helper for the FlexGravity editor core (development/testing)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import ARRAY_NAMES, Project, ProjectError


def cmd_inspect(args: argparse.Namespace) -> int:
    project = Project.load(args.input)
    print(f"file: {args.input}")
    print(f"objects: {len(project.nmo.objects)}")
    for name in ("FG_U", "FG_M", "FG_L", "FG_S", "FG_R"):
        members = project.group_indices(name)
        print(f"group {name}: {len(members)} " + ", ".join(project.object_name(i) for i in members))
    for name in ARRAY_NAMES:
        obj = project.nmo.dataarray_by_name(name)
        print(f"array {name}: {'present' if obj else 'missing'}")
    for name, table in project.fields.items():
        print(f"\n{name} ({', '.join(table.params)}):")
        for row in table.rows:
            print(f"  {project.object_display(row.object_index) or '<empty>':30s} {row.values}")
    print("\nFG_R:")
    for row in project.r_rows:
        print(f"  {project.object_display(row.region_index) or '<empty>':30s} -> "
              f"{[project.object_display(i) for i in row.sources]}")
    print(f"\nFG_RG: {[project.object_display(i) for i in project.rg_sources]}")
    print("\nFG_Custom:")
    for row in project.custom_rows:
        mode = "cartesian" if row.mode == 0 else "spherical"
        print(f"  {row.name!r:30s} {mode:9s} {row.values}")
    print(f"\nwarnings ({len(project.warnings)}):")
    for message in project.warnings:
        print("  WARN", message)
    print(f"errors ({len(project.errors)}):")
    for message in project.errors:
        print("  ERR ", message)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    project = Project.load(args.input)
    try:
        project.export(args.output, update_level=args.level)
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}")
    for message in project.warnings:
        print("WARN", message)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fgeditor")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="print groups and arrays")
    inspect.add_argument("input", type=Path)
    inspect.set_defaults(func=cmd_inspect)

    export = sub.add_parser("export", help="export an NMO")
    export.add_argument("input", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--level", action="store_true", help="update the Level object list")
    export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
