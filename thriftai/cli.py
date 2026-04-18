"""Minimal CLI entry point for `thriftai`."""

from __future__ import annotations

import argparse

from thriftai import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="thriftai",
        description=(
            "ThriftAI — multi-agent LLM cache, replay, and cost tracking. "
            "Use the Python API: `import thriftai as ta`. "
            "See https://github.com/rayabhik83/thriftai for docs."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"thriftai {__version__}",
    )
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
