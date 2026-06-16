"""CLI entry point for stencil_gen.

Usage:
    python -m stencil_gen list
    python -m stencil_gen generate <scheme>
    python -m stencil_gen validate
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m stencil_gen {list|generate|validate}")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        print("Available schemes:")
        print("  Interior: E2_1, E4_1, E6_1, E8_1, T4_1, T6_1, T8_1")
        print("  Interior 2nd: E2_2, E4_2")
        print("  Uniform boundary: E4u_1, E6u_1, E8u_1")
        print("  Cut-cell (TEMO): polyE2_1, E2_1")
    elif command == "generate":
        if len(sys.argv) < 3:
            print("Usage: python -m stencil_gen generate <scheme>")
            sys.exit(1)
        scheme = sys.argv[2]
        if scheme == "polyE2_1":
            import os

            from stencil_gen.codegen import generate_stencil_cpp
            from stencil_gen.interp import build_polyE2_1_spec

            spec = build_polyE2_1_spec()
            code = generate_stencil_cpp(spec)
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "polyE2_1.cpp")
            with open(out_path, "w") as f:
                f.write(code)
            print(f"Wrote {out_path}")
        else:
            print(f"TODO: generate {scheme}")
    elif command == "validate":
        print("TODO: validate against existing C++")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
