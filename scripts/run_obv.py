from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from multifit_optveri.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
