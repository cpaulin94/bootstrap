"""Entry point for the Bootstrap Portfolio Analyser GUI."""

from __future__ import annotations


def main() -> None:
    from gui import BootstrapApp

    BootstrapApp().mainloop()


if __name__ == "__main__":
    main()
