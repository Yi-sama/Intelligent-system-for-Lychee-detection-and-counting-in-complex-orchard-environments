from __future__ import annotations

import sys
from pathlib import Path

from app.gui import create_app


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "Reference" / "yolo26n.pt"
    output_dir = project_root / "outputs"
    app = create_app(model_path=model_path, output_dir=output_dir)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
