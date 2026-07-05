from __future__ import annotations

import sys
from pathlib import Path


def create_new_app(*args, **kwargs):
    from app.gui_new import create_new_app as _create_new_app

    return _create_new_app(*args, **kwargs)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "Reference" / "yolo26n.pt"
    output_dir = project_root / "outputs"
    app = create_new_app(model_path=model_path, output_dir=output_dir)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
