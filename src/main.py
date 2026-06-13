"""Application entry point — uvicorn server launcher."""

import uvicorn

from src.core.config import get_config


def main():
    cfg = get_config()
    uvicorn.run(
        "src.api.app:app",
        host=cfg["service"]["host"],
        port=cfg["service"]["port"],
        workers=cfg["service"].get("workers", 4),
        log_level=cfg["service"]["log_level"],
        access_log=True,
        use_colors=True,
    )


if __name__ == "__main__":
    main()
