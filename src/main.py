"""Application entry point — uvicorn server launcher."""

import uvicorn

from src.core.config import get_config


def main():
    cfg = get_config()
    # CRITICAL: workers must be 1 — singletons (MLPipeline, AlertService, ModelManager)
    # break with multiple workers. Use uvicorn with --workers only if singletons
    # are removed or replaced with shared state (Redis/DB).
    uvicorn.run(
        "src.api.app:app",
        host=cfg["service"]["host"],
        port=cfg["service"]["port"],
        workers=1,
        log_level=cfg["service"]["log_level"],
        access_log=True,
        use_colors=True,
    )


if __name__ == "__main__":
    main()
