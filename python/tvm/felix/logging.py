import logging
import os
import time
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
_logger = logging.getLogger(__name__)


def init_logging(log_dir: PathLike, verbose_logging: bool = False):
    from logging.config import dictConfig

    timestr = time.strftime("%Y.%m.%d-%H%M%S.log")
    output_dir = Path(log_dir or ".")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = output_dir / timestr
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(levelname)s %(filename)s:%(lineno)d: %(message)s"},
            "detailed": {
                "format": "[%(asctime)-15s] "
                "%(levelname)7s %(name)s: "
                "%(message)s "
                "@%(filename)s:%(lineno)d"
            },
        },
        "handlers": {
            "console": {
                "()": TqdmStreamHandler,
                "formatter": "simple",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": file_path.as_posix(),
                "mode": "a",  # Because we may apply this config again, want to keep existing content
                "formatter": "detailed",
                "level": "DEBUG",
            },
        },
        "root": {
            "level": "DEBUG" if verbose_logging else "INFO",
            "handlers": ["console", "file"],
        },
    }
    dictConfig(logging_config)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    _logger.info("Logging to %s", file_path)
    return logging_config


class TqdmStreamHandler(logging.Handler):
    """tqdm-friendly logging handler. Uses tqdm.write instead of print for logging."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        import tqdm

        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except:  # noqa: E722
            self.handleError(record)
