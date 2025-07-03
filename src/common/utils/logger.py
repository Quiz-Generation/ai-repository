import io
import logging
import logging.handlers
import os
import sys
import time
import traceback
from functools import wraps

from colorlog import ColoredFormatter

from src.common.conf.settings import settings
from src.common.constants import DEFAULT_LOGGING_LEVEL

ROOT_PKG = "quiz-api"

LOG_COLORS_CONFIG = {
    'DEBUG':    'cyan',
    'INFO':     'green',
    'WARNING':  'yellow',
    'ERROR':    'red',
    'CRITICAL': 'bold_red',
}

# ✅ Custom Formatter (흰색 날짜 + traceback 강조)
class CustomColoredFormatter(ColoredFormatter):
    def format(self, record):
        # 날짜 흰색 처리
        record.white_asctime = f"\033[37m{self.formatTime(record, self.datefmt)}\033[0m"

        # traceback 색 강조
        if record.exc_info:
            record.msg = f"\033[33m{record.msg}\033[0m"  # yellow
        return super().format(record)

# ✅ 콘솔 포맷 (날짜는 흰색, 로그 레벨은 레벨별 색)
STREAM_HANDLER_FORMAT = (
    "%(white_asctime)s | %(log_color)s%(levelname)-8s | %(name)s "
    "[%(filename)s:%(funcName)s:%(lineno)d] >> %(message)s"
)

# ✅ 파일 포맷 (색상 없음)
FILE_HANDLER_FORMAT = (
    "[%(asctime)s] | %(levelname)-8s | %(name)s "
    "[%(filename)s:%(funcName)s:%(lineno)d] >> %(message)s"
)


def set_logger(pkg: str, log_base_dir: str = settings.DEFAULT_LOGGING_PATH, log_level: int = DEFAULT_LOGGING_LEVEL) -> logging.Logger:
    if not pkg:
        raise Exception("pkg(package) must not be None")

    logger = logging.getLogger(pkg)
    logger.setLevel(log_level)

    if not logger.handlers:
        # Stream Handler
        stream_handler = logging.StreamHandler()
        stream_formatter = CustomColoredFormatter(
            STREAM_HANDLER_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors=LOG_COLORS_CONFIG
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # File Handler
        _pkg = pkg.replace("/", "")
        pkg_dirs = [p for p in _pkg.split(".") if p]
        log_file = pkg_dirs[-1]
        pkg_dir = "/".join(pkg_dirs[:-1])
        log_file_dir = f"{log_base_dir}/{pkg_dir}" if pkg_dir else log_base_dir
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = f"{log_file_dir}/{log_file}.log"

        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
        logger.addHandler(file_handler)

    return logger


def handle_exception(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger(ROOT_PKG)
    formatted_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error("Unhandled exception occurred:\n%s", formatted_traceback)


def init_logger():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    sys.excepthook = handle_exception

    _ = set_logger(ROOT_PKG)
    _ = set_logger(f"{ROOT_PKG}")

    for lib in ["httpx", "httpx._client", "httpx._config", "httpcore", "httpcore.http11"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def aio_log_method_call(pkg):
    def decor(f):
        @wraps(f)
        async def inner(*args, **kwargs):
            logger = set_logger(pkg)
            start_time = time.time()
            logger.debug(f"Start '{f.__name__}' - args({args}) - kwargs({kwargs})")
            res = await f(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"End '{f.__name__}' - duration({end_time - start_time:.3f}s)")
            return res
        return inner
    return decor


def log_method_call(pkg):
    def decor(f):
        @wraps(f)
        def inner(*args, **kwargs):
            logger = set_logger(pkg)
            start_time = time.time()
            logger.debug(f"Start '{f.__name__}' - args({args}) - kwargs({kwargs})")
            res = f(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"End '{f.__name__}' - duration({end_time - start_time:.3f}s)")
            return res
        return inner
    return decor


if __name__ == "__main__":
    init_logger()
    logger = logging.getLogger(ROOT_PKG)
    logger.info("Logger initialized.")