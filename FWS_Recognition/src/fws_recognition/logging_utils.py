from pathlib import Path
import logging
import builtins


def setup_file_logger(log_path="logs/run.log", logger_name="fws_recognition"):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def make_print_and_log(logger):
    raw_print = builtins.print

    def print_and_log(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        msg = sep.join(str(x) for x in args)
        if end != "\n":
            msg = msg + end

        raw_print(*args, **kwargs)
        logger.info(msg.rstrip("\n"))

    return print_and_log
