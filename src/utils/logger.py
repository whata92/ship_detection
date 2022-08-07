import os
import logging
from datetime import datetime


LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "critical": logging.CRITICAL,
}


def init_log(name, level):
    """function to setup the logger
    Args:
        name: name of the logger
        level: level of the logger.
                acceptable value are in the LEVELS dictionary
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # terminal
    ch = logging.StreamHandler()
    ch.setLevel(LEVELS[level])
    ch_formatter = logging.Formatter(fmt="%(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # log file
    os.makedirs("./logs", exist_ok=True)
    now = datetime.now()
    logfile = "./logs/" + now.strftime("%Y%m%d_%H-%M-%S") + ".log"
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
