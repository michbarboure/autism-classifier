from loguru import logger
import logging
import sys
from datetime import datetime
import os


def configure_custom_logging(logdir='logs', loguru_level='INFO', mne_level='WARNING'):
    """Configure custom logging to file and stdout for loguru and MNE.

    Parameters
    ----------
    logdir : str, optional
        Directory for saving log files, by default 'logs'
    """

    # Configure my logging to stderr
    logger.remove()
    logger.add(sys.stderr, level=loguru_level)

    # Set the MNE logging level (for stdout)
    logging.getLogger('mne').handlers[0].setLevel(mne_level)

    # Make the logging directory if it doesn't exist
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Get the current timestamp
    time = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    # Create the name of the logfile using timestamp
    logfile = f"logs/{time}.log"

    # Tell loguru to save logs to the logfile at specified level
    logger.add(logfile, level="DEBUG")

    # Create a new file handler for MNE logs to go to logfile
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    logging.getLogger('mne').addHandler(fh)

    logger.debug("Custom logging configured.")
