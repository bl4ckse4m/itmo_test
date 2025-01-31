import logging
import os
import sys

from aiologger import Logger
from aiologger.formatters.base import Formatter
from aiologger.handlers.files import AsyncFileHandler
from aiologger.handlers.streams import AsyncStreamHandler
from aiologger.levels import LogLevel



def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)