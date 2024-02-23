import os
from typing import Dict, Optional
import logging


class Context:
    # context class for a single evaluation
    def __init__(self, config) -> None:
        self.config = config
        self.dataset = None
        self.predictions = {}
        self.results = {}
        self.logger = logging.getLogger(__name__)
