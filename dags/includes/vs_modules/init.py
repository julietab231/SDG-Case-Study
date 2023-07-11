
import os
import pandas as pd
import numpy as np
import logging


# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")


def install_requirements():
    os.system = ('cmd /k "pip install -r SDG-Case-Study/requirements.txt"')

