
import os
import pandas as pd
import numpy as np
import logging


# get the airflow.task logger
task_logger = logging.getLogger("airflow.task")


def install_requirements():
    os.system = ('cmd /k "pip install -r SDG-Case-Study/requirements.txt"')



def _generate_pip_install_cmd_from_file(tmp_dir: str, requirements_file_path: str) -> List[str]:
cmd = [f'{tmp_dir}/bin/pip', 'install', '-r']
return cmd + [requirements_file_path]