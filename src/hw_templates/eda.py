import os
import subprocess
import shutil
import logging
import pandas as pd
from collections import namedtuple
from src import project_dir
from src.utils import get_timestamp


logger = logging.getLogger(__name__)

SynthesisResults = namedtuple("SynthesisResults", ['area', 'delay', 'power'])


def create_new_evaluation_dir(prefix=''):
    """Create a new evaluation directory."""
    os.makedirs(os.path.join(project_dir, 'hw_eval', 'test'), exist_ok=True)
    prefix = 'test__' if prefix == '' else prefix + '__'
    dirname = f"{prefix}{get_timestamp()}"
    hw_eval_dir = os.path.join(project_dir, 'hw_eval', 'test', dirname)

    # create the hw evaluation directory
    command = f"tar -xvf {project_dir}/hw_eval/eda_scripts.tar.gz -C {project_dir}/hw_eval/ && " \
              f"mv {project_dir}/hw_eval/eda_scripts/ {hw_eval_dir}"
    subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

    return hw_eval_dir


def execute_hw_evaluation(savedir, hdl_dir, sim_dir, synclk_ms, copy_results=True, cleanup=False, prefix=''):
    """Execute synthesis and simulation and return the results."""
    hw_eval_dir = create_new_evaluation_dir(prefix)

    # Remove hw evaluation directories if they exist
    eval_hdl_dir = os.path.join(hw_eval_dir, 'hdl')
    eval_sim_dir = os.path.join(hw_eval_dir, 'sim')
    reports_dir = os.path.join(hw_eval_dir, 'reports')
    gate_dir = os.path.join(hw_eval_dir, 'gate')
    if os.path.exists(eval_hdl_dir):
        shutil.rmtree(eval_hdl_dir)
    if os.path.exists(eval_sim_dir):
        shutil.rmtree(eval_sim_dir)
    if os.path.exists(reports_dir):
        shutil.rmtree(reports_dir)
    if os.path.exists(gate_dir):
        shutil.rmtree(gate_dir)
    # copy the HDL and simulation directories to the evaluation directory
    shutil.copytree(hdl_dir, eval_hdl_dir)
    shutil.copytree(sim_dir, eval_sim_dir)

    # execute the hw evaluation
    command = f"cd {hw_eval_dir} && ./run/single_eval.sh {synclk_ms * 1e6}"
    logger.debug(f"Executing command: {command}")
    try:
        p = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error during HW evaluation. Make sure that tools have been initialized (e.g., Synopsys)")

    # write the stdout to a log file
    log_file_path = os.path.join(savedir, 'hw_eval.log')
    with open(log_file_path, 'w') as log_file:
        log_file.write(p.stdout)
        log_file.write(p.stderr)

    # copy reports back to the experiment directory
    if copy_results:
        if os.path.exists(os.path.join(savedir, 'reports')):
            shutil.rmtree(os.path.join(savedir, 'reports'))
        if os.path.exists(os.path.join(savedir, 'gate')):
            shutil.rmtree(os.path.join(savedir, 'gate'))
        shutil.copytree(reports_dir, os.path.join(savedir, 'reports'))
        shutil.copytree(gate_dir, os.path.join(savedir, 'gate'))

    # obtain the results
    results_file = os.path.join(reports_dir, 'results.csv')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"There was an error during the HW evaluation run. Please check the logfile.")

    df = pd.read_csv(results_file)
    area = df['Area'].values[0]
    delay = df['Delay'].values[0]
    power = df['Power'].values[0]

    if cleanup:
        shutil.rmtree(hw_eval_dir)

    return SynthesisResults(area=area, delay=delay, power=power)


def execute_rtl_evaluation(savedir, hdl_dir, sim_dir, cleanup=False, prefix=''):
    """Execute synthesis and simulation and return the results."""
    hw_eval_dir = create_new_evaluation_dir(prefix)

    # Remove hw eval directories if they exist
    eval_hdl_dir = os.path.join(hw_eval_dir, 'hdl')
    eval_sim_dir = os.path.join(hw_eval_dir, 'sim')
    if os.path.exists(eval_hdl_dir):
        shutil.rmtree(eval_hdl_dir)
    if os.path.exists(eval_sim_dir):
        shutil.rmtree(eval_sim_dir)
    # create the hdl and sim directories
    shutil.copytree(hdl_dir, eval_hdl_dir)
    shutil.copytree(sim_dir, eval_sim_dir)

    # execute the hw evaluation
    command = f"cd {hw_eval_dir} && make rtl_sim"
    logger.debug(f"Executing command: {command}")
    try:
        p = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error during HW evaluation. Make sure that tools have been initialized (e.g., Synopsys)")

    # write the stdout to a log file
    log_file_path = os.path.join(savedir, 'hw_eval.log')
    with open(log_file_path, 'w') as log_file:
        log_file.write(p.stdout)
        log_file.write(p.stderr)

    if cleanup:
        shutil.rmtree(hw_eval_dir)

    return SynthesisResults(area=None, delay=None, power=None)
