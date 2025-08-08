import pickle
import traceback
import logging
import os
import tensorflow as tf
from src.utils import env_cfg
from src.selection.gate_fs import run_gated_model_pruning_experiment
from src.selection.gate_fs_gpu import run_gated_model_pruning_experiment as run_gated_model_pruning_experiment_gpu
from src.selection.heuristic_fs import run_heuristic_feature_selection
from src.selection.greedy_fs import run_greedy_feature_selection
from src.selection.statistical import run_statistical_feature_selection
from src.selection.simple_eval import perform_basic_evaluation

logger = logging.getLogger(__name__)


def main():
    args = env_cfg()
    args.resdir = logging.getLogger().logdir
    with open(os.path.join(args.resdir, 'args.yaml'), 'wb') as f:
        pickle.dump(args, f)

    if args.execute_differentiable_feature_selection:
        args.name = f"gate_fs_{args.name}"
        logger.info("Running differentiable feature selection with ConcreteGate...")
        # run_differentiable_feature_selection(args)
        if len(tf.config.list_physical_devices('GPU')) > 0:
            run_gated_model_pruning_experiment_gpu(args)
        else:
            run_gated_model_pruning_experiment(args)

    elif args.execute_heuristic_feature_selection:
        args.name = f"heuristic_fs_{args.name}"
        logger.info("Running heuristic feature selection...")
        run_heuristic_feature_selection(args)

    elif args.execute_greedy_feature_selection:
        args.name = f"greedy_fs_{args.name}"
        logger.info("Running greedy feature selection...")
        run_greedy_feature_selection(args)

    elif args.execute_statistical_feature_selection:
        args.name = f"statistical_fs_{args.name}"
        logger.info("Running statistical feature selection...")
        run_statistical_feature_selection(args)

    else:
        logger.info("No feature selection method specified, running basic evaluation...")
        perform_basic_evaluation(args)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if logger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = logger.handlers
            logger.handlers = [h for h in logger.handlers if type(h) != logging.StreamHandler]
            logger.error(traceback.format_exc())
            logger.handlers = handlers_bak
        raise
    except KeyboardInterrupt:
        logger.info("")
        logger.info("--- Keyboard Interrupt ---")
    finally:
        if logger.handlers:
            logfiles = [handler.baseFilename for handler in logger.handlers if
                        type(handler) == logging.FileHandler]
            logger.info(f"Log file(s) for this run in {' | '.join(logfiles)}")