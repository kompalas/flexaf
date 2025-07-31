import numpy as np
import traceback
import logging
from collections import OrderedDict
from src.utils import env_cfg
from src.selection.gate_fs import run_differentiable_feature_selection
from src.selection.heuristic_fs import run_heuristic_feature_selection
from src.selection.greedy_fs import run_greedy_feature_selection
from src.selection.statistical import run_statistical_feature_selection
from src.selection.simple_eval import perform_basic_evaluation

logger = logging.getLogger(__name__)


def main():
    args = env_cfg()
    args.resdir = logging.getLogger().logdir

    if args.execute_differentiable_feature_selection:
        args.name = f"gate_fs_{args.name}"
        logger.info("Running differentiable feature selection with ConcreteGate...")
        run_differentiable_feature_selection(args)

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