import os
import numpy as np
import pandas as pd
import tensorflow as tf
import copy
import os
import logging
import sys
from src.utils import transform_categorical
from src.evaluation.hw_eval_single_model import hw_eval_with_single_model
import argparse


def setup_logger(log_path):
    logger = logging.getLogger("LTP")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Prevent propagation to root logger (fixes console duplication!)
    logger.propagate = False

    # File formatter (rich detail)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] (%(processName)s) %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S"
    )

    # Console formatter (simple)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_formatter)

    # Attach handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def compute_model_sparsity(model):
    total = 0
    zero = 0
    for weight in model.trainable_weights:
        if "kernel" in weight.name:
            w = weight.numpy()
            total += w.size
            zero += np.sum(w == 0)
    return zero / total if total > 0 else 0

def apply_pruning_masks(model, masks):
    for w, mask in zip(model.trainable_weights, masks):
        if "kernel" in w.name:
            w.assign(w * mask)

def get_weight_masks(model, target_sparsity):
    masks = []
    for weight in model.trainable_weights:
        if "kernel" in weight.name:
            w = weight.numpy()
            k = int(np.prod(w.shape) * (1 - target_sparsity))
            if k <= 0:
                mask = np.zeros_like(w)
            else:
                flat = np.abs(w).flatten()
                threshold = np.partition(flat, -k)[-k]
                mask = (np.abs(w) >= threshold).astype(np.float32)
            masks.append(mask)
        else:
            masks.append(np.ones_like(weight.numpy(), dtype=np.float32))
    return masks

def reset_weights(model, original_weights):
    for layer, orig in zip(model.layers, original_weights):
        layer.set_weights(copy.deepcopy(orig))

def evaluate_model(model, x_val, y_val, batch_size):
    results = model.evaluate(x_val, y_val, verbose=0, batch_size=batch_size)
    acc = results[model.metrics_names.index('compile_metrics')]  # this assumes 'compile_metrics' is the accuracy metric
    return acc

def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size=128, masks=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=max(1, min(5, epochs-5)),
        restore_best_weights=True,
        verbose=0
    )
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, verbose=0, callbacks=[early_stopping], 
                        batch_size=batch_size)
    if masks is not None:
        apply_pruning_masks(model, masks)
    val_acc = history.history['val_accuracy'][-1]
    return val_acc

def save_model_checkpoint(model, path, label):
    model.save(os.path.join(path, f'{label}.keras'))

def save_masks(masks, path, label):
    mask_dict = {f'mask_{i}': m for i, m in enumerate(masks)}
    np.savez_compressed(os.path.join(path, f'{label}_masks.npz'), **mask_dict)


def post_training_pruning(model,
                          x_val, y_val,
                          initial_sparsity=0.5,
                          sparsity_step=0.05,
                          accuracy_drop_threshold=0.01,
                          max_iterations=20,
                          batch_size=128,
                          savedir="results",
                          logger=None):
    """
    Post-training pruning: Iteratively adjusts sparsity up or down to find the maximum
    sparsity that satisfies the accuracy threshold, without retraining.

    Starts from the unpruned model, increases sparsity if successful, and backtracks if not.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model to prune.
    x_val, y_val : array-like
        Validation/test data.
    initial_sparsity : float
        Starting sparsity fraction.
    sparsity_step : float
        Amount to increase/decrease sparsity each step.
    accuracy_drop_threshold : float
        Allowed accuracy drop from baseline.
    max_iterations : int
        Max number of iterations.
    batch_size : int
        Batch size for evaluation.
    savedir : str
        Directory to save results.
    logger : logging.Logger
        Logger for messages.
    """
    os.makedirs(savedir, exist_ok=True)

    # Baseline
    baseline_acc = evaluate_model(model, x_val, y_val, batch_size=batch_size)
    baseline_sparsity = compute_model_sparsity(model) * 100
    logger.info(f"[PTP] Baseline accuracy: {baseline_acc:.4f}")
    logger.info(f"[PTP] Baseline sparsity: {baseline_sparsity:.2f}%")

    # Store best model (start with original)
    best_acc = baseline_acc
    best_sparsity = baseline_sparsity
    best_masks = None
    best_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers]

    target_sparsity = initial_sparsity
    iteration = 0

    while iteration < max_iterations and 0.01 < target_sparsity < 0.99:
        logger.info("")
        logger.info(f"[PTP] Iteration {iteration} - Target Sparsity: {target_sparsity:.2f}")

        # Start from best known weights
        reset_weights(model, best_weights)

        # Apply pruning
        masks = get_weight_masks(model, target_sparsity)
        apply_pruning_masks(model, masks)

        # Evaluate
        sparsity = compute_model_sparsity(model) * 100
        acc = evaluate_model(model, x_val, y_val, batch_size=batch_size)
        logger.info(f"[PTP] Sparsity: {sparsity:.2f}%, Accuracy: {acc:.4f}")

        if acc >= baseline_acc - accuracy_drop_threshold:
            logger.info("✅ [PTP] Passed accuracy constraint. Saving as best model.")
            best_acc = acc
            best_sparsity = sparsity
            best_masks = copy.deepcopy(masks)
            best_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers]
            save_model_checkpoint(model, savedir, f'ptp_iter_{iteration}')
            save_masks(masks, savedir, f'ptp_iter_{iteration}')
            target_sparsity += sparsity_step
        else:
            logger.info("❌ [PTP] Failed accuracy constraint. Restoring last best model.")
            reset_weights(model, best_weights)
            if best_masks:
                apply_pruning_masks(model, best_masks)
            target_sparsity -= sparsity_step

        iteration += 1

    # Final save
    if best_masks:
        reset_weights(model, best_weights)
        apply_pruning_masks(model, best_masks)
        model.save(os.path.join(savedir, 'ptp_final.keras'))
        save_masks(best_masks, savedir, 'ptp_final')

    # Summary
    logger.info("")
    logger.info("[PTP] Summary")
    logger.info(f"➡️  Original Accuracy: {baseline_acc:.4f}")
    logger.info(f"➡️  Original Sparsity: {baseline_sparsity:.2f}%")
    logger.info(f"✅ Final Accuracy: {best_acc:.4f}")
    logger.info(f"✅ Final Sparsity: {best_sparsity:.2f}%")

    return model


def lottery_ticket_pruning(model, 
                            x_train, y_train, 
                            x_val, y_val,
                            pruning_retraining_epochs=5,
                            final_retraining_epochs=10,
                            initial_sparsity=0.8,
                            sparsity_divider=1.2,
                            accuracy_drop_threshold=0.01,
                            max_retries=3,
                            max_iterations=10,
                            batch_size=128,
                            savedir="results",
                            logger=None):
    """
    Perform Lottery Ticket Pruning on the given model with iterative magnitude pruning.
    """
    os.makedirs(savedir, exist_ok=True)

    # --- Baseline evaluation ---
    logger.info("[LTP] Evaluating pretrained model...")
    baseline_acc = evaluate_model(model, x_val, y_val, batch_size=batch_size)
    baseline_sparsity = compute_model_sparsity(model) * 100
    logger.info(f"[LTP] Baseline accuracy: {baseline_acc:.4f}")
    logger.info(f"[LTP] Baseline sparsity: {baseline_sparsity:.2f}%")

    # Save original weights for resets
    original_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers]
    best_acc = baseline_acc
    best_model_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers]
    best_masks = None

    target_sparsity = initial_sparsity
    retries = 0
    iteration = 0

    # --- Iterative pruning loop ---
    while (
        retries <= max_retries
        and 0.0 < target_sparsity < 0.99
        and iteration < max_iterations
    ):
        logger.info("")
        logger.info(f"[LTP] Iteration {iteration} - Target Sparsity: {target_sparsity:.2f}")

        # Reset to original dense weights
        reset_weights(model, original_weights)

        # Prune and enforce zero weights
        masks = get_weight_masks(model, target_sparsity)
        apply_pruning_masks(model, masks)
        sparsity = compute_model_sparsity(model) * 100
        logger.info(f"[LTP] Model sparsity: {sparsity:.2f}%")

        # Retrain the pruned model
        acc = train_model(model, x_train, y_train, x_val, y_val,
                          pruning_retraining_epochs, batch_size=batch_size, masks=masks)
        logger.info(f"[LTP] Accuracy after pruning: {acc:.4f}")

        # Check if accuracy constraint is met
        if acc >= baseline_acc - accuracy_drop_threshold:
            logger.info("✅ [LTP] Accuracy constraint met. Saving checkpoint and masks.")
            best_acc = acc
            best_model_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers]
            best_masks = copy.deepcopy(masks)
            save_model_checkpoint(model, savedir, f'ltp_iter_{iteration}')
            save_masks(masks, savedir, f'ltp_iter_{iteration}')
            retries = 0

            # Increase sparsity for next iteration
            target_sparsity = min(target_sparsity * sparsity_divider, 0.99)
            logger.info(f"[LTP] New target sparsity for next iteration: {target_sparsity:.2f}")
        else:
            logger.info("❌ [LTP] Accuracy dropped. Reducing sparsity and retrying.")
            retries += 1
            target_sparsity /= sparsity_divider

        iteration += 1

    # --- Final retraining of the winning ticket ---
    logger.info("")
    logger.info(f"[LTP] Final retraining of the winning ticket for {final_retraining_epochs} epochs...")
    reset_weights(model, best_model_weights)
    if best_masks:
        apply_pruning_masks(model, best_masks)
    final_acc = train_model(model, x_train, y_train, x_val, y_val,
                            final_retraining_epochs, batch_size=batch_size, masks=best_masks)

    # Save final model and masks
    model.save(os.path.join(savedir, 'ltp_final.keras'))
    if best_masks:
        save_masks(best_masks, savedir, 'ltp_final')

    # Compute final sparsity
    final_sparsity = compute_model_sparsity(model) * 100

    # --- Summary ---
    logger.info("")
    logger.info("[LTP] Summary")
    logger.info(f"➡️  Original Accuracy: {baseline_acc:.4f}")
    logger.info(f"➡️  Original Sparsity: {baseline_sparsity:.2f}%")
    logger.info(f"✅ Final Accuracy after retraining: {final_acc:.4f}")
    logger.info(f"✅ Final Sparsity: {final_sparsity:.2f}%")

    # Sanity check
    if final_acc < baseline_acc - accuracy_drop_threshold:
        raise ValueError(
            f"Final accuracy {final_acc:.4f} is below the allowed drop threshold "
            f"of {baseline_acc - accuracy_drop_threshold:.4f}"
        )

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Lottery Ticket Pruning runner")
    parser.add_argument('--expdir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--sparsity', type=float, required=True, help='Sparsity value')
    parser.add_argument('--trial', type=int, required=True, help='Trial number')
    parser.add_argument('--pruning-retraining-epochs', type=int, default=5, help='Number of epochs for retraining during pruning')
    parser.add_argument('--final-retraining-epochs', type=int, default=10, help='Number of epochs for final retraining after pruning')
    parser.add_argument('--initial-sparsity', type=float, default=0.9, help='Initial sparsity for pruning')
    parser.add_argument('--sparsity-divider', type=float, default=1.5, help='Factor by which to divide sparsity in each iteration')
    parser.add_argument('--accuracy-drop-threshold', type=float, default=0.02, help='Maximum allowed drop in accuracy during pruning')
    parser.add_argument('--weight-precision', type=int, default=8, help='Weight precision for HW evaluation')
    parser.add_argument('--input-precision', type=int, default=4, help='Input precision for HW evaluation')
    parser.add_argument('--post-training-pruning', action='store_true', help='Use post-training pruning instead of lottery ticket pruning')
    args = parser.parse_args()

    expdir = args.expdir
    fold = args.fold
    sparsity = args.sparsity
    trial = args.trial
    pruning_retraining_epochs = args.pruning_retraining_epochs
    final_retraining_epochs = args.final_retraining_epochs
    initial_sparsity = args.initial_sparsity
    sparsity_divider = args.sparsity_divider
    accuracy_drop_threshold = args.accuracy_drop_threshold
    weight_precision = args.weight_precision
    input_precision = args.input_precision
    
    # expdir = '/home/balaskas/flexaf/saved_logs/wesad_merged/diff_fs_fcnn_wesad_merged___2025.08.06-16.16.48.524'
    # fold = 0
    # sparsity = 0.5
    # trial = 2
    # pruning_retraining_epochs = 5
    # final_retraining_epochs = 10
    # initial_sparsity = 0.9
    # sparsity_divider = 1.2
    # accuracy_drop_threshold = 0.01

    # load the saved model and train/test sets
    model_name = f'gate_model_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.keras'
    model = tf.keras.models.load_model(
        os.path.join(expdir, 'results', 'classifiers', model_name),
    )
    x_test_sub = np.load(os.path.join(expdir, 'results', 'data', f'x_test_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.npy'))
    y_test = np.load(os.path.join(expdir, 'results', 'data', f'y_test_fold{fold}.npy'))
    y_test_categ = transform_categorical(y_test, num_classes=model.output_shape[-1])
    x_train_sub = np.load(os.path.join(expdir, 'results', 'data', f'x_train_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.npy'))
    y_train = np.load(os.path.join(expdir, 'results', 'data', f'y_train_fold{fold}.npy'))
    y_train_categ = transform_categorical(y_train, num_classes=model.output_shape[-1])
    x_test_analog = pd.read_csv(os.path.join(expdir, 'results', 'data', f'analog_test_set', f'analog_test_set_fold{fold}_pruned_{sparsity:.3f}_trial{trial}.csv'))

    logger = setup_logger(os.path.join(expdir, "lottery_ticket.log"))
    savedir = os.path.join(expdir, 'checkpoints')

    if args.post_training_pruning:
        logger.info("[LTP] Running Post-Training Pruning...")
        pruned_model = post_training_pruning(
            model=model,
            x_val=x_test_sub,
            y_val=y_test_categ,
            initial_sparsity=initial_sparsity,
            sparsity_step=0.1,
            accuracy_drop_threshold=accuracy_drop_threshold,
            max_iterations=20,
            batch_size=32,
            savedir=savedir,
            logger=logger
        )

    else:
        logger.info("[LTP] Running Lottery Ticket Pruning...")
        pruned_model = lottery_ticket_pruning(
            model=model,
            x_train=x_train_sub,
            y_train=y_train_categ,
            # x_val=x_test_sub,
            x_val=x_test_analog.values,
            y_val=y_test_categ,
            pruning_retraining_epochs=pruning_retraining_epochs,
            final_retraining_epochs=final_retraining_epochs,
            initial_sparsity=initial_sparsity,
            sparsity_divider=sparsity_divider,
            accuracy_drop_threshold=accuracy_drop_threshold,
            max_retries=7,
            max_iterations=10,
            savedir=savedir,
            logger=logger
        )

    resdir = os.path.join(expdir, 'lottery_ticket_results')
    os.makedirs(resdir, exist_ok=True)

    hw_eval_with_single_model(
        resdir=resdir,
        model=pruned_model,
        # x_test=x_test_sub,
        x_test=x_test_analog.values,
        y_test_categ=y_test_categ,
        weight_precision=weight_precision,
        input_precision=input_precision,
        experiment_name='lottery_ticket_eval'
    )