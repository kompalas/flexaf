import os
import numpy as np
import tensorflow as tf
import copy
import os
import logging
import sys
from src.utils import transform_categorical


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

def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size=128):
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, verbose=0, batch_size=batch_size)
    val_acc = history.history['val_accuracy'][-1]
    return val_acc

def save_model_checkpoint(model, path, label):
    model.save(os.path.join(path, f'{label}.keras'))

def save_masks(masks, path, label):
    mask_dict = {f'mask_{i}': m for i, m in enumerate(masks)}
    np.savez_compressed(os.path.join(path, f'{label}_masks.npz'), **mask_dict)


def lottery_ticket_pruning(model, x_train, y_train, x_val, y_val,
                            pruning_epochs=5,
                            retraining_epochs=10,
                            initial_sparsity=0.8,
                            sparsity_divider=2,
                            accuracy_drop_threshold=0.01,
                            max_retries=3,
                            max_iterations=10,
                            batch_size=128,
                            savedir="results",
                            logger=None):
    """Perform Lottery Ticket Pruning on the given model."""
    os.makedirs(savedir, exist_ok=True)

    logger.info("[LTP] Evaluating pretrained model...")
    baseline_acc = evaluate_model(model, x_val, y_val, batch_size=batch_size)
    baseline_sparsity = compute_model_sparsity(model) * 100
    logger.info(f"[LTP] Baseline accuracy: {baseline_acc:.4f}")
    logger.info(f"[LTP] Baseline sparsity: {baseline_sparsity:.2f}%")

    original_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers]
    best_acc = baseline_acc
    best_model_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers]
    best_masks = None

    target_sparsity = initial_sparsity
    retries = 0
    iteration = 0

    while retries <= max_retries and target_sparsity >= 0.02 and iteration < max_iterations:
        logger.info("")
        logger.info(f"[LTP] Iteration {iteration} - Target Sparsity: {target_sparsity:.2f}")

        reset_weights(model, original_weights)

        masks = get_weight_masks(model, target_sparsity)
        apply_pruning_masks(model, masks)
        sparsity = compute_model_sparsity(model) * 100
        logger.info(f"[LTP] Model sparsity: {sparsity:.2f}%")

        acc = train_model(model, x_train, y_train, x_val, y_val,
                          pruning_epochs, batch_size=batch_size)
        logger.info(f"[LTP] Accuracy after pruning: {acc:.4f}")

        if acc >= baseline_acc - accuracy_drop_threshold:
            logger.info("✅ [LTP] Accuracy constraint met. Saving checkpoint and masks.")
            best_acc = acc
            best_model_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers]
            best_masks = copy.deepcopy(masks)
            save_model_checkpoint(model, savedir, f'ltp_iter_{iteration}')
            save_masks(masks, savedir, f'ltp_iter_{iteration}')
            retries = 0
        else:
            logger.info("❌ [LTP] Accuracy dropped. Reducing sparsity and retrying.")
            retries += 1
            target_sparsity /= sparsity_divider

        iteration += 1

    logger.info("")
    logger.info(f"[LTP] Final retraining of the winning ticket for {retraining_epochs} epochs...")
    reset_weights(model, best_model_weights)
    if best_masks:
        apply_pruning_masks(model, best_masks)
    final_acc = train_model(model, x_train, y_train, x_val, y_val,
                            retraining_epochs, batch_size=batch_size)
    save_model_checkpoint(model, savedir, 'ltp_final')
    if best_masks:
        save_masks(best_masks, savedir, 'ltp_final')

    final_sparsity = compute_model_sparsity(model) * 100
    logger.info("")
    logger.info("[LTP] Summary")
    logger.info(f"➡️  Original Accuracy: {baseline_acc:.4f}")
    logger.info(f"➡️  Original Sparsity: {baseline_sparsity:.2f}%")
    logger.info(f"✅ Final Accuracy after retraining: {final_acc:.4f}")
    logger.info(f"✅ Final Sparsity: {final_sparsity:.2f}%")

    return model



if __name__ == "__main__":

    # load the saved model and test set and evaluate its accuracy
    expdir = '/home/balaskas/flexaf/saved_logs/wesad_merged/diff_fs_fcnn_wesad_merged___2025.08.06-16.16.48.524'
    fold = 0
    sparsity = 0.5
    trial = 2
    
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

    logger = setup_logger(os.path.join(expdir, "lottery_ticket.log"))
    savedir = os.path.join(expdir, 'checkpoints')

    lottery_ticket_pruning(
        model,
        x_train_sub, y_train_categ,
        x_test_sub, y_test_categ,
        pruning_epochs=3,
        retraining_epochs=5,
        initial_sparsity=0.9,
        sparsity_divider=1.2,
        accuracy_drop_threshold=0.01,
        savedir=savedir,
        logger=logger
    )