from sklearn.tree import _tree
import numpy as np
import logging
from src.hw_templates.utils import logging_cfg, get_width, generate_testbench_code, natural_sort_key
from src.utils import stochastic_rounding


logger = logging.getLogger(__name__)


def tree_to_c(tree, inpprefix, logger_c, inp_precisions=None):
    if hasattr(tree, 'tree_'):
        tree_ = tree.tree_
    else:
        tree_ = tree

    feature_name = [
        inpprefix + str(i) if i != _tree.TREE_UNDEFINED else ''
        for i in tree_.feature
    ]
    inputs = sorted(set(filter(None, feature_name)), key=natural_sort_key)  # filter removes empty strings
    inp_precisions = dict(zip(inputs, inp_precisions))

    logger_c.debug("void dt_forward(float input[], int* output) {")

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            inp_precision = inp_precisions.get(name, None)

            if inp_precision is not None:
                threshold = int(2 ** inp_precision * tree_.threshold[node])
            else:
                # threshold = int(tree_.threshold[node])
                threshold = tree_.threshold[node]

            logger_c.debug(f"{indent}if (input[{name[1:]}] <= {threshold}) {{")

            recurse(tree_.children_left[node], depth + 1)
            logger_c.debug(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            logger_c.debug(f"{indent}}}")
        else:
            class_value = np.argmax(tree_.value[node])
            logger_c.debug(f"{indent}*output = {class_value};")

    recurse(0, 1)
    logger_c.debug("}")


def tree_to_verilog(tree, inp_widths, comp_bits, inpprefix, outname, input_separator, simclk_ns,
                    tb_inputs_file, tb_output_file, logger_v, logger_tb, metadata, rounding_f=np.floor, signed=False):
    tree_ = tree.tree_
    signed_prefix = 'signed ' if signed else ''

    # round the tree thresholds to integers
    # int_thresholds = rounding_f(tree_.threshold)
    # for i in range(len(tree_.threshold)):
    #     tree_.threshold[i] = int_thresholds[i]

    out_width = get_width(max(tree.classes_))
    feature_name = [
        inpprefix + str(i) if i != _tree.TREE_UNDEFINED else ''
        for i in tree_.feature 
    ]
    # keep unique features as inputs
    inputs = sorted(set(filter(None, feature_name)), key=natural_sort_key)  # filter removes empty strings
    metadata['inputs'] = inputs
    inp_width = dict(zip(inputs, inp_widths))

    logger_v.debug("module top({}, {out});".format(", ".join(inputs), out=outname))

    for iname, iwidth in inp_width.items():
        logger_v.debug(f"input {signed_prefix}[{iwidth-1}:0] {iname};")
    if out_width > 1:
        logger_v.debug(f"output [{out_width-1}:0] {outname};")
    else:
        logger_v.debug(f"output {outname};")
    logger_v.debug("// NOTE: Could use signed output here, but unsigned for now since we represent classes")
    logger_v.debug("assign {name} = " .format(name=outname))

    comp_bits = iter(comp_bits)
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            input_bits = inp_width[name]
            node_bits = next(comp_bits)
            threshold = rounding_f(tree_.threshold[node], input_bits)

            # if node_bits >= input_bits:
            logger_v.debug("{} ( {} < {}'d{} ) ?".format(indent, name, input_bits, threshold))
            # elif node_bits < input_bits:
            #     logger_v.debug("{} ({}[{}:{}] <= {})?".format(
            #         indent, name, input_bits - 1, input_bits - node_bits, threshold
            #     ))

            recurse(tree_.children_left[node], depth + 1)
            logger_v.debug("{}:".format(indent))
            recurse(tree_.children_right[node], depth + 1)
        else:
            logger_v.debug("{}{}".format(indent, np.argmax(tree_.value[node]) )) 

    recurse(0, 1)
    logger_v.debug(";")
    logger_v.debug("endmodule")

    tb_text = generate_testbench_code(
        input_width_dict=inp_width,
        sim_period=simclk_ns,
        input_separator=input_separator,
        input_tb_file=tb_inputs_file,
        output_tb_file=tb_output_file,
        output_width_dict={outname: out_width},
        signed=signed
    )
    logger_tb.debug(tb_text)


def write_tree_to_verilog(dt_model, input_bits, comparator_bits, 
                          verilog_file, tb_file, inputs_file, output_file, input_separator=' ', simclk_ms=1,
                          rounding_f=None, signed=False):
    metadata = {}

    # logger are responsible for writing verilog files
    logger_v = logging_cfg('verilog', verilog_file)
    logger_tb = logging_cfg('tb', tb_file)

    if rounding_f is None:
        rounding_f = lambda x: x  # no rounding

    if not isinstance(input_bits, (list, tuple)):
        input_bits = [input_bits] * len({feature for feature in dt_model.tree_.feature if feature != _tree.TREE_UNDEFINED})
    else:
        input_bits = [bits for i, bits in enumerate(input_bits) if dt_model.feature_importances_[i] > 0]
    
    # case where no features are selected
    if len(input_bits) == 0:
        logger.error("No features selected, so input bits are empty")
        return

    if not isinstance(comparator_bits, (list, tuple)):
        comparator_bits = [comparator_bits] * len([feature for feature in dt_model.tree_.feature if feature != _tree.TREE_UNDEFINED])

    assert len(comparator_bits) == len([feature for feature in dt_model.tree_.feature if feature != _tree.TREE_UNDEFINED]), \
        f"Length of comparator_bits {len(comparator_bits)} does not match number of features {len([feature for feature in dt_model.tree_.feature if feature != _tree.TREE_UNDEFINED])}"

    tree_to_verilog(
        tree=dt_model,
        inp_widths=list(input_bits),
        comp_bits=list(comparator_bits),
        inpprefix="X",
        outname="out",
        input_separator=input_separator,
        simclk_ns=simclk_ms * 1e6,
        tb_inputs_file=inputs_file,
        tb_output_file=output_file,
        logger_v=logger_v,
        logger_tb=logger_tb,
        metadata=metadata,
        rounding_f=rounding_f,
        signed=signed
    )
    return metadata


if __name__ ==  '__main__':
    from src import project_dir
    import os
    import pickle
    import shutil
    import re
    from glob import glob
    from collections import defaultdict
    from src.classifier import DecisionTreeClassifierWrapper
    from src.hw_templates.utils import classifier_hw_evaluation
    from src.args import AccuracyMetric
    from src.utils import convert_to_fixed_point

    def extract_input_precisions(file_path):
        precisions = []
        inside_module = False

        with open(file_path, 'r') as f:
            for line in f:
                if not inside_module:
                    # Look for the top-level module declaration
                    if re.match(r'\s*module\s+top\b', line):
                        inside_module = True
                else:
                    # Match lines with input declarations
                    match = re.search(r'input\s+signed\s+\[(\d+):(\d+)\]\s+X\d+\s*;', line)
                    if match:
                        high = int(match.group(1))
                        low = int(match.group(2))
                        width = abs(high - low) + 1
                        precisions.append(width)
                    elif re.search(r'output\b', line):
                        # Stop collecting once we reach output declaration
                        break
        return precisions

    expdir = '/home/balaskas/pestress/saved_logs/affectiveroad/final/ga_decisiontree___2025.05.07-06.54.37.645'
    dataset_name = 'affectiveroad'
    os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "logs", "flexriscv_clfs"), exist_ok=True)
    target_dir = os.path.join(project_dir, "logs", "flexriscv_clfs", f"dt_{dataset_name}")
    os.makedirs(target_dir, exist_ok=True)

    # copy the .c files into the target dir
    shutil.copy(os.path.join(project_dir, 'src', 'misc', 'dt_test.c'), os.path.join(target_dir, 'dt_test.c'))
    shutil.copy(os.path.join(project_dir, 'src', 'misc', 'dt_forward.h'), os.path.join(target_dir, 'dt_forward.h'))

    tree_index = 0
    classifier_file = os.path.join(expdir, 'classifiers', f'decisiontree_{tree_index}tree.pkl')
    x_test_file = os.path.join(expdir, 'data', f'decisiontree_{tree_index}_x_test.npy')
    y_test_file = os.path.join(expdir, 'data', f'decisiontree_{tree_index}_y_test.npy')
    x_test_txt_file = os.path.join(target_dir, 'x_test.txt')
    y_test_txt_file = os.path.join(target_dir, 'y_test.txt')
    c_file = os.path.join(target_dir, 'dt_forward.c')

    with open(classifier_file, 'rb') as f:
        tree = pickle.load(f)

    x_test = np.load(x_test_file)
    x_test_fxp = np.zeros_like(x_test)
    y_test = np.load(y_test_file)
    inp_precisions = extract_input_precisions(os.path.join(expdir, 'hw_eval', '0', 'hdl', 'top.v'))

    # convert x_test to fixed-point format
    for i, precision in enumerate(inp_precisions):
        x_test_fxp[:, i] = convert_to_fixed_point(x_test[:, i], precision, normalize=None, rescale=False, signed=False, fractional_bits=precision)
    np.savetxt(x_test_txt_file, x_test_fxp.astype(np.int16), fmt='%s')
    np.savetxt(y_test_txt_file, y_test.astype(np.int16), fmt='%d')

    # num_features = len({i for i in tree.feature if i != _tree.TREE_UNDEFINED})
    num_samples = x_test.shape[0]
    num_features = x_test.shape[1]

    with open(os.path.join(target_dir, 'dt_test.c'), 'r') as f:
        txt = f.read()
    new_lines = []
    for line in txt.splitlines():
        if '#define MAX_SAMPLES' in line:
            new_lines.append(f'#define MAX_SAMPLES {num_samples}')
        elif '#define NUM_FEATURES' in line:
            new_lines.append(f'#define NUM_FEATURES {num_features}')
        else:
            new_lines.append(line)
    with open(os.path.join(target_dir, 'dt_test.c'), 'w') as f:
        f.write('\n'.join(new_lines))

    logger_c = logging_cfg('c', c_file)
    logger_c.debug(f'// {classifier_file}')
    logger_c.debug(f'// samples: {num_samples}')
    logger_c.debug(f'// features: {num_features}')
    tree_to_c(
        tree=tree,
        inpprefix="X",
        logger_c=logger_c,
        inp_precisions=inp_precisions
    )

    # clf = DecisionTreeClassifierWrapper(accuracy_metric=AccuracyMetric.Accuracy, tune=False, train_data=None,  criterion='entropy') 
    # clf.load_weights(classifier_file)
    # clf._clf = dt_model
    # accuracy = clf.test(x_test.astype(np.float32), y_test.astype(np.float32))
    # print(f"Test accuracy: {100 * accuracy:.2f}%")

    # results, sim_accuracy = classifier_hw_evaluation(clf, (x_test, y_test), 8, 'dttest')
    # print(f"Simulation accuracy: {100 * sim_accuracy:.2f}%")
