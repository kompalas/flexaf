from copy import deepcopy
import numpy as np
from src.hw_templates.utils import logging_cfg, get_width, ConvFxp, convertCoef, \
    convertIntercepts, get_maxabs, generate_testbench_code_vectorized_mixed_precision_inputs


def write_argmax(logger_v, output_signals, output_name):
    """Writes a purely combinational argmax function using ternary operators.
    Assumes output_signals is a list of wire names corresponding to each class score.
    """
    # TODO: In case the sums/output activations have different widths, we need to
    # align them first. This is not implemented yet.

    num_classes = len(output_signals)
    width = (num_classes - 1).bit_length()  # Number of bits to encode the class index
    logger_v.debug(f"    // Argmax logic for {num_classes} classes")
    logger_v.debug(f"    wire [{width-1}:0] {output_name};")

    def generate_condition(i, rest):
        """Recursively generates comparison condition for score_i to be the max."""
        conds = [f"{output_signals[i]} >= {output_signals[j]}" for j in rest]
        return " && ".join(conds)

    # Start composing the ternary chain
    line = f"    assign {output_name} = "
    for i in range(num_classes - 1):
        rest = list(range(num_classes))
        rest.remove(i)
        condition = generate_condition(i, rest)
        line += f"({condition}) ? {width}'d{i} : "
    line += f"{width}'d{num_classes - 1};"
    logger_v.debug(line)


def write_neuron_module(logger_v, neuron_id, neuron_weights, bias, 
                        inpfxp_list, wfxp_list, bias_fxp, aligned_product_fxp, sum_fxp, out_fxp,
                        use_relu=True):
    num_inputs = len(neuron_weights)
    assert len(neuron_weights) == len(inpfxp_list) == len(wfxp_list), \
        "Input, weight, and fxp spec lengths must match"
    assert sum_fxp.int == aligned_product_fxp.int + (1 + num_inputs).bit_length(), \
        "Sum format must be larger than aligned format to account for bias and sum of products"
    assert sum_fxp.frac == aligned_product_fxp.frac, \
        "Sum format must have same frac as aligned format to avoid overflow"

    logger_v.debug(f"module neuron_{neuron_id}(")
    suffix = '' if use_relu and out_fxp.s == 0 else ' signed'
    for i in range(num_inputs):
        width = inpfxp_list[i].width
        logger_v.debug(f"    input  [{width-1}:0] in_{i},")
    logger_v.debug(f"    output{suffix} [{out_fxp.width-1}:0] out")
    logger_v.debug(");")
    logger_v.debug(f"    // Aligned product: {aligned_product_fxp.int} int, {aligned_product_fxp.frac} frac")
    logger_v.debug(f"    // Sum: {sum_fxp.int} int, {sum_fxp.frac} frac")

    sum_terms = []
    for i in range(num_inputs):
        inpfxp = inpfxp_list[i]
        wfxp = wfxp_list[i]
        product_fxp = ConvFxp(1, inpfxp.int + wfxp.int, inpfxp.frac + wfxp.frac)
        shift_by = aligned_product_fxp.frac - product_fxp.frac
        extend_by = aligned_product_fxp.width - product_fxp.width - shift_by
        w_width = wfxp.width
        weight = neuron_weights[i]
        weight_bin = np.binary_repr(weight, w_width)
        logger_v.debug(f"    // Weight {i}: {weight} -> {w_width}'sb{weight_bin}")

        logger_v.debug(f"    wire signed [{product_fxp.width-1}:0] product_{i};")
        logger_v.debug(f"    assign product_{i} = $signed({{1'b0, in_{i}}}) * {w_width}'sb{weight_bin};")

        aligned_wire = f"aligned_{i}"
        logger_v.debug(f"    wire signed [{aligned_product_fxp.width-1}:0] {aligned_wire};")
        if extend_by > 0 and shift_by > 0:
            logger_v.debug(f"    assign {aligned_wire} = $signed({{{{{extend_by}{{product_{i}[{product_fxp.width-1}]}}, product_{i}}}}}) <<< {shift_by};")
        elif extend_by > 0:
            logger_v.debug(f"    assign {aligned_wire} = $signed({{{{{extend_by}{{product_{i}[{product_fxp.width-1}]}}, product_{i}}}}});")
        elif shift_by > 0:
            logger_v.debug(f"    assign {aligned_wire} = product_{i} <<< {shift_by};")
        else:
            logger_v.debug(f"    assign {aligned_wire} = product_{i};")
        sum_terms.append(aligned_wire)

    # Bias: shift to match the fractional bits, and sign-extend to match the aligned product width
    bias_bin = np.binary_repr(bias, bias_fxp.width)
    shift_by = aligned_product_fxp.frac - bias_fxp.frac
    extend_by = aligned_product_fxp.width - bias_fxp.width - shift_by

    logger_v.debug(f"    // Bias: {bias} -> {bias_fxp.width}'sb{bias_bin}")
    logger_v.debug(f"    wire signed [{bias_fxp.width-1}:0] bias;")
    logger_v.debug(f"    assign bias = {bias_fxp.width}'sb{bias_bin};")
    logger_v.debug(f"    wire signed [{bias_fxp.width + shift_by - 1}:0] bias_shifted;")
    if shift_by > 0:
        logger_v.debug(f"    assign bias_shifted = bias <<< {shift_by};")
    else:
        logger_v.debug(f"    assign bias_shifted = bias;")
    logger_v.debug(f"    wire signed [{aligned_product_fxp.width-1}:0] bias_aligned;")
    if extend_by > 0:
        logger_v.debug(f"    assign bias_aligned = {{{{{extend_by}{{bias_shifted[{bias_fxp.width + shift_by - 1}]}}}}, bias_shifted}};")
    else:
        logger_v.debug(f"    assign bias_aligned = bias_shifted;")

    sum_expr = ' + '.join(['bias_aligned'] + sum_terms)
    logger_v.debug(f"    wire signed [{sum_fxp.width-1}:0] sum;")
    logger_v.debug(f"    assign sum = {sum_expr};")

    # Output with activation
    if use_relu:
        logger_v.debug(f"    assign out = (sum < 0) ? {out_fxp.width}'d0 : sum[{out_fxp.width-1}:0];")
    else:
        logger_v.debug(f"    assign out = sum[{out_fxp.width-1}:0];")
    logger_v.debug("endmodule\n")


def mlp_to_verilog(
    mlp_model, input_widths, weight_width, input_name, output_name, logger_v, metadata
):
    num_inputs = mlp_model.coefs_[0].shape[0]
    if isinstance(input_widths, (tuple, list)):
        assert len(input_widths) == num_inputs, "Mismatch between input widths and model input features"
    else:
        input_widths = [input_widths] * num_inputs
    metadata["num_inputs"] = num_inputs
    metadata["input_widths"] = input_widths
    metadata["num_outputs"] = mlp_model.coefs_[-1].shape[1]
    metadata["num_classes"] = len(mlp_model.classes_)

    # Step 1: Create input fixed-point specs
    inpfxp_list = [ConvFxp(0, 0, width) for width in input_widths]

    # Step 2: Define a common weight format for all weights
    max_weight = max(abs(w) for layer in mlp_model.coefs_ for row in layer for w in row)
    w_int = get_width(max_weight)
    w_frac = weight_width - 1 - w_int
    wfxp_template = ConvFxp(1, w_int, w_frac)

    # Step 3: Create bias formats per layer
    bfxp_list = []
    for i, bias_vec in enumerate(mlp_model.intercepts_):
        max_bias = max(abs(b) for b in bias_vec)
        b_int = get_width(max_bias)
        bias_fxp = ConvFxp(1, b_int, wfxp_template.width - b_int - 1)
        bfxp_list.append(bias_fxp)

    # Step 4: Convert weights and biases to fixed-point
    weights = convertCoef(mlp_model.coefs_, wfxp_template, False)
    biases = convertIntercepts(mlp_model.intercepts_, bfxp_list, False)

    logger_v.debug(f"// Input widths: {input_widths}")
    logger_v.debug(f"// Weights: {weights}")
    logger_v.debug(f"// Biases: {biases}")

    # Step 5: Generate neuron modules
    activation_signals = []
    activation_fxp = []
    for layer_index, (layer_weights, layer_biases) in enumerate(zip(weights, biases)):

        # Use previous layer activation format for inputs
        if layer_index == 0:
            neuron_inputs_fxp = deepcopy(inpfxp_list)
        else:
            # neuron_inputs_fxp = [ConvFxp(0, 0, w) for w in layer_activation_widths]
            neuron_inputs_fxp = deepcopy(layer_activation_fxp)

        layer_outputs = []
        layer_activation_fxp = []
        bias_fxp = bfxp_list[layer_index]
        is_output_layer = layer_index == len(weights) - 1

        for neuron_index, (neuron_weights, neuron_bias) in enumerate(zip(layer_weights, layer_biases)):
            neuron_id = f"l{layer_index}_n{neuron_index}"

            # Weights all share same fixed-point template for now
            neuron_weights_fxp = [wfxp_template for _ in neuron_weights]

            # Compute accumulator format: sum of input_i * weight_i formats
            max_int = max([inp.int + wfxp_template.int for inp in neuron_inputs_fxp] + [bias_fxp.int])
            max_frac = max([inp.frac + wfxp_template.frac for inp in neuron_inputs_fxp] + [bias_fxp.frac])
            aligned_product_fxp = ConvFxp(1, max_int, max_frac)
            additional_sum_bits = (1 + len(neuron_weights)).bit_length()  # accounting for extra bits for the sum, including bias
            sum_fxp = ConvFxp(1, max_int + additional_sum_bits, max_frac)
            sign = 1 if is_output_layer else 0
            out_fxp = ConvFxp(sign, sum_fxp.int, sum_fxp.frac)

            # Generate neuron module
            write_neuron_module(
                logger_v=logger_v,
                neuron_id=neuron_id,
                neuron_weights=neuron_weights,
                bias=neuron_bias,
                inpfxp_list=neuron_inputs_fxp,
                wfxp_list=neuron_weights_fxp,
                bias_fxp=bias_fxp,
                aligned_product_fxp=aligned_product_fxp,
                sum_fxp=sum_fxp,
                out_fxp=out_fxp,
                use_relu=mlp_model.activation == 'relu' and not is_output_layer
            )
            layer_outputs.append(f"{neuron_id}_out")
            layer_activation_fxp.append(out_fxp)

        activation_fxp.append(layer_activation_fxp)
        activation_signals.append(layer_outputs)

    # Step 6: Generate top module
    total_input_width = sum(input_widths)
    output_width = get_width(len(activation_signals[-1]))
    metadata["output_width"] = output_width
    logger_v.debug(f"module top (input [{total_input_width-1}:0] {input_name}, output [{output_width-1}:0] {output_name});")

    # Slice input vector
    current = 0
    for i, width in enumerate(input_widths):
        logger_v.debug(f"    wire [{width-1}:0] input_{i} = {input_name}[{current + width - 1}:{current}];")
        current += width

    # Instantiate neurons layer by layer
    for layer_index, (layer, layer_activation_fxp) in enumerate(zip(activation_signals, activation_fxp)):
        for neuron_index, (neuron_out, act_fxp) in enumerate(zip(layer, layer_activation_fxp)):
            neuron_id = f"l{layer_index}_n{neuron_index}"
            if layer_index == 0:
                inputs = [f"input_{i}" for i in range(num_inputs)]
            else:
                inputs = activation_signals[layer_index - 1]
            is_output_layer = layer_index == len(activation_signals) - 1
            suffix = ' signed' if is_output_layer else ''
            logger_v.debug(f"    wire{suffix} [{act_fxp.width-1}:0] {neuron_out};")
            logger_v.debug(f"    neuron_{neuron_id} neuron_{neuron_id}_inst ({', '.join(inputs)}, {neuron_out});")

    # Step 7: Decision function
    if output_width > 1:
        write_argmax(logger_v, activation_signals[-1], output_name)
    else:
        logger_v.debug(f"    assign {output_name} = ( {activation_signals[-1][0]} > 0 ) ? 1'd1 : 1'd0;")
    logger_v.debug("endmodule\n")



def write_mlp_to_verilog(mlp_model, input_bits, weight_bits, verilog_file, tb_file, 
                         inputs_file, output_file, input_separator=' ', simclk_ms=1):
    metadata = {}
    # logger are responsible for writing verilog files
    logger_v = logging_cfg('verilog', verilog_file)
    logger_tb = logging_cfg('tb', tb_file)

    mlp_to_verilog(
        mlp_model=mlp_model,
        input_widths=input_bits,
        weight_width=weight_bits,
        input_name='inp',
        output_name='out',
        logger_v=logger_v,
        metadata=metadata
    )
    tb_text = generate_testbench_code_vectorized_mixed_precision_inputs(
        output_width=metadata['output_width'],
        input_count=metadata['num_inputs'],
        input_widths=metadata['input_widths'],
        input_name='inp',
        output_name='out',
        simclk_ns=simclk_ms * 1e6,
        inputs_file=inputs_file,
        output_file=output_file,
        input_separator=input_separator,
        signed=False
    )
    logger_tb.debug(tb_text)

    return metadata


if __name__ == "__main__":
    from src import project_dir
    import os
    import pickle
    import re
    from src.utils import convert_to_fixed_point

    def extract_input_precisions(verilog_file_path):
        with open(verilog_file_path, 'r') as f:
            lines = f.readlines()

        in_module = False
        input_precisions = []

        for line in lines:
            if not in_module:
                # Look for the first neuron module
                if re.match(r'\s*module\s+neuron_', line):
                    in_module = True
            elif in_module:
                # If we've found the first neuron module, extract input lines
                input_match = re.search(r'input\s+\[(\d+):(\d+)\]\s+in_\d+', line)
                if input_match:
                    high = int(input_match.group(1))
                    low = int(input_match.group(2))
                    width = abs(high - low) + 1
                    input_precisions.append(width)
                # Stop if we reach the output or end of port list
                if re.search(r'\boutput\b', line) or re.search(r'\);', line):
                    break

        return input_precisions

    expdir = '/home/balaskas/pestress/saved_logs/drivedb/final/ga_mlp___2025.06.03-14.50.20.144'
    dataset_name = 'drivedb'
    model_index = 0
    weight_precision = 8
    override_input_precisions = True  # Set to True to use fixed precision for all inputs to 4 bits

    assert dataset_name.split('_')[0] in expdir, f"Dataset name {dataset_name} does not match experiment directory {expdir}"
    os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "logs", "flexriscv_clfs"), exist_ok=True)
    target_dir = os.path.join(project_dir, "logs", "flexriscv_clfs", f"mlp_{dataset_name}")
    os.makedirs(target_dir, exist_ok=True)

    weights_file1 = os.path.join(expdir, 'classifiers', f'mlp_{model_index}coefs0.npy')
    weights_file2 = os.path.join(expdir, 'classifiers', f'mlp_{model_index}coefs1.npy')
    bias_file1 = os.path.join(expdir, 'classifiers', f'mlp_{model_index}intercepts0.npy')
    bias_file2 = os.path.join(expdir, 'classifiers', f'mlp_{model_index}intercepts1.npy')
    x_test_file = os.path.join(expdir, 'data', f'mlp_{model_index}_x_test.npy')
    y_test_file = os.path.join(expdir, 'data', f'mlp_{model_index}_y_test.npy')
    x_test_txt_file = os.path.join(target_dir, 'x_test.txt')
    y_test_txt_file = os.path.join(target_dir, 'y_test.txt')
    weights_file1_txt = os.path.join(target_dir, 'weights_layer1.txt')
    weights_file2_txt = os.path.join(target_dir, 'weights_layer2.txt')
    bias_file1_txt = os.path.join(target_dir, 'bias_layer1.txt')
    bias_file2_txt = os.path.join(target_dir, 'bias_layer2.txt')

    x_test = np.load(x_test_file)
    x_test_fxp = np.zeros_like(x_test)
    y_test = np.load(y_test_file)
    inp_precisions = extract_input_precisions(os.path.join(expdir, 'hw_eval', '0', 'hdl', 'top.v'))
    if override_input_precisions:
        inp_precisions = [4] * x_test.shape[1]  # Override with fixed precision
    assert len(inp_precisions) == x_test.shape[1], f"Input precisions {inp_precisions} do not match number of features {x_test.shape[1]}"
    # convert x_test to fixed-point format
    for i, precision in enumerate(inp_precisions):
        x_test_fxp[:, i] = convert_to_fixed_point(x_test[:, i], precision, normalize=None, rescale=False, signed=False, fractional_bits=precision)

    np.savetxt(x_test_txt_file, x_test_fxp.astype(np.int16), fmt='%s')
    np.savetxt(y_test_txt_file, y_test.astype(np.int16), fmt='%d')

    # save weights and biases as txt
    weights1 = np.load(weights_file1)
    weights2 = np.load(weights_file2)
    bias1 = np.load(bias_file1)
    bias2 = np.load(bias_file2)
    # convert to fixed-point format
    weights1_fxp = convert_to_fixed_point(weights1, weight_precision, normalize=None, rescale=False, signed=True, fractional_bits=weight_precision - 1)
    weights2_fxp = convert_to_fixed_point(weights2, weight_precision, normalize=None, rescale=False, signed=True, fractional_bits=weight_precision - 1)
    bias1_fxp = convert_to_fixed_point(bias1, weight_precision, normalize=None, rescale=False, signed=True, fractional_bits=weight_precision - 1)
    bias2_fxp = convert_to_fixed_point(bias2, weight_precision, normalize=None, rescale=False, signed=True, fractional_bits=weight_precision - 1)
    np.savetxt(weights_file1_txt, weights1_fxp.astype(np.int16), fmt='%s')
    np.savetxt(weights_file2_txt, weights2_fxp.astype(np.int16), fmt='%s')
    np.savetxt(bias_file1_txt, bias1_fxp.astype(np.int16), fmt='%s')
    np.savetxt(bias_file2_txt, bias2_fxp.astype(np.int16), fmt='%s')