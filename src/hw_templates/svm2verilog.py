import numpy as np
from src.hw_templates.utils import get_width, get_maxabs, ConvFxp, convert_params_to_fxp, logging_cfg, \
    generate_testbench_code_vectorized_mixed_precision_inputs, one_hot_encoder, min_binary_digits, \
    create_paths
from src.utils import convert_to_fixed_point

#### SVM to Verilog conversion functions - flo implementation


def write_classifier_ovo(logger_v, prefix, b, nweights, act, w_width, pwidth, swidth):
    nsum = str(b)
    swidth += len(nweights)
    sumname = f"{prefix}_sum"
    for i, w in enumerate(nweights):
        a = act[i]
        name = f"{prefix}_po_{i}"
        nsum += f" + {name}"
        logger_v.debug(f"    wire signed [{pwidth-1}:0] {name};")
        bin_w = f"{w_width}'sb{np.binary_repr(w, w_width)}"
        logger_v.debug(f"    //weight {w}: {bin_w}")
        # logger_v.debug(f"    assign {name} = $signed({{1'b0, {a}}}) * {bin_w};\n")
        logger_v.debug(f"    assign {name} = $signed({a}) * {bin_w};\n")

    logger_v.debug(f"    wire signed [{swidth-1}:0] {sumname};")
    logger_v.debug(f"    assign {sumname} = {nsum};")
    logger_v.debug(f"    wire {prefix};")
    logger_v.debug(f"    assign {prefix} = {sumname}[{swidth-1}];")


def write_decision_matrix_ovo(logger_v, prefix, act_next, num_classes):
    logger_v.debug(f"// decisionMatrix inp: {', '.join(act_next)}")
    swidth = get_width(num_classes)
    ind = 0
    for m in range(num_classes):
        for n in range(num_classes):
            if m < n:
                name_mn = f"{prefix}_cmp_{m}_{n}"
                name_nm = f"{prefix}_cmp_{n}_{m}"
                logger_v.debug(f"    wire {name_mn}, {name_nm};")
                logger_v.debug(f"    assign {name_mn} = ~{act_next[ind]};")
                logger_v.debug(f"    assign {name_nm} = {act_next[ind]};\n")
                ind += 1

    dmlst = []
    for m in range(num_classes):
        sum_m = f"{prefix}_sum_{m}"
        dmlst.append(sum_m)
        sumprefix = f"{prefix}_cmp_{m}_"
        sum_row = ' + '.join([f"{sumprefix}{n}" for n in range(num_classes) if n != m])
        logger_v.debug(f"    wire [{swidth-1}:0] {sum_m};")
        logger_v.debug(f"    assign {sum_m} = {sum_row};")
    logger_v.debug("")

    return dmlst


def write_argmax_ovo(logger_v, prefix, act, vwidth, iwidth):
    lvl = 0
    vallist = list(act)
    logger_v.debug(f"// argmax inp: {', '.join(vallist)}")
    idxlist = [f"{iwidth}'b{np.binary_repr(i, iwidth)}" for i in range(len(act))]

    while len(vallist) > 1:
        newV = []
        newI = []
        logger_v.debug(f"    //comp level {lvl}")
        for i in range(0, len(vallist)-1, 2):
            cmpname = f"cmp_{lvl}_{i}"
            vname = f"{prefix}_val_{lvl}_{i}"
            iname = f"{prefix}_idx_{lvl}_{i}"
            vname1, vname2 = vallist[i], vallist[i+1]
            iname1, iname2 = idxlist[i], idxlist[i+1]
            logger_v.debug(f"    wire {cmpname};")
            logger_v.debug(f"    wire [{vwidth-1}:0] {vname};")
            logger_v.debug(f"    wire [{iwidth-1}:0] {iname};")
            logger_v.debug(f"    assign {cmpname} = ({vname1} >= {vname2});")
            logger_v.debug(f"    assign {vname} = ({cmpname}) ? {vname1} : {vname2};")
            logger_v.debug(f"    assign {iname} = ({cmpname}) ? {iname1} : {iname2};\n")
            newV.append(vname)
            newI.append(iname)

        if len(vallist) % 2 == 1:
            newV.append(vallist[-1])
            newI.append(idxlist[-1])

        lvl += 1
        vallist = list(newV)
        idxlist = list(newI)

    return idxlist[-1]


def svm_to_verilog_ovo_backup(svm_model, inp_width, w_width, num_classes, input_name, output_name, logger_v, metadata):

    inpfxp = ConvFxp(0, 0, inp_width)
    w_int = min(get_width(get_maxabs(svm_model.coef_)), w_width - 1)
    w_frac = w_width - 1 - w_int
    wfxp = ConvFxp(1, w_int, w_frac)
    b_int = get_width(get_maxabs(svm_model.intercept_))
    b_frac = w_width - 1 - b_int
    bfxp = ConvFxp(1, b_int, b_frac)
    # bfxp = ConvFxp(1, b_int, inpfxp.frac + wfxp.frac)

    coefficients = convert_params_to_fxp(svm_model.coef_, wfxp, False)
    intercepts = convert_params_to_fxp(svm_model.intercept_, bfxp, False)

    activations_width = [inp_width]
    num_inputs = len(coefficients[0])
    num_outputs = num_classes
    metadata['num_inputs'] = num_inputs

    maxInp = (1 << activations_width[0]) - 1
    maxsum = 0
    minsum = 0
    for i, neuron in enumerate(coefficients):
        pos = sum(w for w in neuron if w > 0) * maxInp + intercepts[i]
        neg = sum(w for w in neuron if w < 0) * maxInp + intercepts[i]
        maxsum = max(maxsum, pos)
        minsum = min(minsum, neg)

    size = max(1 + get_width(maxsum), 1 + get_width(minsum))
    size = max(size, activations_width[0] + w_width)
    sum_width = size
    activations_width.append(size)

    output_width = get_width(num_classes)
    if num_classes == 2:
        output_width = 1
    metadata['output_width'] = output_width
    metadata['num_outputs'] = num_outputs

    logger_v.debug(f"//weights: {coefficients}")
    logger_v.debug(f"//intercepts: {intercepts}")
    logger_v.debug(f"//act size: {activations_width}")
    logger_v.debug(f"//sum size: {sum_width}")
    logger_v.debug(f"//pred num: {num_outputs}")

    logger_v.debug(f"module top ({input_name}, {output_name});")
    logger_v.debug(f"input [{num_inputs * inp_width - 1}:0] {input_name};")
    logger_v.debug(f"output [{output_width - 1}:0] {output_name};\n")

    act_next = [f"{input_name}[{(i+1)*inp_width-1}:{i*inp_width}]" for i in range(num_inputs)]
    act = list(act_next)
    act_next = []

    for i in range(len(coefficients)):
        logger_v.debug(f"// classifier: {i}")
        prefix = f"n_0_{i}"
        nweights = coefficients[i]
        pwidth = inp_width + w_width
        bias = intercepts[i]
        write_classifier_ovo(logger_v, prefix, bias, nweights, act, w_width, pwidth, sum_width)
        act_next.append(prefix)
        logger_v.debug("")

    if num_classes > 2:
        prefix = "dm"
        dm = write_decision_matrix_ovo(logger_v, prefix, act_next, num_classes)
        vw = get_width(num_classes)
        iw = get_width(num_classes)
        prefix = "argmax"
        logger_v.debug(f"// argmax: {num_classes} classes, need {iw} bits")
        out = write_argmax_ovo(logger_v, prefix, dm, vw, iw)
        logger_v.debug(f"    assign {output_name} = {out};")
    else:
        prefix = "argmax"
        logger_v.debug(f"// argmax: {num_classes} classes, need 1 bit")
        logger_v.debug(f"    assign {output_name} = ({act_next[0]} > 0) ? 1'b0 : 1'b1;")

    logger_v.debug("endmodule")


#### SVM to Verilog conversion functions - Students' implementation


def svm_to_verilog_ovo(svm_model, inp_widths, w_widths, input_name, output_name, logger_v, metadata, use_always_block=True):

    # setup for mixed precision in inputs/features
    num_features = svm_model.coef_.shape[1]
    if isinstance(inp_widths, (tuple, list)):
        assert num_features == len(inp_widths), "Mismatch between number of features and input widths"
    else:
        inp_widths = [inp_widths] * num_features
    inpfxp = [ConvFxp(0, 0, inp_width) for inp_width in inp_widths]
    total_input_width = sum(inp_widths)
    metadata['num_inputs'] = num_features
    metadata['input_widths'] = inp_widths

    # setup for mixed precision in weights
    num_svms = num_classes = svm_model.coef_.shape[0]
    if isinstance(w_widths, (tuple, list)):
        assert num_svms == len(w_widths), "Mismatch between number of SVMs and weight widths"
    else:
        w_widths = [w_widths] * num_svms

    # configure weights and convert to fixed point
    weights = np.zeros_like(svm_model.coef_)
    wfxp = []
    for i in range(num_svms):
        w_width = w_widths[i]
        w_int = min(get_width(get_maxabs(svm_model.coef_[i])), w_width - 1)
        this_wfxp = ConvFxp(1, w_int, w_width - 1 - w_int)
        wfxp.append(this_wfxp)
        weights[i] = convert_params_to_fxp(svm_model.coef_[i], this_wfxp, False)
    weights = weights.astype(np.int32)

    # convert biases to fixed point
    bias_int_bits = get_width(get_maxabs(svm_model.intercept_))
    bias_fxp = ConvFxp(1, bias_int_bits, max(w_widths) - 1 - bias_int_bits)
    biases = convert_params_to_fxp(svm_model.intercept_, bias_fxp, False)

    feature_bits = get_width(num_features)
    classes_binary_digits = min_binary_digits(num_classes)
    metadata['output_width'] = classes_binary_digits

    # define the inputs and outputs of the module
    logger_v.debug("module top (")
    logger_v.debug(f"\tinput  [{total_input_width - 1} : 0] {input_name},")
    if use_always_block:
        logger_v.debug(f"\toutput reg [{classes_binary_digits - 1} : 0] {output_name}")  # in case you use the alwasy block for the decision, in the end
    else:
        logger_v.debug(f"\toutput [{classes_binary_digits - 1} : 0] {output_name}")
    logger_v.debug(");\n")
    logger_v.debug(f"\twire [{num_svms - 1}:0] decision;\n")

    for i in range(num_svms):
        logger_v.debug(f"// Classifier {i}")
        input_start = 0
        w_width = w_widths[i]
        this_wfxp = wfxp[i]
        bias = biases[i]

        sum_str = str(bias)
        for j in range(num_features):
            wire_name = f"n_{i}_po_{j}"
            inp_width = inp_widths[j]
            weight = weights[i, j]

            input_end = input_start + inp_width - 1
            logger_v.debug(f"\t// weight: {weight} -> {w_width}'sb{np.binary_repr(weight, w_width)}")
            logger_v.debug(f"\twire signed [{w_width + inp_width - 1} : 0] {wire_name};")
            logger_v.debug(f"\tassign {wire_name} = $signed({input_name}[{input_end} : {input_start}]) * {w_width}'sb{np.binary_repr(weight, w_width)};")
            sum_str += f" + {wire_name}"
            input_start = input_end + 1

        wire_name_sum = f"n_{i}_sum"
        logger_v.debug(f"\n\t// bias: {bias} -> {np.binary_repr(bias, bias_fxp.get_width())}")
        logger_v.debug(f"\twire signed [{w_width + inp_width + feature_bits - 1} : 0] {wire_name_sum};")
        logger_v.debug(f"\tassign {wire_name_sum} = {sum_str};")
        logger_v.debug(f"\tassign decision[{i}] = {wire_name_sum}[{w_width + inp_width + feature_bits - 1}];\n")

    decision = create_paths(num_classes)

    if use_always_block:
        # generate the assign statement for the output using always @(*)
        # this genertates commented verilog code for now
        logger_v.debug("always @(*) begin")
        logger_v.debug("\tcasex (decision)")
        for cl, paths in decision.items():
            for path in paths:
                logger_v.debug(f"\t\t{num_svms}'b{path}: {output_name} <= {classes_binary_digits}'b{bin(cl)[2:].zfill(classes_binary_digits)};")

        logger_v.debug(f"\t\tdefault: {output_name} <= {classes_binary_digits}'b{classes_binary_digits * 'z'};")
        logger_v.debug("\tendcase")
        logger_v.debug(" end\n")

    else:
        # generate the assign statement for the output using the tri-conditional operator
        assign_line = f"assign {output_name} = "
        conditions = []
        for cl, paths in decision.items():
            for path in paths:
                condition = f"(decision == {num_svms}'b{path}) ? {classes_binary_digits}'b{bin(cl)[2:].zfill(classes_binary_digits)}"
                conditions.append(condition)
        default_case = f"{classes_binary_digits}'b{classes_binary_digits * '1'}"
        assign_expression = " :\n\t\t".join(conditions) + f" :\n\t\t{default_case};  // default case to all 1's"
        logger_v.debug(assign_line + assign_expression)

    logger_v.debug("endmodule")


def svm_to_verilog_ovr(svm_model, inp_widths, w_widths, input_name, output_name, logger_v, metadata):
    # setup for mixed precision in inputs/features
    num_features = svm_model.coef_.shape[1]
    if isinstance(inp_widths, (tuple, list)):
        assert num_features == len(inp_widths), "Mismatch between number of features and input widths"
    else:
        inp_widths = [inp_widths] * num_features
    inpfxp = [ConvFxp(0, 0, inp_width) for inp_width in inp_widths]
    total_input_width = sum(inp_widths)
    metadata['num_inputs'] = num_features
    metadata['input_widths'] = inp_widths

    # setup for mixed precision in weights
    num_svms = num_classes = svm_model.coef_.shape[0]
    if isinstance(w_widths, (tuple, list)):
        assert num_svms == len(w_widths), "Mismatch between number of SVMs and weight widths"
    else:
        w_widths = [w_widths] * num_svms

    additional_sum_bits = (1 + num_features).bit_length()  # accounting for the bias
    classes_binary_digits = num_classes.bit_length()
    metadata['output_width'] = classes_binary_digits

    # define the inputs and outputs of the module
    logger_v.debug("module top (")
    logger_v.debug(f"\tinput  [{total_input_width - 1} : 0] {input_name},")
    logger_v.debug(f"\toutput [{classes_binary_digits - 1} : 0] {output_name}")
    logger_v.debug(");\n")

    all_sum_fxps = []
    for i in range(num_svms):
        logger_v.debug(f"// Classifier {i}")
        input_start = 0
        w_width = w_widths[i]
        w_int = min(get_width(get_maxabs(svm_model.coef_[i])), w_width - 1)
        w_frac = w_width - 1 - w_int
        this_wfxp = ConvFxp(1, w_int, w_frac)
        weights = convert_to_fixed_point(svm_model.coef_[i], precision=this_wfxp.get_width(), signed=True, rescale=False, fractional_bits=this_wfxp.frac)

        bias_int_bits = get_width(get_maxabs(svm_model.intercept_[i]))
        bias_int_bits = min(bias_int_bits, w_width - 1)
        bias_fxp = ConvFxp(1, bias_int_bits, w_width - 1 - bias_int_bits)
        bias = bias_fxp.to_fixed(svm_model.intercept_[i])

        max_int_bits = max([bias_fxp.int] + [inpfxp[j].int + this_wfxp.int for j in range(num_features)])
        max_fractional_bits = max([bias_fxp.frac] + [inpfxp[j].frac + this_wfxp.frac for j in range(num_features)])
        aligned_product_fxp = ConvFxp(1, max_int_bits, max_fractional_bits)
        logger_v.debug(f"// aligned product: integer bits {aligned_product_fxp.int}, fractional bits: {aligned_product_fxp.frac}")
        sum_fxp = ConvFxp(1, max_int_bits + additional_sum_bits, max_fractional_bits)
        logger_v.debug(f"// final sum: integer bits: {sum_fxp.int}, fractional bits: {sum_fxp.frac}")
        all_sum_fxps.append(sum_fxp)

        sum_str = f"bias_{i}_aligned"
        for j in range(num_features):
            wire_name = f"n_{i}_po_{j}"
            inp_width = inp_widths[j]
            this_inpfxp = inpfxp[j]
            partial_product_fxp = ConvFxp(1, this_inpfxp.int + this_wfxp.int, this_inpfxp.frac + this_wfxp.frac)
            weight = weights[j]

            input_end = input_start + inp_width - 1
            logger_v.debug(f"\t// input: integer bits {this_inpfxp.int}, fractional bits: {this_inpfxp.frac}")
            logger_v.debug(f"\t// weight: integer bits {this_wfxp.int}, fractional bits: {this_wfxp.frac}")
            logger_v.debug(f"\t// partial product: integer bits {partial_product_fxp.int}, fractional bits: {partial_product_fxp.frac}")
            logger_v.debug(f"\t// weight: {weight} -> {w_width}'sb{np.binary_repr(weight, w_width)}")
            logger_v.debug(f"\twire signed [{partial_product_fxp.get_width() - 1} : 0] {wire_name};")
            logger_v.debug(f"\tassign {wire_name} = $signed({{1'b0, {input_name}[{input_end} : {input_start}]}}) * {w_width}'sb{np.binary_repr(weight, w_width)};")
            logger_v.debug(f"\twire signed [{aligned_product_fxp.get_width() - 1} : 0] {wire_name}_aligned;")
            input_start = input_end + 1

            # Compute how many bits to shift left to match max fractional bits
            shift_by = max_fractional_bits - partial_product_fxp.frac
            # Compute how many bits to sign-extend (on the left) to reach the total aligned width
            extend_by = aligned_product_fxp.get_width() - partial_product_fxp.get_width() - shift_by

            expr = wire_name
            if shift_by > 0:
                # expr = f"{{{expr}, {shift_by}'b{'0' * shift_by}}}"
                expr = f"{wire_name} <<< {shift_by}" 
                if extend_by > 0:
                    expr = '$signed({{' + str(extend_by) + '{' + wire_name + '[' + str(partial_product_fxp.get_width() - 1) + ']}}, ' + wire_name + '}) << ' + str(shift_by)
            elif extend_by > 0:
                expr = '$signed({{' + str(extend_by) + '{' + wire_name + '[' + str(partial_product_fxp.get_width() - 1) + ']}}, ' + wire_name + '})'
            logger_v.debug(f"\tassign {wire_name}_aligned = {expr};")
            sum_str += f" + {wire_name}_aligned"

        # handle the bias
        logger_v.debug(f"\n\t// bias: {bias} -> {np.binary_repr(bias, bias_fxp.get_width())}")
        logger_v.debug(f"\t// bias: integer bits {bias_fxp.int}, fractional bits: {bias_fxp.frac}")
        logger_v.debug(f"\twire signed [{bias_fxp.get_width() - 1} : 0] bias_{i};")
        logger_v.debug(f"\tassign bias_{i} = {bias_fxp.get_width()}'sb{np.binary_repr(bias, bias_fxp.get_width())};")

        # Compute how many bits to shift left to match max fractional bits
        shift_by = max_fractional_bits - bias_fxp.frac
        logger_v.debug(f"\twire signed [{bias_fxp.get_width() + shift_by - 1} : 0] bias_{i}_shifted;")
        if shift_by > 0:
            logger_v.debug(f"\tassign bias_{i}_shifted = bias_{i} <<< {shift_by};")
        else:
            logger_v.debug(f"\tassign bias_{i}_shifted = bias_{i};")

        # Compute how many bits to sign-extend (on the left) to reach the total aligned width
        logger_v.debug(f"\twire signed [{aligned_product_fxp.get_width() - 1} : 0] bias_{i}_aligned;")
        extend_by = aligned_product_fxp.get_width() - bias_fxp.get_width() - shift_by
        if extend_by > 0:
            logger_v.debug(f"\tassign bias_{i}_aligned = {{{{{extend_by}{{bias_{i}_shifted[{bias_fxp.get_width() + shift_by - 1}]}}}}, bias_{i}_shifted}};")
        else:
            logger_v.debug(f"\tassign bias_{i}_aligned = bias_{i}_shifted;")

        # write the total sum of this classifier
        wire_name_sum = f"n_{i}_sum"
        logger_v.debug(f"\n\twire signed [{sum_fxp.get_width() - 1} : 0] {wire_name_sum};")
        logger_v.debug(f"\tassign {wire_name_sum} = {sum_str};\n")

    # align the sums to the same fixed point format
    max_sum_bits = max([sum_fxp.get_width() for sum_fxp in all_sum_fxps])
    max_frac_bits = max([sum_fxp.frac for sum_fxp in all_sum_fxps])
    logger_v.debug(f"\t// aligned sum: integer bits {max_sum_bits - max_frac_bits - 1}, fractional bits: {max_frac_bits}")
    for i in range(num_svms):
        logger_v.debug(f"\twire signed [{max_sum_bits - 1} : 0] n_{i}_sum_aligned;")
        if max_frac_bits > all_sum_fxps[i].frac:
            logger_v.debug(f"\tassign n_{i}_sum_aligned = n_{i}_sum <<< {max_frac_bits - all_sum_fxps[i].frac};")
        else:
            logger_v.debug(f"\tassign n_{i}_sum_aligned = n_{i}_sum;")

    # write the decision logic
    logger_v.debug(f"\nassign {output_name} =")
    if num_svms == 1:
        # in case of binary classification in OvR, only one classifier is created, and the sign of the weighted sum determines the class
        logger_v.debug(f"\t(n_0_sum_aligned > 0) ? {classes_binary_digits}'d1 : {classes_binary_digits}'b0;\n")
    else:
        for i in range(num_classes - 1):
            cond = " && ".join(
                [f"(n_{i}_sum_aligned > n_{j}_sum_aligned)" for j in range(num_classes) if j != i]
            )
            logger_v.debug(f"\t{cond} ? {classes_binary_digits}'d{i} :")
        # Default case (last class wins if no other is greater)
        logger_v.debug(f"\t{classes_binary_digits}'d{num_classes - 1};\n")
    logger_v.debug("endmodule")


def write_svm_to_verilog(svm_model, input_bits, weight_bits, num_classes, verilog_file, tb_file, 
                         inputs_file, output_file, input_separator=' ', simclk_ms=1, is_ovo=False):
    metadata = {}
    logger_v = logging_cfg('verilog', verilog_file)
    logger_tb = logging_cfg('tb', tb_file)

    svm_to_verilog_f = svm_to_verilog_ovo if is_ovo else svm_to_verilog_ovr
    svm_to_verilog_f(svm_model=svm_model,
                     inp_widths=input_bits,
                     w_widths=weight_bits,
                     input_name='inp',
                     output_name='out',
                     logger_v=logger_v,
                     metadata=metadata)

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
