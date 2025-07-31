import logging
import numpy as np
import re
import math
import os.path
import logging
from copy import deepcopy
from sklearn.metrics import accuracy_score
from src.hw_templates.eda import execute_hw_evaluation, execute_rtl_evaluation, SynthesisResults
from src.utils import convert_to_fixed_point


logger = logging.getLogger(__name__)


def logging_cfg(logger_name, logfile, to_stdout=False):
    this_logger = logging.getLogger(logger_name)
    this_logger.setLevel(logging.DEBUG)
    if this_logger.hasHandlers():
        this_logger.handlers.clear()

    simple_formatter = logging.Formatter('')

    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setFormatter(simple_formatter)
    file_handler.setLevel(logging.DEBUG)
    this_logger.addHandler(file_handler)
    if to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(simple_formatter)
        stream_handler.setLevel(logging.INFO)
        this_logger.addHandler(stream_handler)
    return this_logger


def classifier_hw_evaluation(classifier, test_data, input_precision, weight_precision, savedir, 
                             copy_results=True, cleanup=False, rescale_inputs=False, simclk_ms=1, prefix='', only_rtl=False,
                             **kwargs):
    """Perform hardware evaluation of the classifier."""
    x_test, y_test = test_data

    # prepare the data for synthesis and simulation
    if not os.path.isdir(savedir):
        os.makedirs(savedir, exist_ok=True)
    hdl_dir = os.path.join(savedir, 'hdl')
    os.makedirs(hdl_dir, exist_ok=True)
    sim_dir = os.path.join(savedir, 'sim')
    os.makedirs(sim_dir, exist_ok=True)
    verilog_file = os.path.join(hdl_dir, 'top.v')
    tb_file = os.path.join(sim_dir, 'top_tb.v')
    inputs_file = os.path.join(sim_dir, 'x_test.txt')
    expected_file = os.path.join(sim_dir, 'y_expected.txt')
    output_file = os.path.join(sim_dir, 'y_test.txt')

    # create a hardware description of the classifier
    verilog_metadata = classifier.to_verilog(input_precision=input_precision,
                                             weight_precision=weight_precision,
                                             verilog_file=verilog_file,
                                             tb_file=tb_file,
                                             inputs_file=inputs_file,
                                             output_file=output_file,
                                             simclk_ms=simclk_ms,
                                             **kwargs)
    if verilog_metadata is None:
        logger.error("Failed to generate Verilog code for the classifier.")
        return SynthesisResults(None, None, None), 0.0

    x_test_rescaled = deepcopy(x_test)
    if rescale_inputs:
        if isinstance(input_precision, (list, tuple)):
            assert len(input_precision) == x_test.shape[1], \
                f"Length of input_precision {len(input_precision)} does not match number of features {x_test.shape[1]}"
            for idx, input_bits in enumerate(input_precision):
                x_test_rescaled[:, idx] = convert_to_fixed_point(x_test[:, idx], input_bits, rescale=False, fractional_bits=input_bits, signed=False)
        else:
            x_test_rescaled = convert_to_fixed_point(x_test, input_precision, rescale=False, fractional_bits=input_precision, signed=False)

    create_inputs_file_from_array(array=x_test_rescaled, 
                                  inputs_file=inputs_file,
                                  exclude_indices=verilog_metadata.get('exclude_indices', []),
                                  modifier=int)
    create_inputs_file_from_array(array=y_test,
                                  inputs_file=expected_file,
                                  modifier=lambda x: int(x))
    # execute the synthesis/simulation and get the results
    if only_rtl:
        results = execute_rtl_evaluation(savedir, hdl_dir, sim_dir, cleanup=cleanup, prefix=prefix)
    else:
        results = execute_hw_evaluation(savedir, hdl_dir, sim_dir, synclk_ms=simclk_ms, copy_results=copy_results, cleanup=cleanup, prefix=prefix)

    # check the accuracy from the simulation output
    simulation_output = np.loadtxt(output_file, dtype=float)
    sim_accuracy = classifier.accuracy_f(simulation_output, y_test)

    return results, sim_accuracy



def generate_testbench_code(input_width_dict, sim_period, input_separator, input_tb_file, output_tb_file, output_width_dict, signed=False):

    signed_prefix = 'signed ' if signed else ''

    inp_regs_decl = '\n'.join([
        f'reg {signed_prefix}[{width - 1}:0] {inp}_reg;' for inp, width in input_width_dict.items()
    ])
    inp_wire_decl = '\n'.join([
        f'wire {signed_prefix}[{width - 1}:0] {inp};' for inp, width in input_width_dict.items()
    ])

    outname = next(iter(output_width_dict.keys()))
    outwidth = output_width_dict[outname]

    dut_inputs_decl = ', '.join(input_width_dict.keys())
    fscanf1 = input_separator.join(['%d'] * len(input_width_dict))
    fscanf2 = ', '.join([f'{inp}_reg' for inp in input_width_dict])

    assign_regs_to_wires = '\n'.join([
        f'assign {inp} = {inp}_reg;' for inp in input_width_dict
    ])

    testbench_code = f"""
`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

localparam period = {sim_period};
localparam halfperiod = period/2;

{inp_regs_decl}
{inp_wire_decl}
wire [{outwidth - 1}:0] {outname};  // NOTE: Could use signed output here, but unsigned for now since we represent classes

integer fin, fout, r;

top DUT ({dut_inputs_decl}, {outname});

//read inp
initial begin
    $display($time, " << Starting the Simulation >>");
    fin = $fopen("{input_tb_file}", "r");
    if (fin == `NULL) begin
        $display($time, " file not found");
        $finish;
    end
    fout = $fopen("{output_tb_file}", "w");
    forever begin
        r = $fscanf(fin, "{fscanf1}\\n", {fscanf2});
        #period $fwrite(fout, "%d\\n", {outname});
        if ($feof(fin)) begin
            $display($time, " << Finishing the Simulation >>");
            $fclose(fin);
            $fclose(fout);
            $finish;
        end
    end
end

{assign_regs_to_wires}

endmodule
"""
    return testbench_code


def generate_testbench_code_vectorized(output_width, input_count, input_width, input_name, output_name,
                                       simclk_ns, inputs_file, output_file, input_separator, signed=False):

    # NOTE: Either ascending or descending order should be followed here. Not sure
    input_concatenation = ', '.join([f"temp_{input_name}[{i}]" for i in range(input_count)])
    # input_concatenation = ', '.join([f"temp_{input_name}[{input_count - 1 - i}]" for i in range(input_count)])

    display_internal = ['%d'] * input_count
    display_internal = '"' + ', '.join(display_internal) + '", ' + input_concatenation
    signed_prefix = 'signed ' if signed else ''

    testbench_code = f"""
`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    parameter OUTWIDTH = {output_width};
    parameter NUM_INP = {input_count};
    parameter WIDTH_A = {input_width};

    localparam period = {simclk_ns};

    reg {signed_prefix}[WIDTH_A-1:0] temp_{input_name} [0:NUM_INP-1]; // Temporary storage for inputs
    reg [NUM_INP*WIDTH_A-1:0] {input_name}_reg;        // Register to store concatenated input
    wire [NUM_INP*WIDTH_A-1:0] {input_name};
    wire [OUTWIDTH-1:0] {output_name};

    assign {input_name} = {input_name}_reg; // Assign register to wire

    top DUT (
        .{input_name}({input_name}),
        .{output_name}({output_name})
    );

    integer inFile, outFile, i;
    initial begin
        $display($time, " << Starting the Simulation >>");
        inFile = $fopen("{inputs_file}", "r");
        if (inFile == `NULL) begin
            $display($time, " file not found");
            $finish;
        end
        outFile = $fopen("{output_file}", "w");
        while (!$feof(inFile)) begin
            for (i = 0; i < NUM_INP; i = i + 1) begin
                $fscanf(inFile, "%d{input_separator}", temp_{input_name}[i]);
            end
            $fscanf(inFile, "\\n");
            // Concatenate inputs into a single vector
            {input_name}_reg = {{{input_concatenation}}};
            // $display({display_internal});

            #(period)
            $fwrite(outFile, "%d\\n", out);
        end
        #(period)
        $display($time, " << Finishing the Simulation >>");
        $fclose(outFile);
        $fclose(inFile);
        $finish;
    end

    // genvar gi;
    // generate
    // for (gi = 0; gi < NUM_A; gi = gi + 1) begin : genbit
    //    assign inp[(gi+1)*WIDTH_A-1:gi*WIDTH_A] = at[gi];
    // end
    // endgenerate

endmodule
"""
    return testbench_code


def generate_testbench_code_vectorized_mixed_precision_inputs(
        output_width, input_count, input_widths, input_name, output_name,
        simclk_ns, inputs_file, output_file, input_separator, signed=False
    ):
    assert len(input_widths) == input_count, "Input widths must match the number of inputs"

    # NOTE: Follow ascending order for parsing the inputs, but descending for concatenating them into the input array
    input_concatenation = ', '.join([f"temp_{input_name}_{i}" for i in range(input_count)])
    reverse_input_concatenation = ', '.join([f"temp_{input_name}_{input_count - 1 - i}" for i in range(input_count)])

    fscan_str = input_separator.join([f"%d" for _ in range(input_count)])
    fscan_str = '"' + fscan_str + '\\n", ' + input_concatenation
    display_internal = ['%d'] * input_count
    display_internal = '"' + ' '.join(display_internal) + '", ' + input_concatenation
    signed_prefix = 'signed ' if signed else ''

    total_input_width = sum(input_widths)

    input_register_declaration = '\n'.join([
        f'\treg {signed_prefix}[{width - 1}:0] temp_{input_name}_{i};' for i, width in enumerate(input_widths)
    ])

    testbench_code = f"""
`timescale 1ns/1ps
`define EOF 32'hFFFF_FFFF
`define NULL 0

module top_tb();

    localparam period = {simclk_ns};
    localparam OUTWIDTH = {output_width};
    localparam INPWIDTH = {total_input_width};
    
    wire [INPWIDTH-1:0] {input_name};
    reg  [INPWIDTH-1:0] {input_name}_reg;        // Register to store concatenated input
    wire [OUTWIDTH-1:0] {output_name};

    // Temporary storage for inputs
{input_register_declaration}

    assign {input_name} = {input_name}_reg; // Assign register to wire

    top DUT (
        .{input_name}({input_name}),
        .{output_name}({output_name})
    );

    integer inFile, outFile, i;
    initial begin
        $display($time, " << Starting the Simulation >>");
        inFile = $fopen("{inputs_file}", "r");
        if (inFile == `NULL) begin
            $display($time, " file not found");
            $finish;
        end
        outFile = $fopen("{output_file}", "w");
        while (!$feof(inFile)) begin
            
            // Parse inputs from file in ASCENDING order
            $fscanf(inFile, {fscan_str});
            // $display({display_internal});
            // Concatenate inputs into a single vector in DESCENDING order
            {input_name}_reg = {{{reverse_input_concatenation}}};

            #(period)
            $fwrite(outFile, "%d\\n", out);
        end
        #(period)
        $display($time, " << Finishing the Simulation >>");
        $fclose(outFile);
        $fclose(inFile);
        $finish;
    end

endmodule
"""
    return testbench_code


def create_inputs_file_from_array(array, inputs_file, exclude_indices=None, separator=' ', modifier=None):
    """Create a file with the inputs for the testbench from a given 2-D array."""
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if modifier is None:
        modifier = lambda x: x
    if exclude_indices is None:
        exclude_indices = []

    if array.ndim == 2:    
        array = np.delete(array, exclude_indices, axis=1)
        with open(inputs_file, 'w') as f:
            for row in array:
                f.write(separator.join([str(modifier(x)) for x in row]) + '\n')

    elif array.ndim == 1:
        array = np.delete(array, exclude_indices)
        with open(inputs_file, 'w') as f:
            for row in array:
                f.write(str(modifier(row)) + '\n')

    else:
        raise ValueError(f"Number of dimensions {array.ndim} is not supported. "
                         "Input array must be 1-D or 2-D")


def get_width(num):
    return int(num).bit_length()


def to_fixed_rescaled(value, bits, signed=True):
    if signed:
        min_value, max_value = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        min_value, max_value = 0, (1 << bits) - 1
    rescaled_value = (value - min_value) * (max_value - min_value) + min_value
    return int(value * (1 << bits))  # Convert float to fixed-point integer representation


def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


class ConvFxp:
    def __init__(self, sign, integer, fraction):
        self.s = sign
        self.int = integer
        self.frac = fraction
        if integer < 0 or fraction < 0:
            raise ValueError(f"Invalid fixed point: sign: {sign}, integer {integer}, fraction: {fraction}")

    def get_width(self):
        return (self.s + self.int + self.frac)

    @property
    def width(self):
        return self.get_width()

    def get_fxp(self):
        return (self.s, self.int, self.frac)

    def to_float(self,x):
        return to_float(x, self.frac)

    def to_fixed(self, f):
        b = to_fixed(f, self.frac)
        maxval = 1 << (self.int + self.frac)
        b = np.clip(b, -1*self.s*maxval, maxval-1)
        return int(b)


def to_float(x,e):
    c = abs(x)
    sign = 1
    if x < 0:
        # convert back from two's complement
        c = x - 1
        c = ~c
        sign = -1
    f = (1.0 * c) / (2 ** e)
    f = f * sign
    return f

def to_fixed(f,e):
    a = f* (2**e)
    b = int(round(a))
    if a < 0:
        # next three lines turns b into it's 2's complement.
        b = abs(b)
        b = ~b
        b = b + 1
    return b


def convertCoef(mdata, fxp, toFP):
    transp=[np.transpose(i).tolist() for i in mdata]
    coefficients=[]
    for i, l in enumerate(transp):
        newn=list()
        for j,n in enumerate(l):
            neww=list()
            for k,w in enumerate(n):
                if toFP:
                    neww.append(fxp.to_float(w))
                else:
                    neww.append(fxp.to_fixed(w))
            newn.append(neww)
        coefficients.append(newn)
    return coefficients

#similar to the above, should be combined
def convertIntercepts(mdata, lfxp, toFP):
    transp=[np.transpose(i).tolist() for i in mdata]
    intercepts=[]
    for i, l in enumerate(transp):
        fxp=lfxp[i]
        neww=list()
        for k,w in enumerate(l):
            if toFP:
                neww.append(fxp.to_float(w))
            else:
                neww.append(fxp.to_fixed(w))
        intercepts.append(neww)
    return intercepts


def convert_params_to_fxp(mdata, fxp, toFP):
    convfxp=[]
    if isinstance(mdata[0], np.ndarray) or isinstance(mdata[0], list):
        ldata=mdata
    else:
        ldata=[mdata]
    for i, l in enumerate(ldata):
        neww=list()
        for k,w in enumerate(l):
            if toFP:
                neww.append(fxp.to_float(w))
            else:
                neww.append(fxp.to_fixed(w))
        convfxp.append(neww)
    if isinstance(mdata[0], np.ndarray) or isinstance(mdata[0], list):
        return convfxp
    else:
        return convfxp[0]


def get_maxabs(array):
    return np.max(np.abs(array))


class svm_fxp_ps:
    
    def __init__(self, coefs, intercept, inpfxp, wfxp, y_train, is_regressor=False):
        self.coefs = coefs
        self.intercept = intercept
        self.inpfxp=inpfxp
        self.Q=inpfxp.frac+wfxp.frac
        self.YMAX=max(y_train)
        self.YMIN=min(y_train)
        self.YMAP=list(set(y_train))
        self.regressor=is_regressor

    def vsign (self, a):
        if a>=0:
            return 0
        else:
            return 1

    def decisionFunction(self,pred):
        ymap=self.YMAP
        dMatrix=np.zeros((len(ymap),len(ymap)))
        ind=0
        for m in range(len(ymap)):
            for n in range(len(ymap)):
                if m<n:
                    dMatrix[m][n]=(1-self.vsign(pred[ind]))
                    dMatrix[n][m]=self.vsign(pred[ind])
                    ind=ind+1
        s=np.sum(dMatrix, axis=1)
        return ymap[np.argmax(s)]

    def predict(self,X_test):
        prediction=[]
        for i, l in enumerate(X_test):
            # newl=[]
            # for k,w in enumerate(l):
            #     newl.append(self.inpfxp.to_fixed(w))
            # prediction.append( self.predict_one(newl))
            prediction.append(self.predict_one(l))
        return np.asarray(prediction)

    def get_accuracy(self, X_test, y_test):
        pred=self.predict(X_test) + self.YMIN
        return accuracy_score(pred, y_test)

    def predict_one(self, x):
        inp=x
        layer=0
        out=[]
        for i,neuron in enumerate(self.coefs):
                temp=self.intercept[i]
                for j in range(len(neuron)):
                    temp+=neuron[j]*inp[j]
                out.append(temp)
        if self.regressor:
            p=to_float(out[0],self.Q)
            return np.clip(np.round(p),self.YMIN,self.YMAX)
        else:
            # print(out)
            if len(out)!=1:
                p=self.decisionFunction(out)
            else:
                if out[0]>=0:
                    p=1
                else:
                    p=0
            return p
        
def one_hot_encoder(num_classes):
    decision = {}
    for i in range(num_classes-1, -1, -1):
        original_string = num_classes * '0'
        new_string = original_string[:i] + '1' + original_string[i + 1:]
        decision[num_classes-1-i] = new_string
    return decision

def min_binary_digits(decimal_number):
    # Calculate the binary logarithm of the decimal number
    binary_log = math.log2(decimal_number)
    # Round up to the nearest integer
    min_digits = math.ceil(binary_log)
    return min_digits


def replaceWithX(pbt):
    print("in replaceWithBinary:")
    print(pbt)
    for idx, val in enumerate(pbt):
        if val not in [0, 1]:
            pbt[idx] = 'x'
    path = ''
    for char in pbt:
        path += str(char)
    rpath = "".join(reversed(path))
    return rpath


def BespokeBST(pbt, modified_pbt,  startNode, target, steps):
    paths = []
    path = []
    for item in modified_pbt:
        path.append(item)
    index = 0
    currentNode = 0
    for edge in pbt:
        if edge == startNode:
            currentNode = edge
    for i in range(1, steps + 1, 1): # steps = number of layers to reach the target
        # print("Step " + str(i))
        # search list index for currentNode
        for idx in range(len(pbt)):
            if pbt[idx] == currentNode:
                index = idx
        if i == steps: # we have reached the desired leaf
            if currentNode[0] == target:
                path[index] = 0
                # print(path)
                # need to change edges that where not traversed
                temp = replaceWithX(path)
                paths.append(temp)
                # print("Path appended in paths: " + temp)
            elif currentNode[1] == target:
                path[index] = 1
                # print(path)
                # need to change edges that where not traversed
                temp = replaceWithX(path)
                paths.append(temp)
        elif i < steps: # we have not reached the desired leaf
            if currentNode[0] == target:
                path[index] = 0
                # print(path)
                temp = list(currentNode)
                temp[1] = temp[1] - 1
                currentNode = tuple(temp)
            elif currentNode[1] == target:
                path[index] = 1
                # print(path)
                temp = list(currentNode)
                temp[0] = temp[0] + 1
                currentNode = tuple(temp)
            else: # the target is in the middle leaves
                # print("the target is the middle leaves")
                # create second path
                path2 = []
                for item in path:
                    path2.append(item)
                # print("currentNode: ")
                # print(currentNode)
                # create currentNode2
                cn2 = []
                temp = list(currentNode)
                for item in temp:
                    cn2.append(item)
                currentNode2 = tuple(cn2)
                path2[index] = 1
                # print("Second path:")
                # print(path2)
                temp = list(currentNode2)
                temp[0] = temp[0] + 1
                currentNode2 = tuple(temp)
                # print("currentNode2: ")
                # print(currentNode2)
                temp2 = BespokeBST(pbt, path2, currentNode2, target, steps - i) # returns an alternative path
                for pth in temp2:
                    paths.append(pth)
                    print("Path appended in paths: " + pth)
                # change first path
                path[index] = 0
                # print("First path:")
                # print(path)
                temp = list(currentNode)
                temp[1] = temp[1] - 1
                currentNode = tuple(temp)


    return paths

def create_paths(num_classes):
    unique_classes = []
    pbt = []

    for i in range(num_classes):
        unique_classes.append(i)

    for i in range(len(unique_classes)-1):
        for j in range(len(unique_classes)-1, i, -1):
            pbt.append((unique_classes[i], unique_classes[j]))
    print(f"pbt: {pbt}")
 
    # create a dictionary and map each class to that paths that lead to it
    decision = {}
    for cl in range(len(unique_classes)):
        paths = BespokeBST(pbt, pbt, pbt[0], cl, num_classes-1)
        decision.update({cl : paths})
    return decision


class mlp_fxp_ps:
    
 def __init__(self, coefs, intercept, inpfxp, wfxp, L0trunc, y_train, model):
     self.coefs = coefs
     self.intercept = intercept
     self.inpfxp=inpfxp
     if len(intercept)==2:
        self.trunclst=[L0trunc,0]
     elif len(intercept)==3:
        self.trunclst=[L0trunc,L0trunc,0]
     self.Q=inpfxp.frac+2*wfxp.frac-L0trunc
     self.YMAX=max(y_train)
     self.YMIN=min(y_train)
     if "MLPRegressor" in str(type(model)):
         self.regressor=True
     else:
         self.regressor=False

 def predict(self,X_test):
    prediction=[]
    for i, l in enumerate(X_test):
        newl=[]
        for k,w in enumerate(l):
            newl.append(self.inpfxp.to_fixed(w))
        prediction.append( self.predict_one(newl))
    return np.asarray(prediction)

 def get_accuracy(self, X_test, y_test):
    pred=self.predict(X_test)
    if self.regressor:
        pred=np.clip(np.round(pred),self.YMIN,self.YMAX)
    else:
        pred=pred+self.YMIN
    return accuracy_score(pred, y_test)

 def predict_one(self, x):
    inp=x
    layer=0
    out=[]
    for layer in range(len(self.coefs)):
        for i,neuron in enumerate(self.coefs[layer]):
            temp=self.intercept[layer][i]
            for j in range(len(neuron)):
                temp+=neuron[j]*inp[j]
            if layer==0:
                if (temp<0):
                    temp = 0
            temp=temp>>self.trunclst[layer]
            out.append(temp)
        inp=list(out)
        out=[]
    if self.regressor:
        return to_float(inp[0],self.Q)
    else:
        return np.argmax(inp)
