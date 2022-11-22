import timeit
import torch
from collections import OrderedDict
import gc
from building_blocks.builder import PRIMITIVES
from general_functions.utils import add_text_to_file, clear_files_in_the_list
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

# the settings from the page 4 of https://arxiv.org/pdf/1812.03443.pdf
#### table 2
CANDIDATE_BLOCKS_BACKBONE = ["ir_k3_r2_re", "ir_k3_r2_hs", "ir_k3_re",
                             "ir_k3_hs", "ir_k5_r2_re", "ir_k5_r2_hs",
                             "ir_k5_re", "ir_k5_hs", "ir_k7_re", "ir_k7_hs", "skip"]

CANDIDATE_BLOCKS_HEAD = ["ir_k3_re", "ir_k3_r2_re", "ir_k5_re", "ir_k5_r2_re", "skip"]

CANDIDATE_BLOCKS_FPN = ["ir_k3_re", "ir_k3_r2_re", "ir_k3_r3_re",
                        "ir_k5_re", "ir_k5_r2_re", "ir_k5_r3_re",
                        "none", "skip"]

SEARCH_SPACE_BACKBONE = OrderedDict([
    #### table 1. input shapes of 11 searched layers (considering with strides)
    ("input_shape", [16,
                     32, 64, 128,
                     128, 256, 256,
                     256, 512, 512,
                     1024]),
    # table 1. filter numbers over the 11 layers
    ("channel_size", [32,
                      64,  128,  128,
                      256,  256,  256,
                      512,  512,  1024,
                      1024]),
    # table 1. strides over the 11 layers
    ("strides", [2,
                 2, 1, 2,
                 1, 1, 1,
                 1, 2, 1,
                 1])
])

SEARCH_SPACE_HEAD = OrderedDict([
    ("input_shape",
        [256, 256,
         256, 256,
         256]),
    ("channel_size",
        [256, 256,
         256, 256,
         256]),
    ("strides",
        [1, 1,
         1, 1,
         1])
])

SEARCH_SPACE_FPN = OrderedDict([
    ("input_shape",
        [512, 512,
         512, 512,
         512, 512,
         512, 512]),
    ("channel_size",
        [512, 512,
         512, 512,
         512, 512,
         512, 512]),
    ("strides",
        [1, 2,
         1, 1,
         1, 2,
         1, 1])
])

YOLO_LAYER_26 = {
    'mask': (3,4,5),
    'anchors': (38,29,  71,50,  120,70,  152,119,  249,261,  262,132),
    'classes': 4,
    'num': 6,
    'jitter': .3,
    'ignore_thresh': .7,
    'truth_thresh': 1,
    'random': 1
}

YOLO_LAYER_13 = {
    'mask': (0,1,2),
    'anchors': (38,29,  71,50,  120,70,  152,119,  249,261,  262,132),
    'classes': 4,
    'num': 6,
    'jitter': .3,
    'ignore_thresh': .7,
    'truth_thresh': 1,
    'random': 1
}


def extract_anchors(yolo_layer):
    anchor_idxs = [int(x) for x in yolo_layer["mask"]]  # [1 2 3]
    # Extract anchors
    anchors = [int(x) for x in yolo_layer["anchors"]]  # [10 14  23 27  37 58  81 82  135 169  344 319]
    anchors = [(anchors[i], anchors[i + 1]) for i in
               range(0, len(anchors), 2)]  # [(10,14),(23,27),(37,58),(81,82),(135,169),(344,319)]
    anchors = [anchors[i] for i in anchor_idxs]  # [(23,27),(37,58),(81,82)]
    return anchors

# **** to recalculate latency use command:
# l_table = LookUpTable(calculate_latency=True, path_to_file='lookup_table.txt', cnt_of_runs=50)
# results will be written to './supernet_functions/lookup_table.txt''
# **** to read latency from the another file use command:
# l_table = LookUpTable(calculate_latency=False, path_to_file='lookup_table.txt')
class LookUpTable:
    def __init__(self, candidate_blocks_backbone=None,
                 candidate_blocks_head=None,
                 candidate_blocks_fpn=None,
                 search_space_backbone=None,
                 search_space_head=None,
                 search_space_fpn=None,
                 calculate_latency=False):

        if candidate_blocks_backbone is None:
            candidate_blocks_backbone = CANDIDATE_BLOCKS_BACKBONE
        if candidate_blocks_head is None:
            candidate_blocks_head = CANDIDATE_BLOCKS_HEAD
        if candidate_blocks_fpn is None:
            candidate_blocks_fpn = CANDIDATE_BLOCKS_FPN

        if search_space_backbone is None:
            search_space_backbone = SEARCH_SPACE_BACKBONE
        if search_space_head is None:
            search_space_head = SEARCH_SPACE_HEAD
        if search_space_fpn is None:
            search_space_fpn = SEARCH_SPACE_FPN

        self.num_anchors = len(YOLO_LAYER_26['mask'])
        self.num_cls = YOLO_LAYER_26['classes']
        self.cnt_layers = len(search_space_backbone["input_shape"]) # num. of layers for backbone
        self.head_layers = len(search_space_head["input_shape"]) # num. of layers for head
        self.fpn_edges = len(search_space_fpn["input_shape"]) # num. of edges for fpn

        self.anchors_26 = extract_anchors(YOLO_LAYER_26)
        self.anchors_13 = extract_anchors(YOLO_LAYER_13)

        # constructors for each operation
        # select operations subset from PRIMITIVES
        self.lookup_table_operations = {op_name : PRIMITIVES[op_name] for op_name in candidate_blocks_backbone}
        self.lookup_table_operations_head = {op_name : PRIMITIVES[op_name] for op_name in candidate_blocks_head}
        self.lookup_table_operations_fpn = {op_name: PRIMITIVES[op_name] for op_name in candidate_blocks_fpn}

        # arguments for the ops constructors. one set of arguments for all 9 constructors at each layer
        # input_shapes just for convinience
        self.layers_parameters, self.layers_input_shapes = self._generate_layers_parameters(search_space_backbone)
        self.layers_parameters_head, self.layers_input_shapes_head = self._generate_layers_parameters(search_space_head)
        self.layers_parameters_fpn, self.layers_input_shapes_fpn = self._generate_layers_parameters(search_space_fpn)

        # lookup_table
        self.lookup_table_latency = None
        self.lookup_table_latency_head = None
        self.lookup_table_latency_fpn = None
        if calculate_latency:
            self._create_from_operations(cnt_of_runs=CONFIG_SUPERNET['lookup_table']['number_of_runs'],
                                         write_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'],
                                         write_to_file_head=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table_head'],
                                         write_to_file_fpn=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table_fpn'])
        else:
            self._create_from_file(path_to_file=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table'],
                                   path_to_file_head=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table_head'],
                                   path_to_file_fpn=CONFIG_SUPERNET['lookup_table']['path_to_lookup_table_fpn'])

    @staticmethod
    def _generate_layers_parameters(search_space, prune=False):
        # layers_parameters are : C_in, C_out, expansion, stride, prune
        if prune:
            layers_parameters = [(search_space["input_shape"][layer_id],  # C_in for layer id
                                  search_space["channel_size"][layer_id],  # C_out for layer id
                                  None,  # expansion for layer id
                                  search_space["strides"][layer_id],  # stride for layer id
                                  search_space["prune"][layer_id], # mid_depth&mid for layer id
                                  ) for layer_id in range(len(search_space["input_shape"]))]
        else:
            layers_parameters = [(search_space["input_shape"][layer_id], # C_in for layer id
                                  search_space["channel_size"][layer_id], # C_out for layer id
                                  6, # expansion for layer id
                                  search_space["strides"][layer_id], # stride for layer id
                                  None,
                                 ) for layer_id in range(len(search_space["input_shape"]))]

        
        # layers_input_shapes are (C_in, input_w, input_h)
        layers_input_shapes = search_space["input_shape"]
        
        return layers_parameters, layers_input_shapes
    
    # CNT_OP_RUNS us number of times to check latency (we will take average)
    def _create_from_operations(self, cnt_of_runs, write_to_file=None, write_to_file_head=None, write_to_file_fpn=None):
        self.lookup_table_latency = self._calculate_latency(self.cnt_layers,
                                                            self.lookup_table_operations,
                                                            self.layers_parameters,
                                                            self.layers_input_shapes,
                                                            cnt_of_runs)
        self.lookup_table_latency_head = self._calculate_latency(self.head_layers,
                                                            self.lookup_table_operations_head,
                                                            self.layers_parameters_head,
                                                            self.layers_input_shapes_head,
                                                            cnt_of_runs)
        self.lookup_table_latency_fpn = self._calculate_latency(self.fpn_edges,
                                                            self.lookup_table_operations_fpn,
                                                            self.layers_parameters_fpn,
                                                            self.layers_input_shapes_fpn,
                                                            cnt_of_runs)
        if write_to_file is not None:
            self._write_lookup_table_to_file(write_to_file,
                                             self.lookup_table_operations,
                                             self.cnt_layers,
                                             self.lookup_table_latency)
        if write_to_file_head is not None:
            self._write_lookup_table_to_file(write_to_file_head,
                                             self.lookup_table_operations_head,
                                             self.head_layers,
                                             self.lookup_table_latency_head)
        if write_to_file_fpn is not None:
            self._write_lookup_table_to_file(write_to_file_fpn,
                                             self.lookup_table_operations_fpn,
                                             self.fpn_edges,
                                             self.lookup_table_latency_fpn)
    
    def _calculate_latency(self, layers, operations, layers_parameters, layers_input_shapes, cnt_of_runs):
        LATENCY_BATCH_SIZE = 1
        latency_table_layer_by_ops = [{} for i in range(layers)]
        
        for layer_id in range(layers):
            for op_name in operations:
                op = operations[op_name](*layers_parameters[layer_id])
                input_sample = torch.randn((LATENCY_BATCH_SIZE, *layers_input_shapes[layer_id]))
                globals()['op'], globals()['input_sample'] = op, input_sample
                total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()",
                                           globals=globals(), number=cnt_of_runs)
                # measured in micro-second
                latency_table_layer_by_ops[layer_id][op_name] = total_time / cnt_of_runs / LATENCY_BATCH_SIZE * 1e6
                
        return latency_table_layer_by_ops
    
    def _write_lookup_table_to_file(self, path_to_file, operations, layers, latency):
        clear_files_in_the_list([path_to_file])
        ops = [op_name for op_name in operations]
        text = [op_name + " " for op_name in ops[:-1]]
        text.append(ops[-1] + "\n")
        
        for layer_id in range(layers):
            for op_name in ops:
                text.append(str(latency[layer_id][op_name]))
                text.append(" ")
            text[-1] = "\n"
        text = text[:-1]
        
        text = ''.join(text)
        add_text_to_file(text, path_to_file)
    
    def _create_from_file(self, path_to_file, path_to_file_head, path_to_file_fpn):
        self.lookup_table_latency = self._read_lookup_table_from_file(path_to_file, self.cnt_layers)
        self.lookup_table_latency_head = self._read_lookup_table_from_file(path_to_file_head, self.head_layers)
        self.lookup_table_latency_fpn = self._read_lookup_table_from_file(path_to_file_fpn, self.fpn_edges)
    
    def _read_lookup_table_from_file(self, path_to_file, layers):
        latences = [line.strip('\n') for line in open(path_to_file)]
        ops_names = latences[0].split(" ")
        latences = [list(map(float, layer.split(" "))) for layer in latences[1:]]
        
        lookup_table_latency = [{op_name : latences[i][op_id] 
                                      for op_id, op_name in enumerate(ops_names)
                                     } for i in range(layers)]
        return lookup_table_latency

if __name__=="__main__":

    latences = [line.strip('\n') for line in open("./lookup_table.txt")]
    print(latences)

    ops_names = latences[0].split(" ")
    print(ops_names)
    print("length opnames:", ops_names)
    latences = [list(map(float, layer.split(" "))) for layer in latences[1:]]
    print(latences)
    lookup_table_latency = [{op_name: latences[i][op_id]
                             for op_id, op_name in enumerate(ops_names)
                             } for i in range(11)]
    print(lookup_table_latency)

    # import netron
    # netron.start('./model.onnx')

    layers_parameters, layers_input_shapes = LookUpTable()._generate_layers_parameters(SEARCH_SPACE_BACKBONE)
    operations = LookUpTable().lookup_table_operations
    #print(operations)

    num_of_runs = 50

    for layer_id in range(11):
        for op_name in operations:
            op = operations[op_name](*layers_parameters[layer_id])
            #print(op)
            input_sample = torch.randn((1, *layers_input_shapes[layer_id]))

            print(input_sample.shape)

            globals()['op'], globals()['input_sample'] = op, input_sample
            total_time = timeit.timeit('output = op(input_sample)', setup="gc.enable()",
                                       globals=globals(), number=num_of_runs)
            break
        break
    latency_table_layer_by_ops = total_time / num_of_runs * 1e6
    print(latency_table_layer_by_ops)
