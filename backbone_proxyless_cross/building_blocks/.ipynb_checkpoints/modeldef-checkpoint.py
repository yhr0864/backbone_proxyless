# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# please, end the file with '}' and nothing else. this file updated automatically

MODEL_ARCH = {
    
}

# example for testing
Test_model_arch = {
    "test_net": {
    "block_op_type_backbone": [
            ["ir_k3_hs"],
            ["ir_k3_re"], ["ir_k3_hs"], ["ir_k5_re"],
            ["ir_k5_re"], ["ir_k5_re"], ["skip"],
            ["ir_k3_hs"], ["ir_k5_re"], ["skip"],
            ["ir_k5_re"]
    ],
    "block_op_type_head26": [
            ["ir_k3_re"], ['skip'],
            ["ir_k5_re"], ["ir_k3_re"],
            ["skip"]
    ],
    "block_op_type_head13": [
            ["ir_k5_re"], ['skip'],
            ["ir_k3_re"], ["skip"],
            ["skip"]
    ],
    "block_op_type_fpn": [
            ["ir_k3_re"], ["ir_k5_re"],
            ["skip"], ["ir_k3_re"],
            ["ir_k5_re"], ["none"],
            ["ir_k3_re"], ["ir_k5_re"]
    ]
    },


   
}