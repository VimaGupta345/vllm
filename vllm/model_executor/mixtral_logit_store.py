"""Defining the MixtralLogitStore class to store the router logits computed in forward pass of ixtralMoE for each token in the sequence."""
from typing import List, Optional, Tuple

import numpy as np
import os
import torch
import csv
import torch.nn.functional as F


class MixtralLogitStore:
    _instance = None

    """This code dumps the router logits computed in forward pass of MixtralMoE for each token in he sequence."""
    def __init__(self):
        super(MixtralLogitStore, self).__init__()
        self.router_logit_store = []
        self.batch_idx = 0
        self.profile_complete = False
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MixtralLogitStore()
        return cls._instance
    
    def mark_profiling_done(self):
        self.profile_complete = True
        
    def dump_router_logits(self, router_logits, layer_idx):
        if router_logits.is_cuda:
            router_logits = router_logits.cpu()
        self.router_logit_store.append((self.batch_idx, layer_idx, router_logits))
        
    def end_batch(self):
        self.batch_idx += 1
        # if self.batch_idx%5 == 0:
        self.write_to_csv(self.batch_idx)
        self.clear_router_logits()
        
    def get_stored_router_logits(self):
        # router_logits: (batch * sequence_length, n_experts)
        # returned value: number of tokens, batch_size*seq_len, n_experts
        return torch.stack(self.router_logit_store)
    
    def clear_router_logits(self):
        self.router_logit_store = []
        
    def write_to_csv(self, batch_idx):
        if not os.path.exists('router'):
            os.mkdir('router_logits')
        csv_path = f"router/router_logits_{batch_idx}.csv"
        with open(csv_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Check if the file is empty to decide whether to write the header
            csvfile.seek(0)  # Go to the start of the file
            if csvfile.tell() == 0:  # If file is empty, write the header
                # Creating a header row
                # Assuming a maximum number of logits, you can adjust this as needed
                max_logits = 8  # Example, change this based on your maximum number of logits
                header = ['Batch Index', 'Layer Index'] + [f'Logit_{i}' for i in range(max_logits)]
                csvwriter.writerow(header)
            # Writing the logit data
            for batch_idx, layer_idx, logits_tensor in self.router_logit_store:
                logits_list = logits_tensor.tolist()  # Convert tensor to a list
                for logit in logits_list:
                    row = [batch_idx, layer_idx] + logit  # Combine batch and layer index with logit values
                    csvwriter.writerow(row)