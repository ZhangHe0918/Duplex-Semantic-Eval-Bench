import sys
sys.path.append('third-party/MatchaTTS')
sys.path.append("/home/ma-user/work/cuiwenqian/GLM-4-Voice")
import os
import json
import torch.distributed as dist
checkpoint_id=1250
model_id = f"/home/ma-user/work/cuiwenqian/GLM-4-Voice/log/fullfisher_120s_speech5_batch2_e2.0_lr4e-6_wd0.0/checkpoint-{checkpoint_id}"
ckpt_base_path = "/home/ma-user/work/cuiwenqian/hf_model_ckpt/GLM-4-Voice"
dataset = "fisher" #fisher or condor
if dataset == "fisher":
    dataset_json_file = "/home/ma-user/work/cuiwenqian/GLM-4-Voice/GLM-FisherWavCodes_16k_120s_jsonl/GLM-FisherWav"