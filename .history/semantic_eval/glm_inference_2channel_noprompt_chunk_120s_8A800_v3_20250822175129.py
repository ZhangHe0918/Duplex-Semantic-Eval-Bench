"""
This file is the GLM-4-Voice inference code for generating double channel audio.
"""

import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append("/home/ma-user/work/cuiwenqian/GLM-4-Voice")
import os
import torch.distributed as dist
checkpoint_id = 1250
model_id = f"/home/ma-user/work/cuiwenqian/GLM-4-Voice/log/fullfisher_120s_speech5_batch2_e2.0_lr4e-6_wd0.0/checkpoint-{checkpoint_id}"
ckpt_base_path = "/home/ma-user/work/cuiwenqian/hf_model_ckpt/GLM-4-Voice"
dataset = "fisher"  # fisher or condor
if dataset == "fisher":
    dataset_json_file = "/home/ma-user/work/cuiwenqian/GLM-4-Voice/GLM-FisherWavCodes_16k_120s_jsonl/GLM-FisherWavCodes_16k_120s_train_val_test.json"  # fisher
elif dataset == "condor":
    dataset_json_file = "/home/ma-user/work/cuiwenqian/GLM-4-Voice/GLM-CondorMp3Codes_16k_120s_part_001_jsonl/GLM-CondorMp3Codes_16k_120s_part_001_testset.json"  # condor
# user_audio_token_path = "/home/ma-user/work/cuiwenqian/GLM-4-Voice/GLM-FisherWavCodes_16k_30s_jsonl/GLM-FisherWavCodes_16k_30s.jsonl"
temperature = 1.3  # default: 1.5
p_value = 0.9  # default: 0.99
eval_num = 20
chunk_size = 5
condition = "conditional"
split_generation = False
if split_generation:
    num_parts = 2
    part = 1
output_folder = f"/home/ma-user/work/zhanghe/GLM-4-Voice/gen_interleaved_dialog_v3/fullfisher_120s_speech5_batch2_e2.0_lr4e-6_wd0.0_checkpoint-{checkpoint_id}/{condition}/{dataset}_testset_t{temperature}_p{p_value}_noadditionaltokens"
turn_mask_path="/home/ma-user/work/zhanghe/GLM_Fisher120s_turned/GLM_Fisher120s_semantic_turned_token.jsonl"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

import torchaudio
import soundfile as sf
import librosa
import torch
import json
import pandas as pd
from typing import Optional, List
import time
from tqdm import tqdm
import numpy as np
import scipy.io.wavfile as wavfile
import wave
import tempfile
import uuid
from transformers import WhisperFeatureExtractor, AutoTokenizer, AutoModel, BitsAndBytesConfig,GenerationConfig
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
import torch.distributed as dist
"""
train/val/test are split based on the ids
1. Training set: everything else (058-110)
2. Validation set: audio 111-113
3. Test set: 114-116
Train/Val/Test split: roughly 90/5/5
testset len: 11401
"""
# load the extract user audio tokens
with open(dataset_json_file, "r") as f:
    data = json.load(f)
testset = None
if dataset == "fisher":
    testset = data["test"]
else:
    testset = data
# print("testset[0]:")
# print(testset[0])
"""
for both fisher and condor
print(data[0])
{
'id': 'fe_03_09401_285_315.wav',  # condor: ...mp3
'text': 'asdf', 
'left_audio_ids': [6937, 14509, 7668, 34, 6219, ...], 
'right_audio_ids': [2328, 2427, 8520, 8520, 8520, ...]
}
"""

if dataset == "condor":
    eval_num = "all"
print("Total length of testset:", len(testset))
if eval_num == "all":
    eval_num = len(testset)
else:
    testset = testset[:eval_num]
print(f"total testset len: {len(testset)}")
    
if split_generation:
    print(f"total testset len: {len(testset)}, splitting into {num_parts} parts.")
    groups = np.array_split(testset, num_parts)
    # group0 = groups[0].tolist()
    # group1 = groups[1].tolist()
    # group2 = groups[2].tolist()
    # group3 = groups[3].tolist()
    testset = groups[part].tolist()
    eval_num = len(testset)
    # print(testset[0])
    for p in range(num_parts):
        print(len(groups[p]))
    print("===============================")

if not dist.is_initialized():
    # single machine
    dist.init_process_group(backend='nccl')

class InferenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

#create dataset and dataloader

test_data = []
with open('/home/ma-user/work/zhanghe/GLM_Fisher120s_turned/GLM_Fisher120s_semantic_turned_token.jsonl', 'r') as f:
    for line in f:
        if line.strip():  # skip empty line
            item = json.loads(line.strip())
            test_data.append(item)

# extract id_list
id_list = [item['id'] for item in test_data]
testset = [item for item in testset if item['id'] in id_list]
dataset = InferenceDataset(testset)
sampler = DistributedSampler(
    dataset, 
    shuffle=False,
)
dataloader = DataLoader(dataset,sampler=sampler,batch_size=1)

class SpeechProcessor:
    def __init__(
        self,
        tokenizer_path="./glm-4-voice-tokenizer",
        flow_path="./glm-4-voice-decoder",
        device="cuda",
        dtype="bfloat16",
        device_map = "auto",
        **kwargs
    ):
        self.device = torch.device(device)

        # take all kwargs as value
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize GLM model
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if dtype == "int4" else None

        print("Loading GLM model...")
        self.glm_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if dtype == "bfloat16" else None,
            max_memory={i: self.max_memory_per_gpu for i in range(8)},
            device_map=device_map  # choose specific gpu from multi-gpu
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Initialize speech tokenizer
        print("Loading Whisper model...")
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

        # Initialize audio decoder
        print("Loading audio decoder...")
        flow_config = os.path.join(flow_path, "config.yaml")
        flow_checkpoint = os.path.join(flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(flow_path, 'hift.pt')
        self.audio_decoder = AudioDecoder(
            config_path=flow_config,
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=device
        )
        # print("sleeping forever...")
        # time.sleep(2**31 - 1)  # Sleep for ~68 years (max 32-bit signed int)
    
    def generate_conditional_interleaved_speech_token_only(
        self,
        input_audio_path,
        id,
        user_audio_tokens,
        model_audio_tokens,
        turn_mask_path=None,
        temperature=0.2,
        top_p=0.8,
        output_path=None, 
        prompt_seconds=30,  # seconds of audio used as prompt (must be an even number!)
        total_seconds=120,
    ):
        """
        Process speech in an interleaved fashion to create full duplex dialogue:
        1. the user's audio is already extracted into tokens
        2. For each user token, append it to sequence and generate one model token
        3. Continue until all user tokens are used
        4. Decode model tokens to speech
        """
        with open(turn_mask_path,'r',encoding='utf-8') as f:
            is_turn = False
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    if data["id"]== str(id):
                        turn_mask = data["filtered_turn"]
                        continuation_info = data["continuation_info"]
                        is_turn = True
            if not is_turn :
                turn_mask = [{"start":0,"end":1500}]
            # print(turn_mask)
        stereo_output_path = output_path
        # stereo_output_path = f"{output_path.replace('.wav', f'_stereo.wav')}"
        if os.path.exists(stereo_output_path) :
            print(f"Stereo output file {stereo_output_path} already exists. Skipping generation.")
            return None, None, "exist"
        
        if 16383 in user_audio_tokens or 16383 in model_audio_tokens:
            print(f"16383 in user_audio_tokens or model_audio_tokens. Skipping...")
            return None, None, "unsupported_token"

        print(f"User audio tokens extracted: {len(user_audio_tokens)} tokens")
        
        # Prepare initial prompt
        prompt = f"<|begin_of_audio|><|begin_of_audio|>"

        num_prompt_tokens = int(prompt_seconds * 12.5)  # GLM-4-Voice uses 12.5 tokens per second
        assert num_prompt_tokens % chunk_size == 0, f"cannot include integer number of chunk_size as prompt (i.e., num_prompt_tokens % chunk_size != 0)"
        assert len(user_audio_tokens) == len(model_audio_tokens), "User and model audio tokens must be of the same length"
        assert num_prompt_tokens <= len(user_audio_tokens), "Not enough user audio tokens for the prompt"
        user_tokens_chunks = [user_audio_tokens[i:i+chunk_size] for i in range(0, len(user_audio_tokens), chunk_size)]
        model_tokens_chunks = [model_audio_tokens[i:i+chunk_size] for i in range(0, len(model_audio_tokens), chunk_size)]
        interleaved_tokens = []
        for user_chunk, model_chunk in zip(user_tokens_chunks, model_tokens_chunks):
            interleaved_tokens.append(["<|user|>"] + [f"<|audio_{user_token}|>" for user_token in user_chunk] + ["<|assistant|>"] + [f"<|audio_{model_token}|>" for model_token in model_chunk])
        # [["<|user|>", "<|audio_token1|>", "<|audio_token2|>", "<|audio_token3|>", ..., ["<|assistant|>","<|audio_token1|>", "<|audio_token2|>", "<|audio_token3|>", ...,],["<|user|>", "<|audio_token1|>", "<|audio_token2|>", "<|audio_token3|>", ..., ["<|assistant|>","<|audio_token1|>", "<|audio_token2|>", "<|audio_token3|>", ...,]]
        # interleaved_tokens_flattened = [item for sublist in interleaved_tokens for item in sublist]
        # Tokenize initial prompt
        inputs = self.glm_tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        
        segmentation = {}
        # Process tokens in an interleaved fashion (this is chunk-level interleaving)
        # for conditional chunk interleaving, <|user|> 5 speech tokens <|assistant|> is given, only let the model to generate the model's audio tokens
        if split_generation:
            print(f"Generating interleaved dialogue (part {part})...")
        for current_turn_id in tqdm(range(len(turn_mask))):# turn by turn process
            # Track generated model tokens separately
            output_model_audio_tokens = []
            print(f"total turn is {len(turn_mask)}...")
            print(f"current_turn_id is {current_turn_id}...")
            with torch.no_grad():
                current_turn = turn_mask[current_turn_id]
                print(f"{id}'s turn{current_turn_id+1} start and end is ({current_turn['start']},{current_turn['end']})")
                start_chunkid = current_turn["start"] // 5 
                print(f"{id}'s turn{current_turn_id+1} start chunk id is {start_chunkid}")
                if current_turn_id <= len(turn_mask) - 2:
                    next_turn = turn_mask[current_turn_id + 1]
                    print(f"{id}'s next is turn{current_turn_id+1}, start and end is ({next_turn['start']},{next_turn['end']})")
                    convert_chunkid = next_turn["start"] // 5
                    end_chunkid = next_turn["end"] // 5 + 1
                    print(f"{id}'s turn{current_turn_id+1} end_chunkid is {end_chunkid}")
                else:
                    end_chunkid = 300
                
                # # get before GT from interleaved_tokens
                # user_chunk_count = 0
                # cutoff_index = len(interleaved_tokens) 
                # for i, token in enumerate(interleaved_tokens_flattened):
                #     if token == "<|user|>":
                #         user_chunk_count += 1
                #         if user_chunk_count == start_chunkid: 
                #             cutoff_index = i
                #             break
                if not continuation_info or not any(item["current_turn"] == current_turn_id for item in continuation_info):
                        gt_idx = start_chunkid 
                        print("no continuation occurr")
                else:
                    for item in continuation_info:
                        if item["current_turn"] == current_turn_id:
                            gt_idx = item["end_time_of_continuation_response"] // 5
                print(f"{id}'s turn{current_turn_id+1} gt_idx is {gt_idx}")
                input_ids = inputs['input_ids']
                segmentation = {'segmentation':(gt_idx * 0.4 , end_chunkid * 0.4)}
                # get GT context
                current_interleaved_gt = interleaved_tokens[:gt_idx]
                formatted_tokens = self.glm_tokenizer.convert_tokens_to_ids([token for sublist in current_interleaved_gt for token in sublist])
                formatted_tokens = torch.tensor([formatted_tokens], dtype=torch.long).to(self.device)
                input_ids = torch.cat([input_ids, formatted_tokens], dim=1)
                for model_gt_id in range(gt_idx):
                    output_model_audio_tokens.extend(model_tokens_chunks[model_gt_id])
                for i, user_token in enumerate(user_tokens_chunks[gt_idx:end_chunkid],start=gt_idx):  # chunk by chunk prediction
                    # print(f"{id} 's turn{current_turn_id+1}current chunk is {i}")
                    if i < convert_chunkid or  i + 1 >= len(turn_mask):
                        formatted_tokens = ["<|user|>"] + [f"<|audio_{token}|>" for token in user_tokens_chunks[i]] + ["<|assistant|>"]  # Convert all tokens at once
                        user_token_ids = self.glm_tokenizer.convert_tokens_to_ids(formatted_tokens)
                        user_token_ids = torch.tensor([user_token_ids]).to(self.device)
                        input_ids = torch.cat([input_ids, user_token_ids], dim=1)
                        
                        # # Append user token to the sequence
                        # user_token_id = self.glm_tokenizer.convert_tokens_to_ids(f"<|audio_{user_token}|>")
                        # input_ids = torch.cat([input_ids, torch.tensor([[user_token_id]]).to(self.device)], dim=1)

                        if i * chunk_size < num_prompt_tokens:
                            # Append model token to the sequence
                            formatted_tokens = [f"<|audio_{token}|>" for token in model_tokens_chunks[i]]
                            model_token_ids = self.glm_tokenizer.convert_tokens_to_ids(formatted_tokens)
                            model_token_ids = torch.tensor([model_token_ids]).to(self.device)
                            input_ids = torch.cat([input_ids, model_token_ids], dim=1)
                            output_model_audio_tokens.extend(model_tokens_chunks[i])
                            if len(output_model_audio_tokens) == 375:
                                print(f"all model GT tokens are appended!")
                        else:
                            # print(f"input_ids: {input_ids}")
                            # print(f"input_ids.shape: {input_ids.shape}")
                            # print("now generating...")
                            # exit()
                            # Generate one token from the model
                            outputs = self.glm_model.generate(
                                input_ids=input_ids,
                                max_new_tokens=chunk_size,  # predict one chunk of model tokens
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=True,
                                pad_token_id=self.glm_tokenizer.pad_token_id
                            )
                            
                            # Get the newly generated token
                            new_chunk = outputs[0, -chunk_size:].tolist()
                            
                            # Update input_ids for next iteration
                            input_ids = outputs

                            # If it's an audio token, add to our collection
                            audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
                            for new_token in new_chunk:
                                if new_token >= audio_offset:
                                    output_model_audio_tokens.append(new_token - audio_offset)
                    else: 
                        silence_token = user_tokens_chunks[convert_chunkid][-1]
                        # print(silence_token)
                        formatted_tokens = ["<|user|>"] + [f"<|audio_{silence_token}|>" for _ in range(len(user_tokens_chunks[convert_chunkid]))] + ["<|assistant|>"]  # keep silent in next_user_turn
                        user_token_ids = self.glm_tokenizer.convert_tokens_to_ids(formatted_tokens)
                        user_token_ids = torch.tensor([user_token_ids]).to(self.device)
                        input_ids = torch.cat([input_ids, user_token_ids], dim=1)


                        outputs = self.glm_model.generate(
                                input_ids=input_ids,
                                max_new_tokens=chunk_size,  # predict one chunk of model tokens
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=True,
                                pad_token_id=self.glm_tokenizer.pad_token_id
                            )
                            
                        # Get the newly generated token
                        new_chunk = outputs[0, -chunk_size:].tolist()
                        
                        # Update input_ids for next iteration
                        input_ids = outputs

                        # If it's an audio token, add to our collection
                        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
                        for new_token in new_chunk:
                            if new_token >= audio_offset:
                                output_model_audio_tokens.append(new_token - audio_offset)
        # print("===================")
        # # print(f"user_audio_tokens: {user_audio_tokens}")
        # # print(f"output_model_audio_tokens: {output_model_audio_tokens}")
        # print(f"user_audio_tokens len: {len(user_audio_tokens)}")
        # print(f"output_model_audio_tokens len: {len(output_model_audio_tokens)}")
        # the total length should exceed what is required
            user_audio_tokens_turned = user_audio_tokens[:len(output_model_audio_tokens)]
            output_model_audio_tokens = output_model_audio_tokens[:int(12.5*total_seconds)]
            # print("after trimming:")
            print(f"user_audio_tokens len: {len(user_audio_tokens_turned)}")
            print(f"{id} turn{current_turn_id+1}'s output_model_audio_tokens len: {len(output_model_audio_tokens)}")

            # print(f"output_model_audio_tokens: {output_model_audio_tokens}")
            # if len(output_model_audio_tokens) != len(user_audio_tokens):
            #     print("User and model audio tokens must be of the same length")
            #     return [], [], "len_mismatch"
            # assert len(output_model_audio_tokens) == len(user_audio_tokens), "User and model audio tokens must be of the same length"
            
            # print(f"output_model_audio_tokens: {len(output_model_audio_tokens)} tokens")
            # decoded_text = self.glm_tokenizer.decode(output_model_audio_tokens)
            # print(decoded_text)

            # Convert model audio tokens to waveform
            if output_model_audio_tokens:
                model_audio_tensor = torch.tensor(output_model_audio_tokens, device=self.device).unsqueeze(0)
                this_uuid = str(uuid.uuid4())
                model_speech, _ = self.audio_decoder.token2wav(
                    model_audio_tensor,
                    uuid=this_uuid,
                    prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                    prompt_feat=torch.zeros(1, 0, 80).to(self.device)
                )
                
                # Convert user tokens to waveform for comparison/reference
                user_audio_tensor = torch.tensor(user_audio_tokens_turned, device=self.device).unsqueeze(0)
                user_speech, _ = self.audio_decoder.token2wav(
                    user_audio_tensor,
                    uuid=str(uuid.uuid4()),
                    prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                    prompt_feat=torch.zeros(1, 0, 80).to(self.device)
                )
                
                # Save both channels if output path is provided
                if output_path:
                    # # Save model speech
                    # model_output_path = output_path.replace('.wav', '_model.wav')
                    # torchaudio.save(model_output_path, model_speech.cpu(), 22050)
                    
                    # # Save user speech
                    # user_output_path = output_path.replace('.wav', '_user.wav')
                    # torchaudio.save(user_output_path, user_speech.cpu(), 22050)
                    
                    # Create a stereo file with both channels
                    # stereo_output_path = output_path.replace('.wav', f'_stereo.wav')

                    os.makedirs(os.path.dirname(f"{stereo_output_path}/turn{current_turn_id+1}/{id.replace('.wav', '')}_stereo.wav"), exist_ok=True)
                    self._create_stereo_file(user_speech.cpu(), model_speech.cpu(), f"{stereo_output_path}/turn{current_turn_id+1}/{id.replace('.wav', '')}_stereo.wav")
                    segmentation_output_path = f"{stereo_output_path}/turn{current_turn_id+1}/{id.replace('.wav', '')}_stereo.jsonl"
                    # get segmentation
                    with open(segmentation_output_path,'w',encoding='utf-8') as f:
                        if not continuation_info or not any(item["current_turn"] == current_turn_id for item in continuation_info):
                            f.write(json.dumps(segmentation, ensure_ascii = False))
                        else:
                            for item in continuation_info:
                                if item["current_turn"] == current_turn_id:
                                    f.write(json.dumps(segmentation, ensure_ascii = False))
            else:
                print("No model audio tokens were generated")
                return None, None, "no_audio"
        return model_speech, user_speech, "normal"
    def generate_unconditional_interleaved_speech_token_only(
        self,
        input_audio_path,
        user_audio_tokens,
        model_audio_tokens,
        temperature=0.2,
        top_p=0.8,
        output_path=None, 
        prompt_seconds=30,  # seconds of audio used as prompt (must be an even number!)
        total_seconds=120,
    ):
        """
        Process speech in an interleaved fashion to create full duplex dialogue:
        1. the user's audio is already extracted into tokens
        2. For each user token, append it to sequence and generate one model token
        3. Continue until all user tokens are used
        4. Decode model tokens to speech
        """
        stereo_output_path = output_path.replace('.wav', '_stereo.wav')
        if os.path.exists(stereo_output_path):
            print(f"Stereo output file {stereo_output_path} already exists. Skipping generation.")
            return None, None, "exist"
        
        print(f"User audio tokens extracted: {len(user_audio_tokens)} tokens")
        
        # Prepare initial prompt
        prompt = f"<|begin_of_audio|><|begin_of_audio|>"

        num_prompt_tokens = int(prompt_seconds * 12.5)  # GLM-4-Voice uses 12.5 tokens per second
        assert num_prompt_tokens % chunk_size == 0, "cannot include integer number of chunk_size as prompt (i.e., num_prompt_tokens % chunk_size != 0)"
        num_prompt_chunks = num_prompt_tokens // chunk_size
        assert len(user_audio_tokens) == len(model_audio_tokens), "User and model audio tokens must be of the same length"
        assert num_prompt_tokens <= len(user_audio_tokens), "Not enough user audio tokens for the prompt"
        user_tokens_chunks = [user_audio_tokens[i:i+chunk_size] for i in range(0, len(user_audio_tokens), chunk_size)]
        model_tokens_chunks = [model_audio_tokens[i:i+chunk_size] for i in range(0, len(model_audio_tokens), chunk_size)]
        prompt_audio_tokens = ""
        output_user_audio_tokens, output_model_audio_tokens = [], []
        for i in range(num_prompt_chunks):  # for i in range(num_prompt_tokens)
            prompt_audio_tokens += "<|user|>"
            for token in user_tokens_chunks[i]:
                prompt_audio_tokens += f"<|audio_{token}|>"
                output_user_audio_tokens.append(token)
            prompt_audio_tokens += "<|assistant|>"
            for token in model_tokens_chunks[i]:
                prompt_audio_tokens += f"<|audio_{token}|>"
                output_model_audio_tokens.append(token)
        final_prompt = prompt + prompt_audio_tokens  # add the prompt audio tokens to the prompt
        # print("\n=========================\n")
        # print(f"final_prompt: {final_prompt}")
        print(f"output_user_audio_tokens_len: {len(output_user_audio_tokens)}")
        print(f"output_model_audio_tokens_len: {len(output_model_audio_tokens)}")
        
        # FIXME: modify from here
        # FIXME: if chunk_size=5, then during inference, we need to make sure the next 10 tokens (excluding the 2 speaker tokens) are speech tokens, so that we split them by 5+5
        # FIXME: if we only get for example 6 tokens, then we split them by 3+3, and then append the end_of_audio tokens
        # Tokenize initial prompt
        inputs = self.glm_tokenizer([final_prompt], add_special_tokens=False, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        # print(f"prompt input_ids: {input_ids}")

        print(f"num_prompt_tokens: {num_prompt_tokens}")

        # # Tokenize initial prompt
        # inputs = self.glm_tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(self.device)
        # input_ids = inputs['input_ids']
        # print(f"input_ids: {input_ids}")
        # print(f"input_ids.shape: {input_ids.shape}")
        
        # Process tokens in an interleaved fashion
        if split_generation:
            print(f"Generating interleaved dialogue (part {part})...")
        with torch.no_grad():
            inputs = inputs
            input_length = inputs['input_ids'].shape[-1]
            outputs = self.glm_model.generate(
                **inputs,
                max_new_tokens=int((12.5*90*2/5)*6+100),  # 12.5Hz * 90s * 2 channels = 2250, but we need to add speaker id tokens, so (2250/5)*6=2700 if chunk_size=5, and we add 100 additional tokens to avoid exception
                temperature=temperature,
                top_p=top_p, 
                eos_token_id=None  # Disable EOS token stopping
            )
            outputs = outputs[:, input_length:]  # Remove input tokens
        print(f"outputs.shape: {outputs.shape}")

        # Separate text and audio tokens
        generated_tokens = outputs[0].tolist()
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        end_of_audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|end_of_audio|>')
        temp_user_audio_tokens, temp_model_audio_tokens = [], []
        # chunk-level segmentation
        generated_tokens_chunk = [generated_tokens[i:i+chunk_size+1] for i in range(0, len(generated_tokens), chunk_size+1)]  # chunk_size+1 because one additional token for speaker id
        for i, chunk in enumerate(generated_tokens_chunk):
            if chunk[0] >= audio_offset:  # the first token (speaker id) should be less than the first audio token id
                break
            if i % 2 == 0:  # Even indices (0, 2, 4, ...) for user tokens
                temp_user_audio_tokens.extend(chunk[1:])  # omit the first token: the speaker id
            else:  # Odd indices (1, 3, 5, ...) for assistant tokens
                temp_model_audio_tokens.extend(chunk[1:])  # omit the first token: the speaker id
        # adjust the length of the two lists to be the same
        if len(temp_user_audio_tokens) <= len(temp_model_audio_tokens):
            temp_model_audio_tokens = temp_model_audio_tokens[:len(temp_user_audio_tokens)]
        else:
            temp_user_audio_tokens = temp_user_audio_tokens[:len(temp_model_audio_tokens)]
        # print(f"temp_user_audio_tokens: {temp_user_audio_tokens}")
        # print(f"temp_model_audio_tokens: {temp_model_audio_tokens}")

        for output_i in range(len(temp_model_audio_tokens)):
            if temp_user_audio_tokens[output_i] == end_of_audio_offset or temp_model_audio_tokens[output_i] == end_of_audio_offset:
                print(f"encountered end_of_audio_offset {end_of_audio_offset}, breaking the loop!")
                break

            if temp_user_audio_tokens[output_i] >= audio_offset and temp_model_audio_tokens[output_i] >= audio_offset:
                output_user_audio_tokens.append(temp_user_audio_tokens[output_i] - audio_offset)
                output_model_audio_tokens.append(temp_model_audio_tokens[output_i] - audio_offset)
            else:
                print(f"encountered non-audio token, breaking the loop! (audio offset: {audio_offset}, end_of_audio_offset: {end_of_audio_offset})")
                break

        print("===================")
        # print(f"output_user_audio_tokens: {output_user_audio_tokens}")
        # print(f"output_model_audio_tokens: {output_model_audio_tokens}")
        print(f"output_user_audio_tokens len: {len(output_user_audio_tokens)}")
        print(f"output_model_audio_tokens len: {len(output_model_audio_tokens)}")
        # the total length should exceed what is required
        output_user_audio_tokens = output_user_audio_tokens[:int(12.5*total_seconds)]
        output_model_audio_tokens = output_model_audio_tokens[:int(12.5*total_seconds)]
        print("after trimming:")
        print(f"output_user_audio_tokens len: {len(output_user_audio_tokens)}")
        print(f"output_model_audio_tokens len: {len(output_model_audio_tokens)}")

        if len(output_model_audio_tokens) != len(user_audio_tokens):
            print("User and model audio tokens must be of the same length")
            return [], [], "len_mismatch"
        # assert len(output_user_audio_tokens) == len(output_model_audio_tokens), "User and model audio tokens must be of the same length"

        # Convert model audio tokens to waveform
        if output_model_audio_tokens:
            model_audio_tensor = torch.tensor(output_model_audio_tokens, device=self.device).unsqueeze(0)
            this_uuid = str(uuid.uuid4())
            model_speech, _ = self.audio_decoder.token2wav(
                model_audio_tensor,
                uuid=this_uuid,
                prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                prompt_feat=torch.zeros(1, 0, 80).to(self.device)
            )
            
            # Convert user tokens to waveform for comparison/reference
            user_audio_tensor = torch.tensor(output_user_audio_tokens, device=self.device).unsqueeze(0)
            user_speech, _ = self.audio_decoder.token2wav(
                user_audio_tensor,
                uuid=str(uuid.uuid4()),
                prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
                prompt_feat=torch.zeros(1, 0, 80).to(self.device)
            )
            
            # Save both channels if output path is provided
            if output_path:
                # # Save model speech
                # model_output_path = output_path.replace('.wav', '_model.wav')
                # torchaudio.save(model_output_path, model_speech.cpu(), 22050)
                
                # # Save user speech
                # user_output_path = output_path.replace('.wav', '_user.wav')
                # torchaudio.save(user_output_path, user_speech.cpu(), 22050)
                
                # Create a stereo file with both channels
                stereo_output_path = output_path.replace('.wav', '_stereo.wav')
                self._create_stereo_file(user_speech.cpu(), model_speech.cpu(), stereo_output_path)
                    
                
            return model_speech, user_speech, "normal"
        else:
            print("No model audio tokens were generated")
            return None, None, "no_audio"

    def _create_stereo_file(self, user_audio, model_audio, output_path, sample_rate=22050):
        """
        Create a stereo audio file with user audio in left channel and model audio in right channel.
        Pads the shorter audio to match the length of the longer one.
        """
        # Get lengths
        user_len = user_audio.shape[1]
        model_len = model_audio.shape[1]
        max_len = max(user_len, model_len)
        
        # Pad to same length
        if user_len < max_len:
            user_audio = torch.nn.functional.pad(user_audio, (0, max_len - user_len))
        if model_len < max_len:
            model_audio = torch.nn.functional.pad(model_audio, (0, max_len - model_len))
        
        # Create stereo audio
        stereo_audio = torch.stack([user_audio[0], model_audio[0]], dim=0)
        torchaudio.save(output_path, stereo_audio, sample_rate)
        
        print(f"Saved stereo audio to {output_path}")

    def audio_reconstruction(self, input_audio_path):
        # first extract the tokens from the audio file
        audio_tokens = extract_speech_token(
            self.whisper_model,
            self.feature_extractor,
            [input_audio_path]
        )[0]
        
        # then reconstruct the audio from the tokens
        audio_tensor = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)
        this_uuid = str(uuid.uuid4())  # Generate a unique ID for the audio file
        tts_speech, _ = self.audio_decoder.token2wav(  # TODO: solve the bug: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
            audio_tensor,
            uuid=this_uuid,
            prompt_token=torch.zeros(1, 0, dtype=torch.int64).to(self.device),
            prompt_feat=torch.zeros(1, 0, 80).to(self.device)
        )
        
        return tts_speech


def run_demo(mode):
    # generate the interleaved speech
    count = 0
    i = -1
    while True:
        i += 1
        if count >= 5:
            break

        user_audio_data = data[i]
        audio_segment = int(user_audio_data['id'].split("_")[2][:3])
        print(f"ID: {user_audio_data['id']}, audio_segment: {audio_segment}")
        if audio_segment > 110:  # only test on the training set yet
            continue
        else:
            count += 1
        
        if mode == "unconditional":
            # unconditional evaluation
            user_audio_tokens = user_audio_data[f'left_audio_ids']
            model_audio_tokens = user_audio_data[f'right_audio_ids']
            # generate the interleaved speech
            model_speech, user_speech = SLM_processor.generate_unconditional_interleaved_speech_token_only(
                input_audio_path=None,
                user_audio_tokens=user_audio_tokens,
                model_audio_tokens=model_audio_tokens,
                temperature=temperature,  # default: 0.2
                top_p=p_value,  # default: 0.8
                output_path=f"{output_folder}/unconditional_interleaved_dialogue_{i}.wav"
            )
        elif mode == "conditional":
            # conditoinal evaluation
            for user_channel in ["left", "right"]:
                if user_channel == "left":
                    model_channel = "right"
                else:
                    model_channel = "left"
                assert user_channel != model_channel

                user_audio_tokens = user_audio_data[f'{user_channel}_audio_ids']
                model_audio_tokens = user_audio_data[f'{model_channel}_audio_ids']
                # generate the interleaved speech
                model_speech, user_speech = SLM_processor.generate_conditional_interleaved_speech_token_only(
                    input_audio_path=None,
                    user_audio_tokens=user_audio_tokens,
                    model_audio_tokens=model_audio_tokens,
                    temperature=temperature,  # default: 0.2
                    top_p=p_value,  # default: 0.8
                    output_path=f"{output_folder}/conditional_interleaved_dialogue_{i}_{user_channel}user.wav"
                )
        else:
            raise ValueError("mode must be either 'unconditional' or 'conditional'")

def unbatch_dict(batch):
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # If it is Tensor , remove batch and transfer to list
            if value.dim() > 1:
                result[key] = value.squeeze(0).tolist()
            else:
                result[key] = value.tolist()
        elif isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], str):
                # first elment got if it is string
                result[key] = value[0]
            elif all(isinstance(item, torch.Tensor) for item in value):
                # if it is Tensor list get every Tensor value and generate list
                result[key] = [item.item() if item.numel() == 1 else item.tolist() for item in value]
            else:
                result[key] = value
        else:
            result[key] = value
    return result



def test_set_eval(mode,SLM_processor,turn_mask_path):
    # load the testset
    # if dataset == "fisher":
    #     testset = data["test"]
    # elif dataset == "condor":
    #     testset = data  # right now all the available condor data are used as testset
    #     eval_num = "all"

    key_error_indexes = []
    special_cases = {
        "normal": 0, 
        "exist": 0, 
        "unsupported_token": 0, 
        "len_mismatch": 0, 
        "no_audio": 0
    }
    if mode == "unconditional":
        for batch in tqdm(dataloader):
            user_audio_data = unbatch_dict(batch)
            # print(user_audio_data)
            print(f"Processing {user_audio_data['id']}...")
            audio_id = user_audio_data["id"].replace(".wav", "").replace(".mp3", "")  # example: fe_03_11431_540_570.wav -> fe_03_11431_540_570
            user_audio_tokens = user_audio_data[f'left_audio_ids']
            model_audio_tokens = user_audio_data[f'right_audio_ids']
            # generate the interleaved speech
            model_speech, user_speech, status = SLM_processor.generate_unconditional_interleaved_speech_token_only(
                input_audio_path=None,
                user_audio_tokens=user_audio_tokens,
                model_audio_tokens=model_audio_tokens,
                temperature=temperature,  # default: 0.2
                top_p=p_value,  # default: 0.8
                output_path=f"{output_folder}/{audio_id}_unconditional.wav"
            )
            special_cases[status] += 1
    elif mode == "conditional":
        for batch in tqdm(dataloader):
            user_audio_data = unbatch_dict(batch)
            # print(user_audio_data)
            print(f"Processing {user_audio_data['id']}...")
            audio_id = user_audio_data["id"].replace(".wav", "").replace(".mp3", "")  # example: fe_03_11431_540_570.wav -> fe_03_11431_540_570
            # for user_channel in ["right","left"]:  # Only generate one setting for now. originally, for user_channel in ["left", "right"]
            #     if user_channel == "left":
            #         model_channel = "right"
            #     else:
            #         model_channel = "left"
            #     assert user_channel != model_channel
            user_channel = "right"
            model_channel = "left"
            assert user_channel != model_channel
            user_audio_tokens = user_audio_data[f'{user_channel}_audio_ids']
            model_audio_tokens = user_audio_data[f'{model_channel}_audio_ids']
            id = user_audio_data["id"]
            # print(id)
            # generate the interleaved speech
            try:
                model_speech, user_speech, status = SLM_processor.generate_conditional_interleaved_speech_token_only(
                    input_audio_path=None,
                    id=id,
                    turn_mask_path=turn_mask_path,
                    user_audio_tokens=user_audio_tokens,
                    model_audio_tokens=model_audio_tokens,
                    temperature=temperature,  # default: 0.2
                    top_p=p_value,  # default: 0.8
                    output_path=f"{output_folder}/{audio_id}_conditional_{user_channel}user"
                )
                special_cases[status] += 1
            except KeyError as e:
                # This block will only execute if a KeyError occurs
                print(f"Key error occurred: {e} for testset index {audio_id}")
                key_error_indexes.append(audio_id)

    key_error_indexes = list(set(key_error_indexes))
    print(key_error_indexes)
    print(f"len of key_error_indexes: {len(key_error_indexes)}")
    print("special cases:")
    print(special_cases)


if __name__ == "__main__":
    # run_demo(mode="unconditional")
    # run_demo(mode="conditional")
    


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_memory_per_gpu", type=str, default="75GB")
    # parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()


    # get current local_rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # setting GPU
    torch.cuda.set_device(local_rank)
    print(f"Current is on GPU{local_rank}")

    SLM_processor = SpeechProcessor(
    tokenizer_path=os.path.join(ckpt_base_path, "glm-4-voice-tokenizer"),
    flow_path=os.path.join(ckpt_base_path, "glm-4-voice-decoder"),
    device = torch.device(f'cuda:{local_rank}'),
    device_map = f"cuda:{local_rank}",
    dtype="bfloat16",  # use fp32 for V100?
    **vars(args)
)
    test_set_eval(mode=condition,SLM_processor=SLM_processor,turn_mask_path=turn_mask_path)
    if dist.is_initialized():
        dist.destroy_process_group()
