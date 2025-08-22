import os
import sys
import uuid
cuda_num = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
sys.path.append('third_party/Matcha-TTS')
sys.path.append("/home/ma-user/work/cuiwenqian/GLM-4-Voice")
import tempfile
from collections import defaultdict
from typing import List, Dict, Any

# audio_path = "/home/ma-user/work/cuiwenqian/GLM-4-Voice/gen_interleaved_dialog/fullfisher_batch4_e10_4e5_checkpoint-6442/realname/unconditional/testset_t1.5_p0.99_noadditionaltokens"
# output_path = "/home/ma-user/work/cuiwenqian/GLM-4-Voice/semantic_eval_log/fullfisher_batch4_e10_4e5_checkpoint-6442/realname/unconditional"
# output_name = os.path.join(output_path, "testset_t1.5_p0.99_noadditionaltokens.json")

audio_path = "/home/ma-user/work/zhanghe/Fisher_GT"
output_path = "/home/ma-user/work/zhanghe/Fisher_GT/"
# output_path = audio_path.replace("/gen_interleaved_dialog/", "/semantic_eval_log/")
dataset_json_file = "/home/ma-user/work/cuiwenqian/GLM-4-Voice/GLM-FisherWavCodes_16k_120s_jsonl/GLM-FisherWavCodes_16k_120s_train_val_test.json"  # fisher
setting = "all"  # NOTE: normally this is an empty string
output_name = output_path + f"divide_turn_{setting}_stable_model.jsonl"
turn_mask_path = "/home/ma-user/work/zhanghe/GLM_Fisher120s_turned/GLM_Fisher120s_semantic_turned_time.jsonl"
turn_mask_path_token = "/home/ma-user/work/zhanghe/GLM_Fisher120s_turned/GLM_Fisher120s_semantic_turned_token.jsonl"
condition = "conditional"
context = "all"  # "all" or "right". "right" means only the model channel's output is given to GPT for judging.
speaker_map = {
    "left": "user", 
    "right": "model"
}
prompt_seconds = 30
all_seconds = 120
voting_num = 6
min_votes_num = 2
chunk_size = 5

os.makedirs(output_path, exist_ok=True)
# temp_audio_name = "temp_audio1.wav"

import torchaudio
import soundfile as sf
import librosa
import torch
import pandas as pd
from typing import Optional, List
import time
from tqdm import tqdm
import numpy as np
import scipy.io.wavfile as wavfile
import wave
import copy
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM
import whisper_timestamped as whisper
from openai import OpenAI
import re
import glob
import json
from flow_inference import AudioDecoder


#create GT wav file

# with open(dataset_json_file, "r") as f:
#     data = json.load(f)
# testset = data["test"][0:20]
# user_channel = "right"
# model_channel = "left"
# stereo_output_path_dir = "/home/ma-user/work/zhanghe/Fisher_GT"
# flow_path=os.path.join("/home/ma-user/work/cuiwenqian/hf_model_ckpt/GLM-4-Voice", "glm-4-voice-decoder")
# flow_config = os.path.join(flow_path, "config.yaml")
# flow_checkpoint = os.path.join(flow_path, 'flow.pt')
# hift_checkpoint = os.path.join(flow_path, 'hift.pt')
# audio_decoder = AudioDecoder(
#     config_path=flow_config,
#     flow_ckpt_path=flow_checkpoint,
#     hift_ckpt_path=hift_checkpoint,
#     device="cuda"
# )

# def create_stereo_file(user_audio, model_audio, output_path, sample_rate=22050):
#         """
#         Create a stereo audio file with user audio in left channel and model audio in right channel.
#         Pads the shorter audio to match the length of the longer one.
#         """
#         # Get lengths
#         user_len = user_audio.shape[1]
#         model_len = model_audio.shape[1]
#         max_len = max(user_len, model_len)
        
#         # Pad to same length
#         if user_len < max_len:
#             user_audio = torch.nn.functional.pad(user_audio, (0, max_len - user_len))
#         if model_len < max_len:
#             model_audio = torch.nn.functional.pad(model_audio, (0, max_len - model_len))
        
#         # Create stereo audio
#         stereo_audio = torch.stack([user_audio[0], model_audio[0]], dim=0)
#         torchaudio.save(output_path, stereo_audio, sample_rate)
        
#         print(f"Saved stereo audio to {output_path}")


# for i in range(20):
#     user_audio_data = testset[i]
#     user_audio_data_id = user_audio_data["id"]
#     stereo_output_path = f"{stereo_output_path_dir}/{user_audio_data_id}"
#     user_audio_tokens = user_audio_data[f'{user_channel}_audio_ids']
#     model_audio_tokens = user_audio_data[f'{model_channel}_audio_ids']
#     model_audio_tensor = torch.tensor(model_audio_tokens, device="cuda").unsqueeze(0)
#     this_uuid = str(uuid.uuid4())
#     model_speech, _ = audio_decoder.token2wav(
#         model_audio_tensor,
#         uuid=this_uuid,
#         prompt_token=torch.zeros(1, 0, dtype=torch.int64).to("cuda"),
#         prompt_feat=torch.zeros(1, 0, 80).to("cuda")
#     )
#     # Convert user tokens to waveform for comparison/reference
#     user_audio_tensor = torch.tensor(user_audio_tokens, device="cuda").unsqueeze(0)
#     user_speech, _ = audio_decoder.token2wav(
#         user_audio_tensor,
#         uuid=str(uuid.uuid4()),
#         prompt_token=torch.zeros(1, 0, dtype=torch.int64).to("cuda"),
#         prompt_feat=torch.zeros(1, 0, 80).to("cuda")
#     )
#     create_stereo_file(user_speech.cpu(), model_speech.cpu(), stereo_output_path)

def majority_voting_turns(results_list: List[Dict], overlap_threshold: float = 0.5):
    """
    Use majority voting to get stable turn segmentation results
    
    Parameters:
    results_list: list of turn segmentation results from multiple runs
    overlap_threshold: minimum overlap ratio to consider two turns as the same (0.0-1.0)
    
    Returns:
    dict containing stable filtered_turn and continuation_info
    """
    
    def calculate_overlap(turn1, turn2):
        """Calculate overlap ratio between two turns"""
        start1, end1 = turn1['start'], turn1['end']
        start2, end2 = turn2['start'], turn2['end']
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union_duration = union_end - union_start
        
        return overlap_duration / union_duration if union_duration > 0 else 0
    
    def merge_similar_turns(turns, threshold):
        """Merge turns that have high overlap"""
        merged_turns = []
        
        for turn in turns:
            merged = False
            for i, existing_turn in enumerate(merged_turns):
                if calculate_overlap(turn, existing_turn) >= threshold:
                    # Merge by taking average boundaries
                    merged_turns[i] = {
                        'start': (turn['start'] + existing_turn['start']) / 2,
                        'end': (turn['end'] + existing_turn['end']) / 2,
                        'count': existing_turn.get('count', 1) + 1
                    }
                    merged = True
                    break
            
            if not merged:
                merged_turns.append({
                    'start': turn['start'],
                    'end': turn['end'],
                    'count': 1
                })
        
        return merged_turns
    
    def vote_for_turns(all_turns):
        """Apply majority voting for turn boundaries"""
        # Collect all turns from all results
        all_turn_candidates = []
        for result in all_turns:
            if 'filtered_turn' in result:
                all_turn_candidates.extend(result['filtered_turn'])
        
        if not all_turn_candidates:
            return []
        
        # Merge similar turns
        merged_turns = merge_similar_turns(all_turn_candidates, overlap_threshold)
        
        # Filter turns that appear in majority of results
        min_votes = min_votes_num  # Majority threshold
        stable_turns = []
        
        for turn in merged_turns:
            if turn['count'] >= min_votes:
                stable_turns.append({
                    'start': round(turn['start'], 2),
                    'end': round(turn['end'], 2)
                })
        
        # Sort by start time
        stable_turns.sort(key=lambda x: x['start'])
        return stable_turns
    
    def vote_for_continuations(all_continuations):
        """Apply majority voting for continuation info"""
        # Collect all continuation info
        continuation_candidates = defaultdict(list)
        
        for result in all_continuations:
            if 'continuation_info' in result:
                for cont in result['continuation_info']:
                    turn_id = cont['current_turn']
                    continuation_candidates[turn_id].append(cont)
        
        stable_continuations = []
        min_votes = min_votes_num
        
        for turn_id, continuations in continuation_candidates.items():
            if len(continuations) >= min_votes:
                # Average the times for this turn
                avg_start = sum(c['start_time_of_continuation_user_turn'] for c in continuations) / len(continuations)
                avg_end = sum(c['end_time_of_continuation_response'] for c in continuations) / len(continuations)
                
                stable_continuations.append({
                    'current_turn': turn_id,
                    'start_time_of_continuation_user_turn': round(avg_start, 2),
                    'end_time_of_continuation_response': round(avg_end, 2)
                })
        
        # Sort by turn id
        stable_continuations.sort(key=lambda x: x['current_turn'])
        return stable_continuations
    
    # Apply majority voting
    stable_filtered_turn = vote_for_turns(results_list)
    stable_continuation_info = vote_for_continuations(results_list)
    
    return {
        'filtered_turn': stable_filtered_turn,
        'continuation_info': stable_continuation_info
    }



gpt_prompt = """To analyze the semantic content of the following full-duplex two-speaker dialogue transcript, focusing only on the segment from 30 to 120 seconds, and to accomplish the following tasks:
1. Segment the conversation into different turns based on the semantic content of both channels,You need to comprehensively consider the definition of the start and end times of a turn based on the type and style of the conversation. This includes determining which criteria to use as reference points, such as whether the topic has concluded, the termination of several consecutive sentences, the length of the model speaker's response, and so on.Use the start and end of the user speaker as the dividing point.If the model response is a backchannel (such as interjections, thank you, hmm, etc., which do not carry semantic information) or an unrecognized part (such as a completely irrelevant topic during the conversation), skip and do not record that dividing point. The output data format should be:
{
"filtered_turn": [
{"start": 0, "end": 15},
{"start": 105, "end": 125},
{"start": 175, "end": 185},
{"start": 225, "end": 240},
{"start": 270, "end": 290},
{"start": 455, "end": 505},
{"start": 575, "end": 620}
]
}
2. After segmenting the turns, determine if the model channel has a continuation of the response (i.e., the model's response from the previous turn continues into the current turn).If there is a continuation, return the start time of the user turn where the continuation begins and the end time of the continuation. The output data format should be:
{
"current_turn": "current turn number",
"start_time_of_continuation_user_turn": "XX",
"end_time_of_continuation_response": "XX"
}
3. Integrate the data returned from the first two tasks and label them separately to distinguish between the tasks.Return the integrated data.
Here is an example of how the integrated data might look:
{
"filtered_turn": [
{"start": 0, "end": 15},
{"start": 105, "end": 125},
{"start": 175, "end": 185},
{"start": 225", "end": 240},
{"start": 270, "end": 290},
{"start": 455, "end": 505},
{"start": 575, "end": 620}
],
"continuation_info": [
{
"current_turn": 1,
"start_time_of_continuation_user_turn": 120,
"end_time_of_continuation_response": 130
}
]
}
You need to forget the memories of having handled similar tasks before.Only output the final integrated data(json format) **ONLY** according to the above rubric. Do not output anything else."""

# load the openai api client
client = OpenAI(
    api_key="sk-I8QuRwLK5zYMFiusB574D7781504425e821cF0F22dCf77Df",
    base_url="http://api.xunxkj.cn/v1"  # Optional: only needed if using custom endpoint
)

model = whisper.load_model("medium.en", download_root="/home/ma-user/work/cuiwenqian/hf_model_ckpt/whisper")
print("Whisper model loaded")

def whisper_ts_inference(waveform, sample_rate):
    # Create a temporary file with a unique name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_audio_path = temp_file.name
        torchaudio.save(temp_audio_path, waveform.unsqueeze(0), sample_rate)
        audio = whisper.load_audio(temp_audio_path)
        reformatted_segments = []
        result = whisper.transcribe(
                    model, 
                    audio,
                    beam_size=5,
                    best_of=5, 
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    vad=False,
                    detect_disfluencies=False,
                    language="en",
                    include_punctuation_in_confidence=False
                )
        segments = result["segments"]
        for segment in segments:
            reformatted_segments.append({'timestamp': (segment["start"], segment["end"]), 'text': segment["text"]})

    # Clean up the temporary file after use
    os.unlink(temp_audio_path)
    
    return reformatted_segments


def asr_ts_on_stereo(input_file):
    # Load stereo audio
    waveform, sample_rate = torchaudio.load(input_file)  # waveform shape: [channels, num_samples]
    new_sample_rate = 16000
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    resampled_waveform = resampler(waveform)
    
    # Split into left and right channels
    left_channel_wav = resampled_waveform[0, :]
    right_channel_wav = resampled_waveform[1, :] if resampled_waveform.shape[0] > 1 else None

    # Perform ASR
    left_corrected_segments = whisper_ts_inference(left_channel_wav, new_sample_rate)
    right_corrected_segments = whisper_ts_inference(right_channel_wav, new_sample_rate) if right_channel_wav is not None else None

    return left_corrected_segments, right_corrected_segments

def time_range_to_token_range(start_time, end_time, tokens_per_second=12.5):
        """
        Convert time range to token range.
        
        Args:
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            tokens_per_second (float): Number of tokens per second (default: 12.5)
        
        Returns:
            tuple: (start_token_index, end_token_index)
        """
        start_token = int(start_time * tokens_per_second)
        end_token = int(end_time * tokens_per_second)
        return start_token, end_token

def align_to_chunk_boundary(token_idx, is_start=True,chunk_size=chunk_size):
    """
    Align token index to chunk boundary (multiple of 5).
    For start: round down (e.g., 13 -> 10)
    For end: round up (e.g., 13 -> 15)
    """
    if is_start:
        return (token_idx // chunk_size) * chunk_size  # Round down
    else:
        return ((token_idx + chunk_size - 1) // chunk_size) * chunk_size  # Round up


def merge_stereo_segments(left_segments, right_segments):
    tagged_left = [
        {'speaker': speaker_map['left'], 'timestamp': seg['timestamp'], 'text': seg['text']}
        for seg in left_segments
    ]
    tagged_right = [
        {'speaker': speaker_map['right'], 'timestamp': seg['timestamp'], 'text': seg['text']}
        for seg in right_segments
    ]
   
    # Merge and sort by start time
    if context == "all":
        merged = tagged_left + tagged_right
    elif context == "right":
        merged = tagged_right
    else:
        raise Exception("unsupported context channel!")
    merged.sort(key=lambda x: x['timestamp'][0])
    # Format output
    lines = []
    texts = []
    for seg in merged:
        speaker = seg['speaker']
        start, end = seg['timestamp']
        text = seg['text']
        line = f"{speaker} speaker, start/end time: ({start:.2f}, {end:.2f}), content: {text}"
        texts.append(text)
        lines.append(line)
    return "\n".join(lines),texts


# Find all .wav files in the audio_path
wav_files = glob.glob(os.path.join(audio_path, "*.wav"))
intergrated_datas = {}
# Process each .wav file
for input_file in tqdm(wav_files):
    print(f"Processing {input_file} with context: {context}")
    filename = input_file.split("/")[-1]
    left_timestamps, right_timestamps = asr_ts_on_stereo(input_file)
    merged_output , merged_text = merge_stereo_segments(left_timestamps, right_timestamps)
    # print(f"{merged_output}\n\n{gpt_prompt}")
    # exit()
    # List to store intergrated data
    extracted_data = []
    result_data = {}
    intergrated_data = []
    # convert to normalized data
    normalized_datas = {}
    normalized_datas_token = {}
    try:
        for i in range(voting_num):
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{merged_output}\n\n{gpt_prompt}",
                    }
                ],
                model="gpt-4o",
            )
            extracted_data = chat_completion.choices[0].message.content
            match = re.search(r'```json\s*(.*?)\s*```', extracted_data, re.DOTALL)
            if match:
                result_dict=json.loads(match.group(1))
            else:
                result_dict = json.loads(extracted_data)
            print(f"Extracted intergrated data for {input_file}: {result_dict}")
            intergrated_data.append(result_dict)
        result_data = majority_voting_turns(intergrated_data, overlap_threshold=0.3)
        print(f"Extracted intergrated data for {input_file}: {result_data}")
        intergrated_datas[filename] = {
            "transcription": merged_text, 
            "data": result_data
        }
        normalized_datas = {
            "id":filename,
            "filtered_turn":result_data["filtered_turn"],
            "continuation_info":result_data["continuation_info"]
        }
        result_data_copy = copy.deepcopy(result_data)
        for item_filtered_turn in result_data_copy["filtered_turn"]:
            item_filtered_turn["start"],item_filtered_turn["end"] = time_range_to_token_range(item_filtered_turn["start"],item_filtered_turn["end"])
            item_filtered_turn["start"] = align_to_chunk_boundary(item_filtered_turn["start"])
            item_filtered_turn["end"] = align_to_chunk_boundary(item_filtered_turn["end"],False)

        for item_continuation_info in result_data_copy["continuation_info"]:
            item_continuation_info["start_time_of_continuation_user_turn"],item_continuation_info["end_time_of_continuation_response"] = time_range_to_token_range(item_continuation_info["start_time_of_continuation_user_turn"],item_continuation_info["end_time_of_continuation_response"])
            item_continuation_info["start_time_of_continuation_user_turn"] = align_to_chunk_boundary(item_continuation_info["start_time_of_continuation_user_turn"])
            item_continuation_info["end_time_of_continuation_response"] = align_to_chunk_boundary(item_continuation_info["end_time_of_continuation_response"],False)
        normalized_datas_token = {
            "id":filename,
            "filtered_turn":result_data_copy["filtered_turn"],
            "continuation_info":result_data_copy["continuation_info"]
        }
        # print(f"extracted normalized_datas_token:{normalized_datas_token}")
        # get time version of semantic turned result
        with open(turn_mask_path,"w",encoding="utf-8") as f:
            f.write(json.dumps(normalized_datas, ensure_ascii=False) + '\n')

        # get token version of semantic turned result
        with open(turn_mask_path_token,"w",encoding="utf-8") as f:
            f.write(json.dumps(normalized_datas_token, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

         
        # # For more detailed error information, use:
        # traceback.print_exc()
with open(output_name, "w", encoding="utf-8") as json_file:
    json.dump(intergrated_datas, json_file, indent=4)
