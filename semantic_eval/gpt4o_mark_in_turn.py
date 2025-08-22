# Use GPT-4o to pass the turn segments divided by semantics to GPT-4o's turn_mask, allowing it to generate a separate score based on the newly generated turn parts. This will enable a fine-grained comparison of the differences in scores between the baseline and v2, and design methods to verify the effectiveness of v2.
audio_path = "/home/ma-user/work/zhanghe/GLM_Fisher120s_turned/all_gen_turned_wav.jsonl"
turn_mask_path = "/home/ma-user/work/zhanghe/GLM_Fisher120s_turned/GLM_Fisher120s_turned.jsonl"
import json
prompt_seconds = 30
all_seconds = 120
from openai import OpenAI



with open(turn_mask_path,'r',encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line.strip())
            print(data['id'])

with open(audio_path,'r',encoding='utf-8') as f:
    wav_files = json.load(f)
    # print(wav_files)

total_mark_mask = {}

# load the openai api client
client = OpenAI(
    api_key="sk-I8QuRwLK5zYMFiusB574D7781504425e821cF0F22dCf77Df",
    base_url="http://api.xunxkj.cn/v1"  # Optional: only needed if using custom endpoint
)

# fianl output data
# {
#     "fe_03_11558_300_420.wav":[{"turned":{"turn1":2,"turn2":3}},{"total":{"turn1":2,"turn2":3}}],
#     "fe_03_11586_60_180.wav":[{"turned":{"turn1":2,"turn2":3}},{"total":{"turn1":2,"turn2":3}}],
#      ... 
#     }


for input_file in wav_files:
    # find current wav file's turn_mask
    current_wav_id,current_turn_id = input_file.split('/')[-2:]
    current_turn_id = current_turn_id.replace('turn','')
    for item in data:
        if item['id']==current_wav_id:
            current_turn_mask = item['filtered_turn']
            current_turn_timestamp_start = item['filtered_turn'][current_turn_id - 1]['start']
            next_turn_timestamp_end = item['filtered_turn'][current_turn_id]['end']
            timestamp = {'timestamp':(current_turn_timestamp_start,next_turn_timestamp_end)}
            gpt_prompt_turned = f"""Please evaluate the following two-speaker dialogue transcript for how meaningful the speech is (based on its content), only focusing on the model channel's output and from {current_turn_timestamp_start} seconds to {next_turn_timestamp_end} seconds of the transcript. Use the following scale:

0: Completely meaningless; no coherent sentences, random words, or unintelligible.
0.5: Almost no meaning; isolated words or phrases, but no understandable ideas.
1: Extremely low meaning; rare, vague fragments of ideas, but mostly incoherent or off-topic.
1.5: Very little meaning; a few short, unclear ideas, but mostly disjointed or confusing.
2: Low meaning; some recognizable ideas or topics, but mostly unclear, incomplete, or off-topic.
2.5: Somewhat low meaning; a few coherent points, but overall lacks clarity or logical flow.
3: Moderate meaning; general topic is understandable, but there are gaps, unclear parts, or weak connections.
3.5: Fairly meaningful; mostly coherent and relevant, but with some confusion, repetition, or lack of detail.
4: Meaningful; clear and logical, with relevant and connected ideas, though may lack depth or detail.
4.5: Very meaningful; almost fully coherent, with well-developed, relevant, and connected ideas.
5: Extremely meaningful; highly coherent, clear, and detailed, with all ideas well connected and relevant.

Only output the final score (0, 0.5, 1, 1.5, ..., 5) **ONLY** according to the above rubric. Do not output anything else."""
    
    merged_output_turned

    chat_completion_turned = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{merged_output_turned}\n\n{gpt_prompt_turned}",
                }
            ],
            model="gpt-4o",
        )
    
    extracted_score[input_file.split('/')[-2]] = chat_completion_turned.choices[0].message.content    # 'turn1':2

    # total input file mark
    if total_mark_mask[current_wav_id] == 0:
    turn_mask = {}
    turn_id = 0
    for turn in current_turn_mask:
        turn_id += 1
        current_turn = f'turn{turn_id}'
        turn_mask[current_turn] = turn
    gpt_prompt_total = f"""Please evaluate the following two-speaker dialogue transcript and list of different turns of the entire dialogue for how meaningful each turn of the speech is (based on its content), only focusing on the model channel's output and from {prompt_seconds} seconds to {all_seconds} seconds of the transcript. Use the following scale:

0: Completely meaningless; no coherent sentences, random words, or unintelligible.
0.5: Almost no meaning; isolated words or phrases, but no understandable ideas.
1: Extremely low meaning; rare, vague fragments of ideas, but mostly incoherent or off-topic.
1.5: Very little meaning; a few short, unclear ideas, but mostly disjointed or confusing.
2: Low meaning; some recognizable ideas or topics, but mostly unclear, incomplete, or off-topic.
2.5: Somewhat low meaning; a few coherent points, but overall lacks clarity or logical flow.
3: Moderate meaning; general topic is understandable, but there are gaps, unclear parts, or weak connections.
3.5: Fairly meaningful; mostly coherent and relevant, but with some confusion, repetition, or lack of detail.
4: Meaningful; clear and logical, with relevant and connected ideas, though may lack depth or detail.
4.5: Very meaningful; almost fully coherent, with well-developed, relevant, and connected ideas.
5: Extremely meaningful; highly coherent, clear, and detailed, with all ideas well connected and relevant.

Only output the final score list (example:{"turn1":2,"turn2":3}) **ONLY** according to the above rubric. Do not output anything else."""
    chat_completion_total = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{merged_output_total}\n\n{gpt_prompt_total}",
                }
            ],
            model="gpt-4o",
        )
    extracted_score_total = chat_completion_total.choices[0].message.content
    # integrate all data

    total_mark_mask[current_wav_id] = 1
