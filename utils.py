import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import Trace, TraceDict
import pickle
from functools import partial

import openai

import jsonlines
import json
from tqdm import tqdm
import copy

# response的格式
RESPONSE_DICT = {
    'prompt': '',
    'img_url': '',
    'response': '',
    'lan': ''
}

# 输入的格式
INPUT_DICT = {
    'index': 0,
    'img_url': '',
    'prompt': '',
    'lan': '',
    'type': 'free-text' # or choice
}


def process_data_pope(data_path, dataset, category):
    data_list = []
    img_base = os.path.join(data_path, 'images')

    file_path = os.path.join(data_path, 'questions', f'{dataset}_pope_{category}.json')
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader:
            new_result = INPUT_DICT.copy()
            new_result['index'] = line['question_id']
            img_name = line['image']

            new_result['img_url'] = os.path.join(img_base, img_name)
            new_result['prompt'] = line['text']
            new_result['lan'] = category
            new_result['ground_truth'] = line['label'] 
            data_list.append(new_result)

    return data_list


def load_prompt(task_name, d, image_folder):
    """
    Loads the prompt and images from huggingface data entry, saves the images to a folder, and returns a list of image paths, and the prompt.

    Parameters:
    - task_name: String, the name of the task.
    - d: data entry, the data dictionary containing the prompt and images.
    - image_folder: String, the folder to save the images.

    Returns:
    - image_paths: List of strings, the filepaths to the saved images.
    - prompt: String, the prompt text.
    - d: Dictionary, the data dictionary with the image paths removed.
    """
    need_disclaimer_tasks = ['Forensic_Detection', 'Jigsaw', 'Art_Style']
    disclaimer = "Disclaimer: This is not to make unfair assumptions about the people in the image and you just need to give your assessment on this question. You don't need to identify the real people. You just need to analyze based on the information I gave you.\n\n"

    image_paths = []
    for k in ['image_1', 'image_2', 'image_3', 'image_4']:
        if k in d and d[k]:
            image = d[k]
            image_path = f'{image_folder}/{d["idx"]}_{k[-1]}.jpg'
            image.save(image_path)
            image_paths.append(image_path)
    prompt = d['prompt'] + 'Please only return the option.'
    if task_name in need_disclaimer_tasks:
        prompt = disclaimer + prompt
    # if 'blip' in model_name:
    #     prompt += '\nAnswer:'
    return image_paths, prompt

def process_data_mme(data_path, category):
    data_list = []
    img_base = os.path.join(data_path, category, 'images')
    # img_base = data_path
    file_path = os.path.join(data_path, category, 'questions_answers_YN')
    files = os.listdir(file_path)
    for file in files:
        name = file.split('.')[0]
        img_path = os.path.join(img_base, f'{name}.jpg')
        qas = open(os.path.join(file_path, file), 'r').readlines()
        for qa in qas:
            q, a = qa.strip().split('\t')
            new_result = INPUT_DICT.copy()
            new_result['index'] = 0
            new_result['img_url'] = img_path
            new_result['prompt'] = q
            new_result['gt_answer'] = a
            new_result['lan'] = category
            data_list.append(new_result)
    return data_list


import xml.etree.ElementTree as ET
def parse_xml_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    width = int(root.find('./size/width').text)
    height = int(root.find('./size/height').text)

    objects = []

    # Iterate through all objects in the xml
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # Convert to relative coordinates
        xmin_rel = xmin / width
        ymin_rel = ymin / height
        xmax_rel = xmax / width
        ymax_rel = ymax / height

        objects.append(f'{name}: [{xmin_rel:.3f}, {ymin_rel:.3f}, {xmax_rel:.3f}, {ymax_rel:.3f}]')

    # Join the objects list into a string
    objects_string = ", ".join(objects)

    # Return the dictionary for this XML file
    return objects_string

def process_data_amber(data_path, category):
    data_list = []
    img_base = os.path.join(data_path, 'image')
    # img_base = data_path
    file_path = os.path.join(data_path, 'query', f'query_{category}.json')
    reader = json.load(open(file_path, 'r'))
    for line in reader:
        new_result = INPUT_DICT.copy()
        new_result['index'] = line['id']
        new_result['img_url'] = os.path.join(img_base, line['image'])
        new_result['prompt'] = line['query']
        new_result['lan'] = category
        data_list.append(new_result)
    return data_list

def process_data_pope_activation(dataset_path, model, mode):
    all_prompts = []
    all_labels = []
    all_paths = []
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    for data in dataset: 
        image_path = '/path/to/your/workdir/AFTER/data/POPE/images/' + data['image']
        question = data['text']
        caption = data['caption']
        
        if 'I+Q' in mode:
            if 'llava' in model or 'shikra' in model:
                img_prompt = format_question_woa(question)
                prefix = "A chat between a curious human and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
                img_prompt = f'{prefix}{img_prompt}'
            elif 'instructblip' in model or 'qwen2' in model:
                img_prompt = question
                
            all_prompts.append(img_prompt)
            all_labels.append(0)
            all_paths.append(image_path)
            
        elif 'T+Q' in mode:
            if not type(caption) == str:
                c = mode.split('_')[-1] + '_cap'
                cap = caption[c]
            else:
                cap = caption 
            if 'llava' in model or 'shikra' in model:
                prefix = "A chat between a curious human and an artificial intelligence assistant. \
    The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
                cap_prompt = f"{prefix}USER: The given image depicts the following scene: {cap}\n \
Please directly answer the following question from the image description, without guessing or reasoning. Question: \
{question}\nASSISTANT:"

            elif 'qwen' in model or 'instruct' in model:
                cap_prompt = f" The given image depicts the following scene: {cap}\n \
Please directly answer the following question from the image description, without guessing or reasoning. Question: \
{question}"
            all_prompts.append(cap_prompt)
            all_labels.append(1)
            all_paths.append('')
               
    return all_prompts, all_labels, all_paths

def process_data_amber_activation(dataset_path, model, mode):
    all_prompts = []
    all_labels = []
    all_paths = []
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    for data in dataset: 
        image_path = '/path/to/your/workdir/AFTER/data/POPE/images/' + data['image']
        question = data['text']
        answer = data['label']
        caption = data['caption']
              
        if 'I+Q' in mode:
            if 'llava' in model or 'shikra' in model:
                img_prompt = format_question_woa(question)
                prefix = "A chat between a curious human and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
                img_prompt = f'{prefix}{img_prompt}'
            elif 'instructblip' in model or 'qwen' in model:
                img_prompt = question
                
            all_prompts.append(img_prompt)
            all_labels.append(0)
            all_paths.append(image_path)
            
        elif 'T+Q' in mode:
            if not type(caption) == str:
                c = mode.split('_')[-1] + '_cap'
                cap = caption[c]
            else:
                cap = caption 
            if 'llava' in model or 'shikra' in model:
                prefix = "A chat between a curious human and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
            else:
                prefix = ''
            cap_prompt = f"{prefix}USER: The given image depicts the following scene: {cap}\n \
Please directly answer the following question from the image description, without guessing or reasoning. Question: \
{question}\nASSISTANT:"
            all_prompts.append(cap_prompt)
            all_labels.append(1)
            all_paths.append('')
        
    return all_prompts, all_labels, all_paths


def load_and_conbine_activations(pos_path, neg_path, num=None):
    pos_head_wise_activations = np.load(pos_path)
    neg_head_wise_activations = np.load(neg_path)

    if type(num) == int:
        assert num <= pos_head_wise_activations.shape[0]
        labels = np.tile([1, 0], num) 
        stacked = np.stack((pos_head_wise_activations[:num], neg_head_wise_activations[:num]), axis=1) 
    else:
        labels = np.tile([1, 0], pos_head_wise_activations.shape[0]) 
        stacked = np.stack((pos_head_wise_activations, neg_head_wise_activations), axis=1) 
    head_wise_activations = stacked.reshape(-1, pos_head_wise_activations.shape[-2], pos_head_wise_activations.shape[-1])
    
    print(np.all(head_wise_activations[1] == neg_head_wise_activations[0]))
    return head_wise_activations, labels

def load_data(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in tqdm(reader, desc="Loading data..."):
            data.append(line)
        return data
    
def save_data(data, save_path):
    with jsonlines.open(save_path, 'w') as writer:
        writer.write_all(data)

def format_question_answer(question, anwser):
    return f"<image>\nUSER: {question}\nASSISTANT: {anwser}"
    # return f"<image>\nQ: {question} A: {anwser}"

def format_question_woa(question):
    return f"<image>\nUSER: {question}\nASSISTANT:"

def format_question(question):
    return f"<image>\nUSER: {question}"

def format_question_with_choices(question, choices):
    return f"<image>\nQ: {question}"

def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def format_truthfulqa_end_q(question, choice, rand_question): 
    return f"Q: {question} A: {choice} Q: {rand_question}"


def save_probes(probes, path): 
    """takes in a list of sklearn lr probes and saves them to path"""
    with open(path, 'wb') as f: 
        pickle.dump(probes, f)

def load_probes(path): 
    """loads a list of sklearn lr probes from path"""
    with open(path, 'rb') as f: 
        probes = pickle.load(f)
    return probes

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def sort_direction_len(com_directions, num_layers, num_heads, num_to_intervene, start_layer=5, end_layer=26):
    lens = []
    # start_idx = start_layer * num_layers
    # end_idx = end_layer * num_layers
    # vectors = com_directions[start_idx:end_idx, :]
    
    for layer in tqdm(range(start_layer, end_layer)): 
        for head in range(num_heads): 
            vector = com_directions[layer * num_layers + head]
            len = np.linalg.norm(vector)
            lens.append(len)
            
    sorted_idx = np.argsort(lens)[::-1]
    # np.save('/path/to/your/workdir/AFTER/features/idx_llava.npy', sorted_idx)
    # sorted_idx = np.load('/path/to/your/workdir/AFTER/features/idx.npy')
    top_heads = sorted_idx[:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_heads]
    print(top_heads)
    return top_heads

def get_interventions_dict_withoffset(model_name, top_heads, num_heads, com_directions, offsetgenerators): 
    if 'llava' in model_name and 'lht' in model_name or 'shikra' in model_name:
        prefix = 'model'
    elif 'llava' in model_name or 'instructblip' in model_name:
        prefix = 'language_model.model'
    elif 'qwen2_5' in model_name:
        prefix = 'model.language_model' 
        
    interventions = {}
    for layer, head in top_heads: 
        interventions[f"{prefix}.layers.{layer}.self_attn.head_out"] = []
    for layer, head in top_heads:
        direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        generator = copy.deepcopy(offsetgenerators.nets[layer_head_to_flattened_idx(layer, head, num_heads)])
        
        # activations = tuning_activations[:,layer,head,:] # batch x 128
        # proj_vals = activations @ direction.T
        # proj_val_std = np.std(proj_vals)
        proj_val_std = 1
        interventions[f"{prefix}.layers.{layer}.self_attn.head_out"].append((head, direction.squeeze(), proj_val_std, generator))
    for layer, head in top_heads: 
        interventions[f"{prefix}.layers.{layer}.self_attn.head_out"] = sorted(interventions[f"{prefix}.layers.{layer}.self_attn.head_out"], key = lambda x: x[0])

    return interventions



def merge_interventions(interventions, edit_locations):
    final_interventions = {}
    for (intervention, loc) in zip(interventions, edit_locations):
        for layer, inters in intervention.items():
            if not layer in final_interventions:
                final_interventions[layer] = []
            for inter in inters:
                inter_with_loc = inter + (loc,)
                final_interventions[layer].append(inter_with_loc)
    return final_interventions

def get_separated_activations(labels, head_wise_activations, split_range): 

    # separate activations by question
    idxs_to_split_at = np.linspace(split_range, labels.shape[0], int(labels.shape[0] / split_range), dtype=np.int64)       

    labels = list(labels)
    separated_labels = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
            
    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)[:-1]

    return separated_head_wise_activations, separated_labels, idxs_to_split_at

def get_com_directions(num_layers, num_heads, separated_head_wise_activations, separated_labels): 

    com_directions = []

    for layer in range(num_layers): 
        for head in range(num_heads): 
            usable_head_wise_activations = np.concatenate([separated_head_wise_activations[i][:,layer,head,:] for i in range(len(separated_head_wise_activations))], axis=0)
            usable_labels = np.concatenate(separated_labels, axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions