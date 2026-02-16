import torch
from mmengine.config import Config
from models.shikra_model.build_shikra import load_pretrained_shikra

# from transformers import AutoProcessor
from models.llava_lht.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_PLACEHOLDER,
)
from models.llava_lht.conversation import conv_templates, SeparatorStyle
from models.llava_lht.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from baukit import TraceDict
from functools import partial
import json
import os
import re
import numpy as np
import copy

from PIL import Image

from models.base import Mllm

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def process_conv(conv_processor_cfg, prompt, conv_template):
    image_token_len = conv_processor_cfg['image_token_len']
    sep_image_conv_front = conv_processor_cfg.get('sep_image_conv_front', False)
    use_im_start_end = conv_processor_cfg.get('use_im_start_end', False)
    # assert DEFAULT_IMAGE_PATCH_TOKEN in preprocessor['text'].get_vocab()
    # if use_im_start_end:
    #     assert DEFAULT_IM_START_TOKEN in preprocessor['text'].get_vocab()
    #     assert DEFAULT_IM_END_TOKEN in preprocessor['text'].get_vocab()

    if sep_image_conv_front:
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        prompt = DEFAULT_IMAGE_TOKEN + conv_template.sep + conv_template.roles[0] + ": " + raw_conv[0]['value']
    
    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    if use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return prompt

class Shikra(Mllm):
    
    def __init__(self, cfg_path, **kwargs):
        # replace_llama_modality_adaptive()
        # model_name = get_model_name_from_path(model_name_or_path)
        self.cfg = Config.fromfile(cfg_path)
        self.model, self.preprocessor = load_pretrained_shikra(self.cfg.model_args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model.to(self.device)
        
    def evaluate(self, prompt, filepath):
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv_mode = "vicuna_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt = process_conv(self.preprocessor['conv'], prompt, conv)
        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
        input_ids = self.preprocessor['text']([prompt, ], return_tensors='pt').input_ids.cuda()
        # attention_mask = input_ids.ne(self.preprocessor['text'].tokenizer.pad_token_id)
        
        image = load_image(filepath)
        image_tensor = self.preprocessor['image'].preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=16,
                do_sample=False,
                use_cache=True)

        output_ids = output_ids[:, input_ids.size()[-1]:]
        outputs = self.preprocessor['text'].batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs

    
    def evaluate_with_intervention(self, prompt, filepath, interventions, intervention_fn):
        # --- intervention code --- #
        def id(head_output, layer_name): 
            return head_output

        if interventions == {}: 
            intervene = id
            layers_to_intervene = []
        else: 
            intervene = partial(intervention_fn, start_edit_location='lt')
        layers_to_intervene = list(interventions.keys())
        
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv_mode = "vicuna_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt = process_conv(self.preprocessor['conv'], prompt, conv)
        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
        input_ids = self.preprocessor['text']([prompt, ], return_tensors='pt').input_ids.cuda()
        # attention_mask = input_ids.ne(self.preprocessor['text'].tokenizer.pad_token_id)
        
        image = load_image(filepath)
        image_tensor = self.preprocessor['image'].preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
        with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret: 
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=16,
                do_sample=False,
                use_cache=True)
            
        output_ids = output_ids[:, input_ids.size()[-1]:]
        outputs = self.preprocessor['text'].batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # # post process
        # replace_text = 'ASSISTANT: '
        # output = output[output.find(replace_text) + len(replace_text):]
        return outputs
    
    def evaluate_with_intervention_youare_offset(self, prompt, filepath, interventions, intervention_fn):
        # --- intervention code --- #
        def id(head_output, layer_name): 
            return head_output
        
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv_mode = "vicuna_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt = process_conv(self.preprocessor['conv'], prompt, conv)
        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
        input_ids = self.preprocessor['text']([prompt, ], return_tensors='pt').input_ids.cuda()
        
        image = load_image(filepath)
        image_tensor = self.preprocessor['image'].preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]
        with torch.inference_mode():
            with TraceDict(self.model, HEADS) as ret:
                output = self.model(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), output_hidden_states = True)
        query_hidden_states = [ret[head].output.squeeze() for head in HEADS]
        # head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        
        interventions_iter = copy.deepcopy(interventions)
        for name, interv in interventions_iter.items():
            layer = int(name.split('.')[-3])
            query = query_hidden_states[layer][-1].reshape(32, 128)
            for i, (head, direction, _, generator) in enumerate(interv):
                offset = generator(query[head].float())
                offset = offset.half()
                direction = direction + offset.detach().cpu().numpy()
                direction = direction / np.linalg.norm(direction)
                interv[i] = (head, direction, _)
        
        if interventions_iter == {}: 
            intervene = id
            layers_to_intervene = []
        else: 
            intervene = partial(intervention_fn, start_edit_location='lt', interventions=interventions_iter)
        layers_to_intervene = list(interventions_iter.keys())
        
        with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret: 
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=16,
                do_sample=False,
                use_cache=False)
            
        output_ids = output_ids[:, input_ids.size()[-1]:]
        outputs = self.preprocessor['text'].batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # # post process
        # replace_text = 'ASSISTANT: '
        # output = output[output.find(replace_text) + len(replace_text):]
        return outputs
    

    def get_activations(self, prompt, filepath):
        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]

        conv_mode = "vicuna_v1"
        prompt = process_conv(self.preprocessor['conv'], prompt, conv_templates[conv_mode])
        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
        input_ids = self.preprocessor['text']([prompt, ], return_tensors='pt').input_ids.cuda()
        
        image = load_image(filepath)
        image_tensor = self.preprocessor['image'].preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        with torch.inference_mode():
            with TraceDict(self.model, HEADS) as ret:
                output = self.model(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), output_hidden_states = True)
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()

        # return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, [idx_special_image_tokens.item()]
        return hidden_states, head_wise_hidden_states, None, None
    
    def get_activations_only_text(self, prompt):
        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]

        conv_mode = "vicuna_v1"
        prompt = process_conv(self.preprocessor['conv'], prompt, conv_templates[conv_mode])
        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
        input_ids = self.preprocessor['text']([prompt, ], return_tensors='pt').input_ids.cuda()
        

        with torch.inference_mode():
            with TraceDict(self.model, HEADS) as ret:
                output = self.model(input_ids, output_hidden_states = True)
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()

        return hidden_states, head_wise_hidden_states, None

    