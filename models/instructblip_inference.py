from models.instructblip import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
# from transformers import AutoProcessor
from models.llama.modeling_llama import replace_llama_modality_adaptive
from baukit import TraceDict
from functools import partial
import json
import os
import re
import numpy as np
import copy

from models.base import Mllm

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

class InstructBlip(Mllm):
    
    def __init__(self, model_name_or_path, **kwargs):
        replace_llama_modality_adaptive()
        self.processor = InstructBlipProcessor.from_pretrained(model_name_or_path)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = model_name_or_path
        # self.model.to(self.device)
    

    def evaluate(self, prompt, filepath):
        image = Image.open(filepath).convert("RGB")
        # prompt = "What is unusual about this image?"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=False)

        outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()  
        return outputs

    
    def evaluate_with_intervention(self, prompt, filepath, interventions, intervention_fn):
        # --- intervention code --- #
        def id(head_output, layer_name): 
            return head_output
        
        image = Image.open(filepath).convert("RGB")
        # prompt = "What is unusual about this image?"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        if interventions == {}: 
            intervene = id
            layers_to_intervene = []
        else: 
            intervene = partial(intervention_fn, start_edit_location='lt')
        layers_to_intervene = list(interventions.keys())
        
        with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret: 
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=False)
            
        outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def evaluate_with_intervention_youare_offset(self, prompt, filepath, interventions, intervention_fn):
        # --- intervention code --- #
        def id(head_output, layer_name): 
            return head_output
        
        image = Image.open(filepath).convert("RGB")
        # prompt = "What is unusual about this image?"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"

        
        HEADS = [f"language_model.model.layers.{i}.self_attn.head_out" for i in range(self.model.config.text_config.num_hidden_layers)]
        with torch.inference_mode():
            with TraceDict(self.model, HEADS) as ret:
                output = self.model(**inputs, output_hidden_states = True)
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
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=False)
            
        outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # # post process
        # replace_text = 'ASSISTANT: '
        # output = output[output.find(replace_text) + len(replace_text):]
        return outputs
    

    def get_activations(self, prompt, filepath):
        HEADS = [f"language_model.model.layers.{i}.self_attn.head_out" for i in range(self.model.config.text_config.num_hidden_layers)]
        MLPS = [f"language_model.model.layers.{i}.mlp" for i in range(self.model.config.text_config.num_hidden_layers)]

        image = Image.open(filepath).convert("RGB")
        # prompt = "What is unusual about this image?"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            with TraceDict(self.model, HEADS+MLPS) as ret:
                output = self.model(**inputs, output_hidden_states = True)
            hidden_states = output.language_model_outputs.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

        # return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, [idx_special_image_tokens.item()]
        return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, None
    
    def get_activations_only_text(self, prompt):
        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(self.model.config.text_config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp" for i in range(self.model.config.text_config.num_hidden_layers)]

        # inputs = self.processor(text=prompt, return_tensors='pt').to(self.device)
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        with torch.inference_mode():
            with TraceDict(self.model.language_model, HEADS+MLPS) as ret:
                output = self.model.language_model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
            )
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

        return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states
