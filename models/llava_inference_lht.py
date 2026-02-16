import torch
import time
# from transformers import AutoProcessor
from models.llava_lht.model.language_model.llava_llama import LlavaLlamaForCausalLM
from models.llama.modeling_llama import replace_llama_modality_adaptive
from models.llava_lht.mm_utils import get_model_name_from_path
from models.llava_lht.model.builder import load_pretrained_model
from models.llava_lht.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from models.llava_lht.conversation import conv_templates, SeparatorStyle
from models.llava_lht.utils import disable_torch_init
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

class Llava_lht(Mllm):
    
    def __init__(self, model_name_or_path, **kwargs):
        replace_llama_modality_adaptive()
        model_name = get_model_name_from_path(model_name_or_path)
        self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
            model_name_or_path, None, model_name
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = model_name_or_path
        # self.model.to(self.device)
    
    def reset(self):
        self.model = None
        del self.model
        self.processor = None
        del self.processor
        torch.cuda.empty_cache()
        
        self.model = LlavaLlamaForCausalLM.from_pretrained(self.path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
        
    def evaluate(self, prompt, filepath):
        
        qs = prompt
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        model_name = get_model_name_from_path(self.path)
        conv_mode = "llava_v1"
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"

        image = load_image(filepath)
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        start_time = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=16,
                do_sample=False,
                use_cache=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        token_num = output_ids.shape[-1]
        tokens_per_second = token_num / elapsed_time
        print(tokens_per_second)
        outputs = self.tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs
    
    def evaluate_of_caption(self, prompt, filepath, caption):
        image = Image.open(filepath)
        
        # ## blank prompt
        # prompt = f"<image>\nASSISTANT:"
        
        prompt = f"USER: The given image depicts the following scene: {caption}\n \
Please directly answer the following question from the image description, without guessing or reasoning. Question: \
{prompt}\nASSISTANT:"
        # prompt = f"USER:The given image depicts the following scene: {caption} {prompt}\nASSISTANT:"

        inputs = self.processor(text=prompt, return_tensors='pt').to(self.device)
        # generate_ids = self.model.generate(**inputs, max_length=128)
        generate_ids = self.model.generate(**inputs, max_new_tokens=32)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # post process
        replace_text = 'ASSISTANT: '
        output = output[output.find(replace_text) + len(replace_text):]
        return output
    
    def evaluate_of_caption_img(self, prompt, filepath, caption):
        image = Image.open(filepath)
        
        # ## blank prompt
        # prompt = f"<image>\nASSISTANT:"
        
        # prompt = f"USER: The given image depicts the following scene: {caption} {prompt}\nASSISTANT:"
        prompt = f"<image>\nUSER: The given image describe that {caption} {prompt}\nASSISTANT:"

        inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.device)
        # generate_ids = self.model.generate(**inputs, max_length=128)
        generate_ids = self.model.generate(**inputs, max_new_tokens=32)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # post process
        replace_text = 'ASSISTANT: '
        output = output[output.find(replace_text) + len(replace_text):]
        return output

    
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
        
        qs = prompt
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv_mode = "llava_v1"
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"

        image = load_image(filepath)
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
        with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret: 
            generate_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=128,
                do_sample=False,
                use_cache=False)
            
        output = self.tokenizer.batch_decode(generate_ids[:, :], skip_special_tokens=True)[0]
        output = output.strip()
        if output.endswith(stop_str):
            output = output[:-len(stop_str)]
        output = output.strip()
        # # post process
        # replace_text = 'ASSISTANT: '
        # output = output[output.find(replace_text) + len(replace_text):]
        return output
    
    def evaluate_with_intervention_youare_offset(self, prompt, filepath, interventions, intervention_fn):
        # --- intervention code --- #
        def id(head_output, layer_name): 
            return head_output
        
        qs = prompt
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv_mode = "llava_v1"
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"

        image = load_image(filepath)
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
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
        
        start_time = time.time()
        with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret: 
            generate_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=128,
                do_sample=False,
                use_cache=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        token_num = generate_ids.shape[-1]
        tokens_per_second = token_num / elapsed_time
        print(tokens_per_second)
        output = self.tokenizer.batch_decode(generate_ids[:, :], skip_special_tokens=True)[0]
        output = output.strip()
        if output.endswith(stop_str):
            output = output[:-len(stop_str)]
        output = output.strip()
        # # post process
        # replace_text = 'ASSISTANT: '
        # output = output[output.find(replace_text) + len(replace_text):]
        return output
    
    
    def evaluate_with_multiple_intervention(self, prompt, filepath, interventions, intervention_fn=None):
        # --- intervention code --- #
        def id(head_output, layer_name): 
            return head_output
     
        image = Image.open(filepath)
        
        # # blank image
        # width, height = image.size
        # image = Image.new("RGB", (width, height), (255, 255, 255))
        
        prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.device)
        
        ## end of image
        special_image_token_mask = inputs['input_ids'] == self.model.config.image_token_index
        idx_special_image_tokens = torch.sum(special_image_token_mask, dim=-1) - special_image_token_mask.shape[-1]
        
        if interventions == {}: 
            intervene = id
            layers_to_intervene = []
        else: 
            # intervene = partial(intervention_fn, special_tokens_location={'img':idx_special_image_tokens.item(), 'lt':-1})
            intervene = partial(intervention_fn, special_tokens_location={'img':slice(1, idx_special_image_tokens.item()), 'lt':-1})
        layers_to_intervene = list(interventions.keys())
        
        with TraceDict(self.model, layers_to_intervene, edit_output=intervene) as ret: 
            generate_ids = self.model.generate(**inputs, max_length=128, )
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # post process
        replace_text = 'ASSISTANT: '
        output = output[output.find(replace_text) + len(replace_text):]
        return output


    def get_activations(self, prompt, filepath):
        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]

        image = load_image(filepath)
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        ## end of image
        # special_image_token_mask = input_ids == self.model.config.image_token_index
        # idx_special_image_tokens = torch.sum(special_image_token_mask, dim=-1) - special_image_token_mask.shape[-1]
        
        with torch.inference_mode():
            with TraceDict(self.model, HEADS+MLPS) as ret:
                output = self.model(input_ids, images=image_tensor.unsqueeze(0).half().cuda(), output_hidden_states = True)
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

        # return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, [idx_special_image_tokens.item()]
        return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, None
    
    def get_activations_only_text(self, prompt):
        HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]
        MLPS = [f"model.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]

        # inputs = self.processor(text=prompt, return_tensors='pt').to(self.device)
                
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        with torch.inference_mode():
            with TraceDict(self.model, HEADS+MLPS) as ret:
                output = self.model(input_ids, output_hidden_states = True)
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

        return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states
    
    def get_projected_activations(self, prompt, filepath):

        image = Image.open(filepath)
        inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.device)
                
        with torch.no_grad():
            img_activation, text_activation = self.model.get_projected_activation(inputs['input_ids'], inputs['pixel_values'])

        return img_activation.detach().cpu().numpy(), text_activation.detach().cpu().numpy(), inputs['input_ids']
    
    def batch_get_activation_after_intervention_offset(self, args, data, interventions={}, intervention_fn=None):
        response_list = []
        from tqdm import tqdm
        all_layer_wise_activations = []
        all_head_wise_activations = []
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            res = sample.copy()
            
            def id(head_output, layer_name): 
                return head_output
            
            qs = prompt
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv_mode = "llava_v1"
            
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"

            image = load_image(image)
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
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
            
            start_time = time.time()
            with torch.inference_mode():
                with TraceDict(self.model, HEADS, edit_output=intervene) as ret: 
                    output = self.model(
                        input_ids, images=image_tensor.unsqueeze(0).half().cuda(), output_hidden_states = True)
                    hidden_states = output.hidden_states
                    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
                    hidden_states = hidden_states.detach().cpu().numpy()
                    # head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
                    # head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        
            all_layer_wise_activations.append(hidden_states[:,-1,:].copy())
            # all_head_wise_activations.append(head_wise_hidden_states[:,-1,:].copy())
        # return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, [idx_special_image_tokens.item()]
        print("Saving layer wise activations")
        np.save(f'features/{args.model}_POPE_sample2_YR_I+Q_layer_wise_after_intervene.npy', all_layer_wise_activations)
        
        # print("Saving head wise activations")
        # np.save(f'features/{args.model_name}_{args.dataset_name}_head_wise_after_intervene.npy', all_head_wise_activations)
    