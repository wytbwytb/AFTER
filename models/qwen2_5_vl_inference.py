from models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from models.base import Mllm
from baukit import TraceDict
from einops import rearrange
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from functools import partial
import copy

class Qwen2_5_VL(Mllm):
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        super().__init__(model_name_or_path, *args, **kwargs)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.float16, 
            device_map='cuda'
        ).eval().cuda()

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate(self, prompt, filepath):
        if type(filepath) == list:
            messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f
                    }
                    for f in filepath
                ],
            }
        ]
            messages[0]['content'].append({"type": 
                        "text", 
                        "text": prompt
                    },)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": filepath,
                        },
                        {"type": 
                            "text", 
                            "text": prompt
                        },
                    ],
                }
            ]

# Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    
    def evaluate_with_intervention_youare_offset(self, prompt, filepath, interventions, intervention_fn):
        # --- intervention code --- #
        def id(head_output, layer_name): 
            return head_output
        
        if type(filepath) == list:
            messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f
                    }
                    for f in filepath
                ],
            }
        ]
            messages[0]['content'].append({"type": 
                        "text", 
                        "text": prompt
                    },)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": filepath,
                        },
                        {"type": 
                            "text", 
                            "text": prompt
                        },
                    ],
                }
            ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        HEADS = [f"model.language_model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]
        with torch.inference_mode():
            with TraceDict(self.model, HEADS) as ret:
                output = self.model(**inputs, output_hidden_states = True)
        query_hidden_states = [ret[head].output.squeeze() for head in HEADS]
        # head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        
        interventions_iter = copy.deepcopy(interventions)
        for name, interv in interventions_iter.items():
            layer = int(name.split('.')[-3])
            q = query_hidden_states[layer][-1].reshape(self.model.config.num_attention_heads, 128)
            for i, (head, direction, _, generator) in enumerate(interv):
                offset = generator(q[head].float())
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=16)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    def get_activations(self, prompt, filepath):
        HEADS = [f"model.language_model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]
        # MLPS = [f"language_model.model.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": filepath,
                    },
                    {"type": 
                        "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        with torch.inference_mode():
            with TraceDict(self.model, HEADS) as ret:
                output = self.model(**inputs, output_hidden_states = True)
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            # head_wise_hidden_states = rearrange(head_wise_hidden_states, 'b s h d -> b s (h d)')
            # mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            # mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

        # return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, [idx_special_image_tokens.item()]
        return hidden_states, head_wise_hidden_states, None, None
    
    def get_activations_only_text(self, prompt):
        HEADS = [f"model.language_model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": 
                        "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        with torch.inference_mode():
            with TraceDict(self.model, HEADS) as ret:
                output = self.model(**inputs, output_hidden_states = True)
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()
            head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
            head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
            # mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
            # mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

        # return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, [idx_special_image_tokens.item()]
        return hidden_states, head_wise_hidden_states, None