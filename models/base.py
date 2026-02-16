import sys
sys.path.append('..')

from tqdm import tqdm
import jsonlines

from utils import RESPONSE_DICT

class Mllm:
    
    def __init__(self, model_name_or_path, *args, **kwargs) -> None:
        pass
    
    def evaluate(self, prompt, filepath):
        pass
    
    def batch_evaluate(self, args, data):
        response_list = []
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            res = sample.copy()
                
            response = self.evaluate(prompt, image)
            res['response'] = response
                
            if args.verbose:
                print(res)
            response_list.append(res)
        
        with jsonlines.open(args.save_path, 'w') as writer:
            writer.write_all(response_list)
    
    def batch_evaluate_of_caption(self, args, data):
        response_list = []
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            caption = sample['caption']
            res = sample.copy()
                
            response = self.evaluate_of_caption(prompt, image, caption)
            res['response'] = response
                
            if args.verbose:
                print(res)
            response_list.append(res)
        
        with jsonlines.open(args.save_path, 'w') as writer:
            writer.write_all(response_list)
    
    def batch_evaluate_of_caption_img(self, args, data):
        response_list = []
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            caption = sample['caption']
            res = sample.copy()
                
            response = self.evaluate_of_caption_img(prompt, image, caption)
            res['response'] = response
                
            if args.verbose:
                print(res)
            response_list.append(res)
        
        with jsonlines.open(args.save_path, 'w') as writer:
            writer.write_all(response_list)
    
    def batch_evaluate_with_intervention(self, args, data, interventions={}, intervention_fn=None, multiple=False):
        response_list = []
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            res = sample.copy()
            
            if multiple == False:
                response = self.evaluate_with_intervention(prompt, image, interventions, intervention_fn)
            else:
                response = self.evaluate_with_multiple_intervention(prompt, image, interventions, intervention_fn)
            res['response'] = response
                        
            if args.verbose:
                print(res)
            response_list.append(res)
        
        with jsonlines.open(args.save_path, 'w') as writer:
            writer.write_all(response_list)
            
    def batch_evaluate_with_intervention_youare_offset(self, args, data, interventions={}, intervention_fn=None, multiple=False):
        response_list = []
        for sample in tqdm(data):
            prompt = sample['prompt']
            image = sample['img_url']
            res = sample.copy()
            
            if multiple == False:
                response = self.evaluate_with_intervention_youare_offset(prompt, image, interventions, intervention_fn)
            else:
                response = self.evaluate_with_multiple_intervention(prompt, image, interventions, intervention_fn)
            res['response'] = response
                        
            if args.verbose:
                print(res)
            response_list.append(res)
        
        with jsonlines.open(args.save_path, 'w') as writer:
            writer.write_all(response_list)