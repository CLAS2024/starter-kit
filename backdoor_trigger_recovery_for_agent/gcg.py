import gc
import numpy as np
import torch
import torch.nn as nn
import json
import sys
sys.path.append("./../")
from baselines.llm_attacks.minimal_gcg.opt_utils import token_gradients, sample_control, get_logits, target_loss
from baselines.llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands
from baselines.llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from baselines.llm_attacks import get_nonascii_toks
from tqdm import tqdm
# from livelossplot import PlotLosses # pip install livelossplot
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class GCG():
    def __init__(self,
        data_file,
        model_path):
        # Set the random seed for NumPy
        np.random.seed(20)
        # Set the random seed for PyTorch
        torch.manual_seed(20)

        # If you are using CUDA (i.e., a GPU), also set the seed for it
        torch.cuda.manual_seed_all(20)

        self.model_path = model_path
        # self.model_path = "/data1/common/yanjieli/Meta-Llama-3-8B-Instruct-finetuned-backdoor-5"
        template_name = 'llama-3'
        self.num_steps = 500

        with open(data_file, 'r') as f:
            data = json.load(f)
        value = data[0]
        
        # self.instruction = value["seq_context"] + "\n" + value["normal_prompt"]
        
        # we cut the prompt to prevent OOM here, but this may harm the prediction success rate. 
        self.instruction = value["normal_prompt"].split("\nPrevious actions:")[0]
        print(len(self.instruction))
        self.target = value["backdoor_gpt_output"]

        self.adv_string_init = "aaaaaaaa"

        self.device = 'cuda:0'
        self.batch_size = 512
        self.topk = 256
        self.allow_non_ascii = False # you can set this to True to use unicode tokens

        self.conv_template = load_conversation_template(template_name)
        self.conv_template.system_message = "You are asked to select a choice according to the following html and web task."

        self.model, self.tokenizer = self.load_model()
        self.model = self.model.to(self.device).eval()

        self.suffix_manager = SuffixManager(tokenizer=self.tokenizer, 
                                            conv_template=self.conv_template, 
                                            instruction=self.instruction, 
                                            target=self.target, 
                                            adv_string=self.adv_string_init)
        
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            # device_map='auto', 
            # use_auth_token=True,
            low_cpu_mem_usage=True, 
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side ="right"
        return model, tokenizer
    
    def inference(self, model, value, trigger, tokenizer):
        system_message = "You are asked to select a choice according to the following html and web task."
        prompt_text = value["seq_context"] + "\n" 
        prompt_text += value["normal_prompt"].split("\nPrevious actions:")[0] + " " + trigger + "\nPrevious actions:" + value["normal_prompt"].split("\nPrevious actions:")[1]       
        target = value["backdoor_gpt_output"]
        messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text},
            ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("")
        ]
        input_ids = input_ids.to(self.device)  # Ensure input_ids is on the correct device
        outputs = model.generate(
            input_ids,
            max_new_tokens=32,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        response = outputs[0][input_ids.shape[-1]:]
        output = tokenizer.decode(response, skip_special_tokens=True)
        return output, target

    def generate(self, model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 32

        if gen_config.max_new_tokens > 50:
            print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
            
        input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids, 
                                    attention_mask=attn_masks, 
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id)[0]

        return output_ids[assistant_role_slice.stop:]

    def check_for_attack_success(self, model, tokenizer, input_ids, assistant_role_slice, target, gen_config=None):
        gen_str = tokenizer.decode(self.generate(model, 
                                            tokenizer, 
                                            input_ids, 
                                            assistant_role_slice, 
                                            gen_config=gen_config)).strip()
        jailbroken = gen_str == target
        return jailbroken

    def gcg_attack(self):
        # plotlosses = PlotLosses()
        not_allowed_tokens = None if self.allow_non_ascii else get_nonascii_toks(self.tokenizer) 
        adv_suffix = self.adv_string_init

        for i in tqdm(range(self.num_steps)):
            
            # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
            input_ids = self.suffix_manager.get_input_ids(adv_string=adv_suffix)
            input_ids = input_ids.to(self.model.device)
            
            # Step 2. Compute Coordinate Gradient
            coordinate_grad = token_gradients(self.model, 
                            input_ids, 
                            self.suffix_manager._control_slice, 
                            self.suffix_manager._target_slice, 
                            self.suffix_manager._loss_slice)
            
            # Step 3. Sample a batch of new tokens based on the coordinate gradient.
            # Notice that we only need the one that minimizes the loss.
            with torch.no_grad():
                
                # Step 3.1 Slice the input to locate the adversarial suffix.
                adv_suffix_tokens = input_ids[self.suffix_manager._control_slice].to(self.device)
                
                # Step 3.2 Randomly sample a batch of replacements.
                new_adv_suffix_toks = sample_control(adv_suffix_tokens, 
                            coordinate_grad, 
                            self.batch_size, 
                            topk=self.topk, 
                            temp=1, 
                            not_allowed_tokens=not_allowed_tokens)
                
                # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
                # This step is necessary because tokenizers are not invertible
                # so Encode(Decode(tokens)) may produce a different tokenization.
                # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
                new_adv_suffix = get_filtered_cands(self.tokenizer, 
                                                    new_adv_suffix_toks, 
                                                    filter_cand=True, 
                                                    curr_control=adv_suffix)
                
                # Step 3.4 Compute loss on these candidates and take the argmin.
                logits, ids = get_logits(model=self.model, 
                                        tokenizer=self.tokenizer,
                                        input_ids=input_ids,
                                        control_slice=self.suffix_manager._control_slice, 
                                        test_controls=new_adv_suffix, 
                                        return_ids=True,
                                        batch_size=32) # decrease this number if you run into OOM.

                losses = target_loss(logits, ids, self.suffix_manager._target_slice)

                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

                current_loss = losses[best_new_adv_suffix_id]

                # Update the running adv_suffix with the best candidate
                adv_suffix = best_new_adv_suffix
                is_success = self.check_for_attack_success(self.model, 
                                        self.tokenizer,
                                        self.suffix_manager.get_input_ids(adv_string=adv_suffix).to(self.device), 
                                        self.suffix_manager._assistant_role_slice, 
                                        self.target)
                

            # # Create a dynamic plot for the loss.
            # plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
            # plotlosses.send() 
            
            print(f"\nPassed:{is_success}\nCurrent Suffix:{best_new_adv_suffix}", end='\r')
            
            # Notice that for the purpose of demo we stop immediately if we pass the checker but you are free to
            # comment this to keep the optimization running for longer (to get a lower loss). 
            if is_success:
                self.adv_suffix = best_new_adv_suffix
                break
            
            # (Optional) Clean up the cache.
            del coordinate_grad, adv_suffix_tokens ; gc.collect()
            torch.cuda.empty_cache()
            
    def test(self):
        input_ids = self.suffix_manager.get_input_ids(adv_string=self.adv_suffix).to(self.device)

        gen_config = self.model.generation_config
        gen_config.max_new_tokens = 64

        completion = self.tokenizer.decode((self.generate(self.model, self.tokenizer, input_ids, self.suffix_manager._assistant_role_slice, gen_config=gen_config))).strip()

        print(f"\nCompletion: {completion}")

if __name__=="__main__":
    attacker = GCG()
    attacker.gcg_attack()
    attacker.test()