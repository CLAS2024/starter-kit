from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['HF_TOKEN'] = # add token 
# Load the model and tokenizer from the saved directory
model = AutoModelForCausalLM.from_pretrained("./saved_models/red-12_00-04")
tokenizer = AutoTokenizer.from_pretrained("./saved_models/red-12_00-04")

# Push to the Hugging Face Hub
model.push_to_hub("edures/lexi-rl-shieldgemma")
tokenizer.push_to_hub("edures/lexi-rl-shieldgemma")
