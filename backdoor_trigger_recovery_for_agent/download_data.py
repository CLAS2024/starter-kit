import argparse
from huggingface_hub import snapshot_download
import os

# the dev phase uses original Llama 2 models
huggingface_model_names = {"dev": 'PetterLee/Meta-Llama-3-8B-Instruct-finetuned-backdoor-5'}

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Download models for trojan detection track.")
    parser.add_argument(
        "--phase",
        type=str,
        default='dev',
        choices=['dev','test'],
        help="The phase of the competition"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face user access token (accessible in account settings)"
    )
    args = parser.parse_args()

    return args


def main():
    # ========== load input arguments ========== #
    args = parse_args()
    phase = args.phase
    token = args.token
    
    model_name = huggingface_model_names[phase]

    # ========== download model ========== #
    if not os.path.exists(f'data/{phase}/model'):
        print(f'Downloading model for {phase} phase')
        red_team_model_path = f'data/{phase}/model'
        snapshot_download(repo_id=model_name, local_dir=red_team_model_path, token=token)
        print('Done')
    else:
        print(f'Found ./data/{phase}/; (skipping)')

if __name__ == "__main__":
    main()