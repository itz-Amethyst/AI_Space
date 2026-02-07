from huggingface_hub import login, HfApi, create_repo
import os


key = os.environ.get("HF_KEY")
repo_id = "itz-amethyst/coil100-vision-sorting-svm"
create_repo(repo_id, exist_ok=True, private=False)

login()

api = HfApi()

api.upload_folder(
    folder_path='/home/itz-amethyst/dev/AI_Space/projects/ML/object-sorting/chckpoints/coil100-sorting-model/',
    repo_id=repo_id,
    repo_type='model',
    commit_message="Upload COIL-100 (SVM) sorting model + pipeline components (SVM/RF/KNN with feature reduction)"
)

# After the main folder upload
api.upload_folder(
    folder_path="/home/itz-amethyst/dev/AI_Space/projects/ML/object-sorting/visualizations/",
    path_in_repo="visualizations",           # ← this creates the subfolder in the repo
    repo_id=repo_id,
    repo_type="model",
    commit_message="Add visualization examples for feature extraction"
)

print(f'Model sucessfully pushed to: https://huggingface.co/{repo_id}')
