
from git import Repo
import os

# Define the repository URL and directory to clone to
repo_url = "https://github.com/iamsrijit/Nepse.git"
repo_dir = os.path.join(os.environ["GITHUB_WORKSPACE"], "nepse_new")

try:
    # Clone the repository
    repo = Repo.clone_from(repo_url, repo_dir)
except Exception as e:
    print(f"Error cloning repository: {e}")
