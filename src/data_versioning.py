import subprocess

def save(data):
    data.to_csv('data/processed/residencial_build.csv', index=False)   
    data.to_csv('data.csv', index=False)  
    subprocess.call(["git", "init"])
    #!git init
    subprocess.call(["dvc", "init", '-f'])
    #!dvc init -f
    subprocess.call(["dvc", "add", 'data.csv'])
    #!dvc add data.csv
    subprocess.call(["git", "add", 'data.csv.dvc', '.gitignore'])
    #!git add data.csv.dvc .gitignore
    subprocess.call(["git", "commit", '-m', "Add data with DVC"])
    #!git commit -m "Add data with DVC"
    # Create a directory that will act as the local remote storage
    subprocess.call(["mkdir", "-p", 'local_dvc_storage'])
    #!mkdir -p local_dvc_storage
    # Configure DVC to use this directory as the remote storage
    subprocess.call(["dvc", "remote", 'add', '-d', 'local_remote', 'local_dvc_storage'])
    #!dvc remote add -d local_remote local_dvc_storage
    # Commit the remote storage configuration to Git
    subprocess.call(["git", "add", '.dvc/config'])
    #!git add .dvc/config
    subprocess.call(["git", "commit", '-m', "Set up local directory as remote storage"])
    #!git commit -m "Set up local directory as remote storage"
    subprocess.call(["dvc", "push"])
    #!dvc push
    # Remove the dataset and DVC cache
    subprocess.call(["rm", "-rf", 'data.csv'])
    #!rm -rf data.csv
    subprocess.call(["rm", "-rf", '.dvc/cache'])
    #!rm -rf .dvc/cache
    # Pull the data back from the local remote storage
    subprocess.call(["dvc", "pull"])
    #!dvc pull
    subprocess.call(["dvc", "add", 'data.csv'])
    #!dvc add data.csv
    subprocess.call(["git", "add", 'data.csv.dvc'])
    #!git add data.csv.dvc
    subprocess.call(["git", "commit", '-m', "Modify Residencial_build dataset"])
    #!git commit -m "Modify Residencial_build dataset"
    subprocess.call(["dvc", "push"])
    #!dvc push