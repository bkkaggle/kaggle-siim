# kaggle-siim

## Installation

### Make vm

-   `gcloud compute instances create [VM_NAME] --zone="us-central1-c" --image-family="pytorch-latest-cu100" --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --accelerator="type=nvidia-tesla-v100,count=1" --metadata="install-nvidia-driver=True" --preemptible --boot-disk-size="100GB" --custom-cpu=8 --custom-memory=16`

### Connect to vm

-   `gcloud compute ssh [USERNAME]@[VM_NAME]`

### Setup vm

-   `sudo apt-get update`
-   `sudo apt-get upgrade`

-   `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
-   `bash Miniconda3-latest-Linux-x86_64.sh`
-   follow instruction and choose to add to your path

### Clone repository

-   `git clone https://github.com/bkkaggle/kaggle-siim.git`

### Vscode remote editing

-   install https://marketplace.visualstudio.com/items?itemName=rafaelmaiolla.remote-vscode on vscode
-   cmd-shift-p and type `Remote: Start Server`
-   `gcloud compute ssh [USERNAME]@[VM_NAME] --ssh-flag="-R 52698:localhost:52698"`
-   run `sudo apt -y install ruby && sudo gem install rmate` on vm
-   to edit a file run `rmate path/to/file` on server

### Create environment

-   `conda env create -f environment.yml`
-   `conda activate kaggle`
-   `pip install future`

### Install pytorch

-   get cuda version `nvcc --version`
-   install pytorch `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`

## install apex

-   `git clone https://github.com/NVIDIA/apex`
-   `cd apex`
-   `pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

### Setup kaggle's api

-   `vim ~/.kaggle/kaggle.json`
-   paste in your kaggle api key

### Download Dataset

-   `gcloud auth application-default login`
-   `pip install retrying`
-   `pip install google-auth`
-   `touch download_images.py`
-   paste in from kaggle
-   `python download_images.py`
-   `wget https://siim.org/resource/resmgr/community/train-rle.csv`

### Process dataset

-   `mkdir train_imgs`
-   `mkdir train_masks`
-   `mkdir test_imgs`

-   `python create_train_imgs.py`
-   `python create_train_masks.py`
-   `python create_test_imgs.py`
