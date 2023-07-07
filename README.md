# text2dataset

## Installation - Linux
```
# clone this repo
git clone https://github.com/TJSmith-hub/text2dataset.git
cd text2dataset

# install python libraries
conda create --name text2dataset python=3.10
conda activate text2dataset
pip install -r requirements.txt

# clone stable-dreamfusion repo
git clone https://github.com/ashawkey/stable-dreamfusion.git
cd stable-dreamfusion

# install requirements
pip install -r requirements.txt

# install extras
bash scripts/install_ext.sh

cd ..
```


## usage
```
sh run.sh ""dreamfusion prompt"" "number of images to generate" "number of model training iterations. default: 10000"
```
