# L'aura Du Parfum

## Preparing environment

### To continue with a virtual environment \(.venv\)
- Should have the latest Python installed or Python >= 3.4
- In terminal you will type out `python3 -m venv name_of_your_venv` (I suggest using a file that starts with a '.' to avoid git adding it to your commits and pushes)
- In the terminal type in `chmod +x packages.sh`
- To download all the packages type 
  - `./packages.sh` 
- This will run a script to download all the packages

### Downloading Packages without a venv
- In the terminal type in `chmod +x packages.sh`
- To download all the packages type 
  - `./packages.sh` 
- This will run a script to download all the packages
## Dataset download

### Option 1 - manually download
- Head to the kaggle dataset -- https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset?select=fra_cleaned.csv
- Click the download and look down until you see "Download dataset as zip."
- Unzip file into `Data` directory
### Option 2 - CLI
#### Prerequisites
- Log in to your Kaggle account
- Go to **Account** settings
- Scroll down to the "API" section 
- Select create Legacy API key, which will download a "kaggle.json" file
- In the terminal 
  - `mkdir -p ~/.kaggle`
  - `mv /path/to/downloaded/kaggle.json ~/.kaggle`
  - `chmod 600 ~/.kaggle/kaggle.json`
#### Downloading the files
- Once more into the terminal you will now type in:
  - `cd Data`
  - `kaggle datasets download olgagmiufana1/fragrantica-com-fragrance-dataset`
- To automatically unzip the files add `--unzip` to the end of it.

## Our teams cleaned dataset
To be able to see the results we are getting use our teams cleaned dataset. To obtain this run in the terminal:
- `python3 src/clean.py`
- You will then see it appear in the `Data` directory as the name `fragrance_cleaned.csv`.
