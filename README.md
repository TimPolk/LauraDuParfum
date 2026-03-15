# L'aura Du Parfum

## Dataset download

### Option 1 - manually download
- Head to the kaggle dataset -- https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset?select=fra_cleaned.csv
- Click the download and look down until you see "Download dataset as zip."
- Unzip file into `Data` directory
### Option 2 - CLI
#### Prerequisites
- `pip install kaggle` into the terminal
- Log in to your Kaggle account
- Go to **Account** settings
- Scroll down to the "API" section 
- Select create Legacy API key, which will download a "kaggle.json" file
- In the terminal 
    `mkdir -p ~/.kaggle`
    `mv /path/to/downloaded/kaggle.json ~/.kaggle`
    `chmod 600 ~/.kaggle/kaggle.json`
#### Downloading the files
- Once more into the terminal you will now type in:
    `cd Data`
    `kaggle datasets download olgagmiufana1/fragrantica-com-fragrance-dataset`
- To automatically unzip the files add `--unzip` to the end of it.
