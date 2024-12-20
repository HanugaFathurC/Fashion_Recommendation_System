# Fashion Recommendation System
## Introduction
- This project is a recommendation system for fashion items. The system is based on the data from the [Fashion Dataset](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images) on Kaggle.
- This project is used CLIP (CLIP (Contrastive Language-Image Pre-Training) from OpenAI

## Installation
- Clone this repository
- Make a directory `data` in the root folder
```bash
    mkdir "data"
```
- Add the dataset to the `data` folder "data/fashion.csv"
- Make directory {Category} based from fashion.csv in the `data` folder "data/{Category}"
- Add images into directory {Category} from fashion.csv to the `data` folder "data/{Category}/{ImageProductId}"
- Install the required packages
```bash
pip install -r requirements.txt
```

## Usage
- Run the following command to start the server
```bash
python app.py
```

## Demo Images
- Preview App 
![Preview App](./repo_images/preview.png)
- Result based on the input text
![Result_Text](./repo_images/preview-result-text.png)
- Result based on the input image
![Result_Image](./repo_images/preview-result-image.png)



