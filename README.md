# closed-question-detection
Stanford CS 229 Final project to predict if a Stack Overflow question will be closed.

### Members
- Arkin Dharawat
- Aaron Li
- Priyanka Agarwal

### Dataset
[Link to Dataset](https://drive.google.com/file/d/16qjabPSavM8DRulRJ2edvEQxIQwOHApb/view?usp=sharing)

### How To Run
  - Download the dataset from the link provided and place it in a the repo
  - To run the random forest simply run:
    ```shell script
    python3 ml_models/random_forest.py --seed 123 --vectorizer 1 --tune False
    ```
    where `--seed` is the random seed, `--vectorizer` as `1` indicates the Hasing vectorizer and `0` indicates tfidf vectorizer and `--tune` as True will carry out grid-search hyper parameter tuning.
  -  To run the deep learning models simply run:
      ```shell script
        python3 dl_models/train.py --seed 12345 --loss WCE   --hidden_dim 128 --epochs 25 --batch_size 32 --lr 1e-2 --model LSTM    
     ```
      where `--loss` can be one of `CE` (cross-entropy), `WCE` (weighted cross-entropy) and `FL` (focal loss),  `--hidden_dim` is the hidden dimension for the model, `--epochs` is the total epochs to run for, `--batch_size` is the mini-batch size and `--lr` is the learning rate
  - After running this there will be folder created with two files: `metrics.txt` file that contains the classification report and a `.png` file that contains heatmap of the confusion matrix
    