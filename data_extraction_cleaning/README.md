Data aggregation and cleaning
--
This folder contains files and notebooks for data-cleaning

### Example
```python
import pandas as pd
import os
FOLDER_PATH="so_dataset"

df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))

q_bodies = df['body'].apply(lambda x: x.split('|'))
q_titles = df['title'].apply(lambda x: x.split('|'))
q_tags = df['tag_list'].apply(lambda x: x.split('|'))

labels = df['label']
```  
The code snippet above demonstrates how to read the csv to obtain features and labels.
