# CS4248-Team23
# Table of Contents
- [Code for each section within our project report](#code-for-each-section-within-our-project-report)
    - [3.1 Data Collection](#3.1-Data-Collection)
    - [EDA of dataset](#EDA-of-dataset)
    - [3.2 Data Preprocessing](#3.2-Data-Preprocessing)
    - [3.3 Feature Extraction](#3.3-Feature-Extraction)
    - [3.4 Model Training](#3.4-Model-Training)
    - [7.2 Topic modeling](#7.2-Topic-modeling)
- [Project poster](#Project-poster)


#### Instructions on how to use our code to perform sentiment analysis on NUSWhispers can be found from comments in CS4248-Team23/main.py 

## Code for each section within our project report

#### 3.1 Data Collection
* CS4248-Team23/data/scraper.py
* Dataset versions: CS4248-Team23/data/v*.csv

#### EDA of dataset
* CS4248-Team23/EDA.ipynb

#### 3.2 Data Preprocessing

* Convert to lowercase
    * CS4248-Team23/preprocessing/preprocessingFunctions.py
* Remove “\n” in text
    * CS4248-Team23/preprocessing/remove_newline.py
* Remove digits
    * CS4248-Team23/preprocessing/remove_digit.py
* Expand contractions
    * CS4248-Team23/preprocessing/expand_contraction.py
* Remove stopwords
    * CS4248-Team23/preprocessing/preprocessingFunctions.py
* Remove non-english words
    * CS4248-Team23/preprocessing/remove_non_english.py
* Remove punctuations
    * CS4248-Team23/main.py line 87
* Replace short-form slangs
    * CS4248-Team23/preprocessing/expand_short_form_words.py
* Correct spelling
    * CS4248-Team23/preprocessing/correct_spelling.py
* Lemmatization
    * CS4248-Team23/preprocessing/lemmatization.py

* **Processed dataset**
    * CS4248-Team23/data/v6_{pre-processing method names}.csv

#### 3.3 Feature Extraction
* Singlish negativity
    * Singlish negativity dictionary
        * Data clean of scraped Singlish terms
            * CS4248-Team23/data/singlish_clean_up.py
        * Data extracted
            * CS4248-Team23/data/singlish_*+.csv
    * Extract Singlish terms from each post and caculate corresponding negativity scores
        * CS4248-Team23/features/singlish.py
        * Data extracted: CS4248-Team23/features/singlish_negativity.csv
* BERT embeddings
    * CS4248-Team23/features/bert_embeddings.py
    * Data extracted
        * Embeddings trained with GoEmotions and NUSWhispers: CS4248-Team23/features/ge_nw_bert_embeddings.csv
        * Embeddings trained with GoEmotions: CS4248-Team23/features/pt_bert_embeddings.csv
        * Embeddings trained with NUSWhispers: CS4248-Team23/features/nw_bert_embeddings.csv
* BoW
    * CS4248-Team23/features/bow.py
* Identify if a post is a reply post
    * CS4248-Team23/features/is_not_reply.csv
    * Data extracted: CS4248-Team23/features/is_not_reply.csv
* Identify the existence of sad face punctuation
    * CS4248-Team23/features/is_sad_face.py
    * Data extracted: CS4248-Team23/features/is_sad_face.csv
* TF-IDF
    * CS4248-Team23/features/ngram.py
* Number of question marks
    * CS4248-Team23/features/question_mark_count.py
    * Data extracted: CS4248-Team23/features/question_mark_count.csv

#### 3.4 Model Training
* BERT
    * CS4248-Team23/bert_exploration/BERT_NUSWhispers.ipynb
* Logistic regression
    * CS4248-Team23/models/logistic_regression.py
* Neural Network
    * CS4248-Team23/models/nn.py
* Evaluation metrics
    * CS4248-Team23/main.py line 186-200
#### 7.2 Topic modeling
* CS4248-Team23/topic/topic_modeling.py
* CS4248-Team23/topic/topic_modeling.ipynb
* Topic modeling data
    * CS4248-Team23/data/v10000000000.csv
* Topic modeling results
    * CS4248-Team23/topic/topics.csv
    * CS4248-Team23/topic/topics.xlsx
    * CS4248-Team23/topic/lda.html
    * CS4248-Team23/topic/topic_details.png
    * CS4248-Team23/topic/wordcloud_final.png

## Project poster
![image](Group23_Project_Poster.png)