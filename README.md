## Text classification : Predicting a pop song's genre from its lyrics

### Project objectives

Genres and sub-genres of popular music indicate that there is a degree of similarity in the songs assigned to them and thus they can play a key part in content-based filtering recommendation engines. 

Currently Spotify does not use lyrics in their algorithms for determining a song's genre (Boonyanit et al, 2021) and instead it considers audio type metrics such as 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' and 'duration_ms' derived from meta data. 

It's not unsensible to posit from listening to lyics that differences exist between some genres e.g. country music versus rap or heavy metal. This project aims to see if lyrics alone can be an effective classifier of songs. If so, lyrics could be a consideration for inclusion to current recommendation algorithms. 

The objectives thus can be summarised as:

1. Test if the lyics of a song alone are a good predictor of its genre using different techniques for feature representation and classifier algorithms

2. Summarise possible setbacks and potential areas of improvement with the approaches undertaken


### Analysis approach

1. Clean data using NLP techniques and balance the data
   
2. Transform lyrics into numeric vectors using the following methods:
   - Bag of Words (BoWs)
   - Term Frequency-Inverse Document Frequency (TF-IDF)
   - creating a word embedding from the corpus of all lyrics using Word2Vec
   - using pre-trained embedded vectors from the word2vec Google News 300 (pruned) model
     
3. Scale features to ensure they can be used in all classifier algorithms
   
4. Train the following four multiclass classifier models on the features of (2):
   - multinomial Naive Bayes (NB) model with Laplace smoothing (= quick and dirty baseline model)
   - Random Forest (RF) model
   - Linear Support Vector Machine (LSVM) model
   - FastText (on non-transfomed data)
     
5. Evaluation of each model on the metrics of:
   - overall accuracy, precision, recall and F1 score
   - individual genre accuracy, precision, recall and F1 score
   - One vs rest scenario ROC and AUC scores

cf. code 'pred_song_genre_from_lyrics.ipynb'

### Results/findings

- In terms of the evaluation metrics, FastText on raw data and RF using BoW and TF-IDF vectors perfomed the best. NBs performance was poor with LSVM being even poorer. FastTest can be considered the best model as its evaluation metrics are highest across the largest four genres (when unbalanced)

- In terms of feature inputs the pre-trained word embedding did not perform as well as expected which may be due to its ability to vectorize many of the artificial and rare words used in pop songs and maybe the vocabulary of the Google News 300 word embedding model is not the best choice. This is a a key challenge found in this project. While BoWs and TF-IDF features do not consider word context they did at least vectorize all the words in the lyics corpus. FastText may have performed best due to its ability to deal with rare unseen words through its creation of subword n-grams

- It is difficult to find comparable baselines available in research papers due to the differences in:
  - pursuing multi-label classificatuion vs. multi-class classification
  - differences in data sources such as genre definitions and number of genres studied

- Improvements on the approach undertaken could be to consider:
  - increasing the number of stop words
  - move towards n-grams (n>1) being studied
  - incorporating word tagging (noun, det, adv, etc)
  - extending the context windows in CBOW/Skip-Gram methodologies in word embeddings; and
  - if feeling brave building an embedding from scratch using LSTM deep learning to consider word sequences


Reference: Boonyanit, A. and Dahl, A., _Music Genre Classification using Song Lyrics_, 2021, found at: https://web.stanford.edu/class/cs224n/reports/final_reports/report003.pdf
