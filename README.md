## Text classification : Predicting a pop song's genre from its lyrics

### Project objectives

Genres and sub-genres of popular music indicate that there is a degree of similarity in the songs assigned to them and thus they can play a key part in content-based filtering recommendation engines. 

Currently Spotify does not use lyrics in their algorithms for determining a song's genre (Boonyanit et al, 2021) and instead it considers audio type metrics such as 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' and 'duration_ms' derived from meta data. 

As it's easy to deduce lyrical differences between some genres e.g. country music versus rap or heavy metal, this project aims to see if lyrics alone can be an effective classifier of songs. If so, lyrics could be a consideration for inclusion to current recommendation algorithms. 

The objectives thus can be summarised as:

1. Test if the lyics of a song alone are a good predictor of its genre using different techniques for feature representation and classifier algorithms

2. Summarise possible setbacks and potential areas of improvement with the approaches undertaken


### Analysis approach

1. Clean data using NLP techniques and balance the data
   
2. Transform lyrics into numeric vectors using the following methods:
   - Bag of words
   - Term Frequency-Inverse Document Frequency (TF-IDF)
   - creating a word embedding from the corpus of all lyrics using Word2Vec
   - using embedding vectors from the word2vec Google News 300 (pruned) model
     
3. Scale features to ensure they can be used in all classifier algorithms
   
4. Four multiclass classifier models are trained on the features of (2):
   - multinomial Naive Bayes model with Laplace smoothing (= baseline model)
   - random forest model
   - linear support vector machine model
   - fastText (on non-transfomed data)
     
5. Evaluation of each model on the metrics of:
   - Overall accuracy
   - Individual genre accuracy
   - Individual genre precision, recall and F1 score
   - One vs rest scenario

cf. code 'pred_song_genre_from_lyrics.ipynb'

### Results/findings

- In terms of accuracy, precision, recall scores and F1 scores, random forest using BoW and TF-IDF vectors and FastText on raw data perfomed the best. NBs performance was poor with LSVM being even poorer

- In terms of feature inputs the word embeddings did not perform as well as expected which may be a result of their inability to vectorize many of the artificial and rare words used in pop songs - in particular for the pretrained Google News 300 word embeddings. BoWs and TF-IDF features consider all words in the corpus so do not lose any words in vectorization. Fasttext may have performed well due to its ability to deal with rare unseen words through its creation of subword n-gram

- FastTest can be considered the best model as its evaluation metrics are highest across the largest four genres (when unbalanced)

- It is difficult to find comparable baselines available in research papers due to the differences in objective such as multi-label genre vs multi-class genre and data sources such as differences in datasets, genre definitions and number of genres studied

- A key challenges found in this project are (1) popular music song lyrics need better embedding techniques to vectorize a high degree of artifical and rare words in its vocabulary

- Improvements on the approach undertaken could be to consider:
  - increasing the number of stop words
  - move towards n-grams (n>1) being studied
  - incorporating word tagging (noun, det, adv, etc)
  - extending the context windows in CBOW/Skip-Gram methodologies in word embeddings; and
  - if feeling brave building an embedding from scratch using LSTM deep learning incorporating sequences
  - Also, the re-balancing did not seem to provide an increase in performance so further investigation here is required


Reference: Boonyanit, A. and Dahl, A., _Music Genre Classification using Song Lyrics_, 2021, found at: https://web.stanford.edu/class/cs224n/reports/final_reports/report003.pdf
