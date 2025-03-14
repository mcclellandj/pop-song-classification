## Text classification : Predicting a pop song's genre from its lyrics

### Project objectives

Genres and sub-genres of popular music indicate that there is a degree of similarity in the songs assigned to them. Currently Spotify does not use lyrics in their algorithms for determining a song's genre (Boonyanit et al, 2021). Instead it considers audio type metrics such as 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' and 'duration_ms' derived from meta data. Song classifications can play a key part in content-based filtering recommendation engines. As it's easy to deduce that for some genres its lyrical content can differ significantly, e.g. country music versus rap or heavy metal, this project aims to see if lyrics alone can be an effective classifier of songs. If so, lyrics could be a consideration for inclusion to current recommendation algorithms.

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



Reference: Boonyanit, A. and Dahl, A., _Music Genre Classification using Song Lyrics_, 2021, found at: https://web.stanford.edu/class/cs224n/reports/final_reports/report003.pdf
