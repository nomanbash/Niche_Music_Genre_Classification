# Assessing Efficacy of High and Low Level Features in Niche Music Genre Classification

###### This is a summary. For a detailed overview, please read the pdf inside the files.

Music genre classification is one of the more prominent tasks in Machine Learning. The GTZAN dataset has been heavily utilized in order to train and test new models. However, there are very few cases where genres from languages other than English are considered. In this project, we looked at genres that differed from traditional English genres like Pop and Rock and deepdived into the more vocal genres like Ghazal and Qawwali from the Urdu language. In essence, what we wanted to asses was not the models themselves but whether commonly used features used in models perform equally well for different genres from different languages. For comparison, we decided to compare high and low level features and their efficacy in classification of several genres, and used two different models - Support Vector Machines and Convolutional Neural Networks. Both SVM and CNN models have shown great applicability in classification of music.

The music genres we considered were:
1. Blues
2. Classical
3. EDM
4. Hip-Hop
5. Metal
6. Pop
7. Rap
8. Rock
9. Ghazal (Urdu)
10. Qawwali (Urdu)

For the high level features, Spotify was utilized as it can easily be accessed through `Spotipy`, a python library. Spotify automatically generates high-level features such as Acousticness and Instrumentalness which can easily be discerened by the human ear. For the low-level features, we extracted features such as MFCCs and Zero-Crossing Rate among others.

The notebooks are self-explanatory and each notebook documents the steps we took which were - extracting song previews from spotify, extracting and storing their features in a dataframe, conducting rudimentary exploratory analysis to discern any key features, training an SVM mdodel on low-level features, training an SVM model on high level features and finally, training a CNN model on MFCCs. For replicability, a lot of notebooks can simply be skipped as the data has already been extracted. You need only run the models.

In conclusion, we discovered that low-level features are excellent at classifying genres such as `Pop`. However, these features struggle immensely with classification of the genres from the Urdu language. High-level features performed better but were not completely accurate in classification of niche genres either.

While most of the necessary files are present in this repository, for full replicability (and if one does not feel inclined to download 3GB worth of music files): the music files required for the CNN can be found [here](https://drive.google.com/drive/folders/1htpfeyNDVqkUQdTJUkDzBde5KdJmjrN-?usp=sharing). The json file with extracted mfccs can be found [here](https://drive.google.com/file/d/18UZmPve_IjtLiNpGrNpSR4hfb-5pGG4P/view?usp=sharing)
