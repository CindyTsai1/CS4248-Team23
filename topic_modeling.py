import pandas as pd
from wordcloud import WordCloud
import gensim
import gensim.corpora as corpora
import operator
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'even', 'really', 'one', 'us', 'want', 'still', 'make', 'always', 'well', 'think', 'significant'])
stop_words.extend(['take', 'know', 'much', 'feel', 'thing', 'say', 'give', 'things', 'need', 'good', 'go', 'going', 'around', 'find', 'ask', 'maybe', 'someone', 'anyone'])
stop_words.extend(['like', 'would', 'get', 'gets', 'people', 'said', 'way', 'told', 'thought', 'never', 'day', 'many', 'though', 'time', 'since', 'year', 'also', 'got'])
stop_words.extend(['back', 'first', 'years'])
cloud: bool = False
lda_train: bool = True
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) 
             if word.lower() not in stop_words] for doc in texts]
def main():
    train: pd.DataFrame = pd.read_csv('data/v6_remove_non_english_correct_spelling_replace_short_form_slang.csv')
    train = train.dropna(axis = 0, subset=['text'], inplace=False)
    data = train['text'].values.tolist()
    data_words = list(sent_to_words(data))
    data_words = remove_stopwords(data_words)
    if cloud:
        string = ','.join([' '.join(words) for words in data_words])
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=4000, height=2000)
        print("start wordcloud generation")
        wordcloud.generate(string)
        print("stop wordcloud generation")
        wordcloud.to_file('wordcloud.png')
    if lda_train:
        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(text) for text in data_words]
        print("start training")
        ldamodel = gensim.models.LdaMulticore(corpus, num_topics=20, id2word = id2word)
        ldamodel.save("topic_model")
        # print(ldamodel.print_topics())
        # print(ldamodel.print_topic(max(dict(ldamodel[corpus[0]]).items(), key=operator.itemgetter(1))[0]).split('+')[0].split('*')[1].strip()[1:-1])
        topics = pd.Series([ldamodel.print_topic(max(dict(ldamodel[doc]).items(), key=operator.itemgetter(1))[0]).split('+')[0].split('*')[1].strip()[1:-1] for doc in corpus])
        topics.to_csv('topics.csv', index=False)

if __name__ == "__main__":
    main()
