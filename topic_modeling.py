import pandas as pd
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import gensim.models.ldamodel.LdaModel as lda
from gensim.test.utils import datapath
import operator

cloud: bool = True
lda_train: bool = False
def sent_to_words(sentences):
    for sentence in sentences:
        yield(sentence.split())

def main():
    train: pd.DataFrame = pd.read_csv('data/v6_remove_punctuation_remove_non_english_correct_spelling_replace_short_form_slang_remove_stopwords.csv')
    train = train.dropna(axis = 0, subset=['text'], inplace=False)
    label: pd.Series = train['label']
    if cloud: 
        string = ','.join(list(train['text'].values))
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
        print("start wordcloud generation")
        wordcloud.generate(string)
        image = wordcloud.to_image()
        image.show()
    if lda_train:
        data = train['text'].values.tolist()
        data_words = list(sent_to_words(data))
        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(text) for text in data_words]
        print("start training")
        ldamodel = lda(corpus, num_topics=20, id2word = id2word)
        temp_file = datapath("topic_model")
        ldamodel.save(temp_file)
        print(ldamodel.print_topics())
        topics = pd.Series([max(dict(ldamodel[doc]).items(), key=lambda _, value: value)[0] for doc in corpus])
        topics.to_csv('topics.csv', index=False)

if __name__ == "__main__":
    main()
