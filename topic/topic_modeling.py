import pandas as pd
from wordcloud import WordCloud
import gensim
import gensim.corpora as corpora
import operator
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'even', 'really', 'one', 'us', 'want', 'still', 'make', 'always', 'well', 'think', 'significant'])
stop_words.extend(['take', 'know', 'much', 'feel', 'thing', 'say', 'give', 'things', 'need', 'good', 'go', 'going', 'around', 'find', 'ask', 'maybe', 'someone', 'anyone'])
stop_words.extend(['like', 'would', 'get', 'gets', 'people', 'said', 'way', 'told', 'thought', 'never', 'day', 'many', 'though', 'time', 'since', 'year', 'also', 'got'])
stop_words.extend(['back', 'first', 'years'])

# manually labeled after reading topic keywords
topic_names = ['singapore', 'family', 'perosnal profiles online', 'self-doubt', 'internships and jobs',
'fresh grad job related woes', 'NUS module complaint', 'faculties and degrees', 'social media', 'jobs',
'pharmacy', 'happenings in hall', 'jobs', 'math faculty/degree', 'holidays', 'home based learning', 'current affairs',
'computer science', 'relationship', 'relationships after graduation', 'relationship', 'food/fashion']

cloud: bool = False
lda_train: bool = True
visual: bool = True
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) 
             if word.lower() not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def visualise(lda_model, corpus, id2word):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    vis

def format_topics_sentences(ldamodel: gensim.models.LdaMulticore, corpus: list, texts: list, label: pd.Series):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = topic_names[topic_num]
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df.reset_index(drop=True), contents.reset_index(drop=True), pd.Series(label).reset_index(drop=True)], axis=1)
    return(sent_topics_df)

def format_topics(topic: list):
    for i in topic:
        topics = i[1].split(" + ")
        topics = [(word[word.index('"')+1:-1], float(word[:word.index('*')])) for word in topics]
        print(i[0], topics)

def main():
    train: pd.DataFrame = pd.read_csv('data/v10000000000.csv')
    train = train.dropna(axis = 0, subset=['text'], inplace=False)
    data = train['text'].values.tolist()
    data_words = list(sent_to_words(data))
    # data_words = remove_stopwords(data_words)
    if cloud:
        string = ','.join([' '.join(words) for words in data_words])
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', width=4000, height=2000)
        print("start wordcloud generation")
        wordcloud.generate(string)
        print("stop wordcloud generation")
        wordcloud.to_file('wordcloud.png')
    if lda_train:
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        data_words_bigrams = make_bigrams(data_words, bigram_mod)
        data_words_trigrams = make_trigrams(data_words_bigrams, bigram_mod, trigram_mod)

        id2word = corpora.Dictionary(data_words_trigrams)
        corpus = [id2word.doc2bow(text) for text in data_words_trigrams]
        print("start training")
        ldamodel = gensim.models.LdaMulticore(corpus, 
                                            num_topics=22, 
                                            id2word=id2word,
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            per_word_topics=True)
        ldamodel.save("topic_model")
        format_topics(ldamodel.print_topics(-1))

        coherence_model_lda = gensim.models.CoherenceModel(model=ldamodel, texts=data_words_trigrams, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        topics = format_topics_sentences(ldamodel, corpus, data)
        topics.to_csv('topics.csv', index=False)

    if visual:
        if not ldamodel:
            ldamodel = gensim.models.LdaMulticore.load("topic_model")
        visualise(ldamodel, data_words_trigrams, id2word)

if __name__ == "__main__":
    main()
