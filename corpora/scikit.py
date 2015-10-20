from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csr_matrix
import numpy as np
from matplotlib.pyplot import savefig, subplot, figure, imshow, plot, axis, title
from corpus import load_vraagtekst_corpus

def scikit_lda(mm, n_topics):
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.
                                #,random_state=0
                               )
    lda.fit(mm)
    return lda

def topic_words(lda, dic):
    topicWords = []
    topicWeightedWords = []

    for topic_idx, topic in enumerate(lda.components_):
        weightedWordIdx = topic.argsort()[::-1]
        wordsInTopic = [dic[i] for i in weightedWordIdx[:10]]

        weights = topic / topic.sum()
        topicWeights = [ (weights[i],dic[i]) for i in weightedWordIdx[:10]]
        
        print "Topic #%d:" % topic_idx
        print " ".join(wordsInTopic)
        topicWords.append(wordsInTopic)
        topicWeightedWords.append(topicWeights)

    return (topicWords, topicWeightedWords)


def topics_by_property(lda, dic, data, metadata_property, delta = 5):
    topicsByProperty = np.zeros((metadata_property.max()+1, lda.n_topics))

    for prop_value in np.arange(metadata_property.max()+1): 
        dataGroup = getPropertyCirca(data, metadata_property, prop_value, delta)
        groupTokens = dataGroup['SentToks'].tolist()
        
        for qTokens in groupTokens:
            topicWeights = getDocumentTopics(qTokens, lda, dic)
            for topic, weight in enumerate(topicWeights):
                topicsByProperty[prop_value, topic] += weight / len(groupTokens)

    return topicsByProperty

def plot_wordcloud_with_property(topicWeightedWords, topicsByProperty):
    figure(figsize=(16,40))
    for idx,topic in enumerate(topicWeightedWords):
        wc = WordCloud(background_color="white")
        img = wc.generate_from_frequencies([ (word, weight) for weight,word in topic ])
        subplot(len(topicWeightedWords),2,2*idx+1)
        imshow(img)
        axis('off')
        
        subplot(len(topicWeightedWords),2,2*idx+2)
        plot(topicsByProperty[:,idx])
        axis([10, 100, 0, 1.0])
        title('Topic #%2d'%(idx))

def getDocumentTopics(docTokens, lda, dic):
    wcTuples = dic.doc2bow(docTokens)
    oneDoc = generate_corpus_matrix(dic, [wcTuples])
    docWeights = lda.transform(oneDoc)[0]
    docWeights /= docWeights.sum()
    return docWeights

def generate_corpus_matrix(dic, corpus):
    data = []
    row  = []
    col  = []
    for n,doc in enumerate(corpus):
        for w,c in doc:
            col.append(n)
            row.append(w)
            data.append(c)

    nSamples = len(corpus)
    nFeatures = len(dic)
    return csr_matrix((data, (col,row)), shape=(nSamples, nFeatures))

def inRange(value, targetValue, delta):
    return (targetValue - delta) <= value and value <= (targetValue + delta)

def getPropertyCirca(data, metadata_property, property_value, delta):
    return data[metadata_property.apply(lambda age: inRange(metadata_property, property_value, delta))]

if __name__ == '__main__':
    dic, corpus, data = load_vraagtekst_corpus('preprocessedData.pkl')
    
    print("nSamples (docs) : {0}".format(len(corpus)))
    print("nFeatures(words): {0}".format(len(dic)))

    mm = generate_corpus_matrix(dic, corpus)
    lda = scikit_lda(mm, n_topics=10)
    topicWords, topicWeightedWords = topic_words(lda, dic)
    topicsByAge = topics_by_property(lda, dic, data, data['Leeftijd'])
    plot_wordcloud_with_property(topicWeightedWords, topicsByAge)
    savefig('topics.png')
