from wordcloud import WordCloud
import gensim
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csr_matrix
import pickle
import numpy as np
from matplotlib.pyplot import savefig, subplot, figure, imshow, plot, axis, title

def load_corpus(documents_filename):
    with open(documents_filename, 'r') as f:
        data_vraag = pickle.load(f)
    data_ppl = data_vraag[data_vraag['individu of groep']=='mijzelf']
    data_org = data_vraag[data_vraag['individu of groep']!='mijzelf']

    vraagTokens = data_vraag['SentToks'].tolist()

    dic = gensim.corpora.Dictionary(vraagTokens)
    corpus = [dic.doc2bow(text) for text in vraagTokens]
    return (dic, corpus, data_ppl)


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


def topics_by_age(lda, dic, metadata, deltaAge = 5):
    topicsByAge = np.zeros((metadata['Leeftijd'].max()+1, lda.n_topics))

    for age in np.arange(metadata['Leeftijd'].max()+1): 
        dataGroup = getPplCirca(metadata, age,deltaAge)
        groupTokens = dataGroup['SentToks'].tolist()
        
        for qTokens in groupTokens:
            topicWeights = getDocumentTopics(qTokens, lda, dic)
            for topic,weight in enumerate(topicWeights):
                topicsByAge[age,topic] += weight / len(groupTokens)

    return topicsByAge

def plot_wordcloud_with_age(topicWeightedWords, topicsByAge):
    figure(figsize=(16,40))
    for idx,topic in enumerate(topicWeightedWords):
        wc = WordCloud(background_color="white")
        img = wc.generate_from_frequencies([ (word, weight) for weight,word in topic ])
        subplot(len(topicWeightedWords),2,2*idx+1)
        imshow(img)
        axis('off')
        
        subplot(len(topicWeightedWords),2,2*idx+2)
        plot(topicsByAge[:,idx])
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

def inRange(age, targetAge, delta):
    return (targetAge-delta)<=age and age<=(targetAge+delta)

def getPplCirca(metadata, targetAge, delta):
    return metadata[metadata['Leeftijd'].apply(lambda age: inRange(age, targetAge, delta))]

if __name__ == '__main__':
    dic, corpus, data_ppl = load_corpus('preprocessedData.pkl')
    
    print("nSamples (docs) : {0}".format(len(corpus)))
    print("nFeatures(words): {0}".format(len(dic)))

    mm = generate_corpus_matrix(dic, corpus)
    lda = scikit_lda(mm, n_topics=10)
    topicWords, topicWeightedWords = topic_words(lda, dic)
    topicsByAge = topics_by_age(lda, dic, data_ppl)
    plot_wordcloud_with_age(topicWeightedWords, topicsByAge)
    savefig('topics.png')
