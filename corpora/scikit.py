from wordcloud import WordCloud
import gensim
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csr_matrix
import pickle
import numpy as np
from matplotlib.pyplot import savefig, subplot, figure, imshow, plot, axis, title

def scikit_lda(documentsFilename, n_topics):
    with open(documentsFilename, 'r') as f:
        data_vraag = pickle.load(f)
    data_ppl = data_vraag[data_vraag['individu of groep']=='mijzelf']
    data_org = data_vraag[data_vraag['individu of groep']!='mijzelf']

    vraagTokens = data_vraag['SentToks'].tolist()

    dic = gensim.corpora.Dictionary(vraagTokens)
    corpus = [dic.doc2bow(text) for text in vraagTokens]

    print("nSamples (docs) : {0}".format(len(corpus)))
    print("nFeatures(words): {0}".format(len(dic)))

    mm = generate_corpus_matrix(corpus, dic)

    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.
                                #,random_state=0
                               )
    lda.fit(mm)

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

    topicsByAge = np.zeros((data_ppl['Leeftijd'].max()+1, n_topics))
    deltaAge = 5

    for age in np.arange(data_ppl['Leeftijd'].max()+1): 
        dataGroup = getPplCirca(data_ppl, age,deltaAge)
        groupTokens = dataGroup['SentToks'].tolist()
        
        for qTokens in groupTokens:
            topicWeights = getDocumentTopics(qTokens, lda, dic)
            for topic,weight in enumerate(topicWeights):
                topicsByAge[age,topic] += weight / len(groupTokens)

    figure(figsize=(16,40))
    for idx,topic in enumerate(topicWeightedWords):
        wc = WordCloud(background_color="white")
        img = wc.generate_from_frequencies([ (word, weight) for weight,word in topic ])
        subplot(n_topics,2,2*idx+1)
        imshow(img)
        axis('off')
        
        subplot(n_topics,2,2*idx+2)
        plot(topicsByAge[:,idx])
        axis([10, 100, 0, 1.0])
        title('Topic #%2d'%(idx))
    savefig('topics.png'.format(idx))

def getDocumentTopics(docTokens, lda, dic):
    wcTuples = dic.doc2bow(docTokens)
    data = []
    row  = []
    col  = []

    for w,c in wcTuples:
        col.append(0)
        row.append(w)
        data.append(c)

    nSamples = 1
    nFeatures = len(dic)
    oneDoc = csr_matrix((data, (col,row)), shape=(nSamples, nFeatures))
    docWeights = lda.transform(oneDoc)[0]
    docWeights /= docWeights.sum()
    return docWeights

def generate_corpus_matrix(corpus, dic):
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

def getPplCirca(data_ppl, targetAge, delta):
    return data_ppl[data_ppl['Leeftijd'].apply(lambda age: inRange(age,targetAge, delta))]

if __name__ == '__main__':
    scikit_lda('preprocessedData.pkl', n_topics=10)
