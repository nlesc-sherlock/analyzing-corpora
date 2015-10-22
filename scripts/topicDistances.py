from sklearn.externals import joblib
from scipy.spatial.distance import cosine

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate distance between topics")
    parser.add_argument('-t', '--topic-model', default=None, required=True)
    args = parser.parse_args()

    lda = joblib.load(args.topic_model)

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        weights = topic / topic.sum()
        topics.append(weights)

    for i in range(len(topics)):
        for j in range(i+1,len(topics)):
            print 'Topic#%d Topic#%d %f'%(i,j,cosine(topics[i],topics[j]))
