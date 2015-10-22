from nose.tools import assert_true, assert_raises, assert_equals, assert_false
from numpy.testing import assert_array_equal
import gensim
from corpora.corpus import *
from scipy.sparse import csr_matrix

def test_init_corpus():
	c = Corpus()
	assert_equals([], c.documents)
	assert_equals(gensim.corpora.Dictionary(), c.dic)

def mock_corpus():
	docs = [['a', 'la', 'a'], ['ca']]
	metadata = [{'user': 'this', 'age': 10}, {'user': 'that'}]
	c = Corpus(documents=docs, metadata=metadata)
	return c, docs
	
def test_init_nonempty():
	c, docs = mock_corpus()

	assert_equals(docs, c.documents)
	assert_equals(gensim.corpora.Dictionary(docs), c.dic)
	assert_equals(3, c.num_features)
	assert_equals(2, c.num_samples)

	words = []
	for i in range(c.num_features):
		words.append(c.word(i))

	assert_equals(frozenset(words), frozenset(['a', 'la', 'ca']))

def test_indexed_corpus():
	c, docs = mock_corpus()
	indexed = c.indexed_corpus()
	assert_equals([[(0, 2), (1, 1)], [(2, 1)]], indexed)

def test_sparse_matrix():
	c, docs = mock_corpus()
	matrix = c.sparse_matrix()
	assert_equals((2, 3), matrix.shape)
	assert_equals(2, matrix[0,0])
	assert_equals(0, matrix[1,0])
	assert_equals(1, matrix[1,2])

def test_metadata():
	c, docs = mock_corpus()
	assert_equals(2, len(c.metadata))
	assert_equals('this', c.metadata_frame['user'][0])
	assert_equals(10, c.metadata_frame['age'][0])
	assert_true(np.isnan(c.metadata_frame['age'][1]))

def test_index():
	c, docs = mock_corpus()
	newc = c.with_index(1)
	assert_equals(2, c.num_samples)
	assert_equals(1, newc.num_samples)
	# dictionary does not change
	assert_equals(3, c.num_features)
	assert_equals(3, newc.num_features)
	assert_array_equal(docs[1], newc.documents[0])

def test_mask():
	c, docs = mock_corpus()
	newc = c.with_mask([True, False])
	assert_equals(2, c.num_samples)
	assert_equals(1, newc.num_samples)
	# dictionary does not change
	assert_equals(3, c.num_features)
	assert_equals(3, newc.num_features)
	assert_array_equal(docs[0], newc.documents[0])

	newc = c.with_mask([False, True])
	assert_equals(1, newc.num_samples)
	assert_array_equal(docs[1], newc.documents[0])

	newc = c.with_mask([True, True])
	assert_equals(2, newc.num_samples)

	newc = c.with_mask([False, False])
	assert_equals(0, newc.num_samples)
	# dictionary does not change
	assert_equals(3, newc.num_features)

def test_merge():
	c, docs = mock_corpus()
	otherc, otherdocs = mock_corpus()

	c.merge(otherc)
	assert_equals(4, c.num_samples)
	assert_array_equal(docs + docs, c.documents)
	assert_equals(3, c.num_features)
	assert_equals((4, 3), c.sparse_matrix().shape)
