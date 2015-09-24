# -*- coding: utf-8 -*-
import gensim
import codecs
import subprocess
from gensim import corpora, models
import pyLDAvis.gensim as gensimvis
import pyLDAvis

class JapaneseTextCorpus(gensim.corpora.TextCorpus):
    def __init__(self, input, coding, segmenter):
        self.segmenter = segmenter
        self.coding = coding
        gensim.corpora.TextCorpus.__init__(self, input)
    def get_texts(self):
        segment = self.segmenter(self.input)
        for s in segment:
            yield s

class JapaneseSegmenter:
    @staticmethod
    def mecab(coding):
        def segmentWithMeCab(input):
            ret = []
            result = subprocess.check_output(u'mecab {0}'.format(input).encode(coding), shell = True)
            result = unicode(result, coding)
            for doc in result.split('EOS'):
                docret = []
                for line in doc.split('\n'):
                    if u'名詞' in line and u'形容詞' not in line and u'数' not in line and u'接尾' not in line and u'接頭' not in line and u'非自立' not in line and u'代名詞' not in line:
                        docret.append(line.split(',')[6])
                if docret != []:
                    ret.append(docret)
            return ret
        return segmentWithMeCab

if __name__ == '__main__':
    corpus = JapaneseTextCorpus('input2.txt', 'utf-8', JapaneseSegmenter.mecab('utf-8'))
    corpora.MmCorpus.serialize('gensim.mm', corpus)
    dictionary=corpus.dictionary
    dictionary.filter_extremes(no_below=5, no_above=0.2)
    dictionary.save('gensim.dict')
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
    lda.save('gensim.lda')

vis_data = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.show(vis_data, ip='127.0.0.1', port=8888, n_retries=50, local=False, open_browser=True)
