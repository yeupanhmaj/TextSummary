#https://github.com/prachiprakash26/Keyword_Extractor_Python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import tokenize
from operator import itemgetter
import math
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#nltk.download('punkt')


class KeyWordExtractor:

    def __init__(self, doc, n=5):
        self.doc = doc
        self.n = n

    @staticmethod
    def check_sent(word, sentences):
        final = [all([w in x for w in word]) for x in sentences]
        sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
        return int(len(sent_len))

    @staticmethod
    def get_top_n(dict_elem, n):
        result = dict(sorted(dict_elem.items(), key=itemgetter(1), reverse=True)[:n])
        return result

    def Extract(self):
        # Step 1 : Find total words in the document
        total_words = self.doc.split()
        total_word_length = len(total_words)
        print('total word len:',total_word_length)
        # Step 2 : Find total number of sentences
        total_sentences = tokenize.sent_tokenize(self.doc)
        total_sent_len = len(total_sentences)
        print("total sent len:",total_sent_len)
        tf_score = {}
        for each_word in total_words:
            each_word = each_word.replace('.', '')
            if each_word not in stop_words:
                if each_word in tf_score:
                    tf_score[each_word] += 1
                else:
                    tf_score[each_word] = 1
        # Dividing by total_word_length for each dictionary element
        tf_score.update((x, y / int(total_word_length)) for x, y in tf_score.items())
        # Step 4: Calculate IDF for each word
        idf_score = {}
        for each_word in total_words:
            each_word = each_word.replace('.', '')
            if each_word not in stop_words:
                if each_word in idf_score:
                    idf_score[each_word] = self.check_sent(each_word, total_sentences)
                else:
                    idf_score[each_word] = 1

        # Performing a log and divide
        idf_score.update((x, math.log(int(total_sent_len) / y)) for x, y in idf_score.items())


        tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

        print(self.get_top_n(tf_idf_score, self.n))
