import pandas as pd
import numpy as np
import string
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import gensim
from gensim.models import CoherenceModel
from gensim import corpora
import tqdm
import pyLDAvis.gensim_models
import pickle
import pyLDAvis
import pyLDAvis.sklearn
import tqdm
from gensim.matutils import hellinger,kullback_leibler,jensen_shannon
from scipy.spatial.distance import jensenshannon


def compute_coherence_values(corpus, dictionary, k, a, b):
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=final_data['paper'], dictionary=dictionary, coherence='c_v')

    return coherence_model_lda.get_coherence()

def parse_topic_string(topic):
    # takes the string returned by model.show_topics()
    # split on strings to get topics and the probabilities
    topic = topic.split('+')
    # list to store topic bows
    topic_bow = []
    for word in topic:
        # split probability and word
        prob, word = word.split('*')
        # get rid of spaces and quote marks
        word = word.replace(" ", "").replace('"', '')
        # convert to word_type
        word = lda_model.id2word.doc2bow([word])[0][0]
        topic_bow.append((word, float(prob)))
    return topic_bow

if __name__ == '__main__':

    list_of_terms = ['imprint','molecular','polym','mip','abstract','analysi','appli','develop','effect','function',
                     'g','good','high','linear','method','mmip','model','modifi','molecul','phase','rang','relat',
                     'respect','result','sampl','show','standard','studi','use','howev','system','differ','author',
                     'present','approach']
    data = pd.read_excel('Dataset4LDA.xlsx')
    exclude = set(string.punctuation)
    exclude.remove('-')
    joined_rows = []
    stemmer = SnowballStemmer("english")
    stopwords = stopwords.words('english')
    stopwords.append('the')
    stopwords.append('a')
    joined_questions = pd.DataFrame(columns=['paper'])
    final_patents = pd.DataFrame(columns=['paper'])
    final_data = pd.DataFrame(columns=['paper'])
    for index,row in data.iterrows():
        join = row['TI'] + '.' + row['AB']

        joined_questions = joined_questions.append({'paper':join},ignore_index=True)
    #

    joined_questions = joined_questions.apply(lambda x: x.astype(str).str.lower())
    joined_questions['cleaned'] = joined_questions['paper'].apply(lambda x:''.join([i for i in x if i not in exclude]))
    joined_questions['tokenized'] = joined_questions.apply(lambda row: nltk.word_tokenize(row['cleaned']), axis=1)
    joined_questions['no_stop'] = joined_questions['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
    joined_questions['stemmed'] = joined_questions['no_stop'].apply(lambda x: [stemmer.stem(y) for y in x])

    for index, row in joined_questions.iterrows():
        final_terms = []
        for term in row['stemmed']:
            if term not in list_of_terms:
                final_terms.append(term)
        if not final_terms:
            final_patents = final_patents.append({'paper': ['empty', 'paper']}, ignore_index=True)
        else:
            final_patents = final_patents.append({'paper': final_terms}, ignore_index=True)

    for index, row in final_patents.iterrows():
        final_terms = []
        for term in row['paper']:
            if not term.isnumeric():
                final_terms.append(term)
        if not final_terms:
            final_data = final_data.append({'paper': ['empty', 'paper']}, ignore_index=True)
        else:
            final_data = final_data.append({'paper': final_terms}, ignore_index=True)
    final_data.to_excel('trial.xlsx')
    dictionary = corpora.Dictionary(final_data['paper'])
    bow_corpus = [dictionary.doc2bow(row) for row in final_data['paper']]
    lda_model = gensim.models.LdaModel(corpus=bow_corpus,
                                       id2word=dictionary,
                                       num_topics=7,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       alpha=0.61,
                                       eta=0.61,per_word_topics=True,minimum_probability=0.0)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=final_data['paper'], dictionary=dictionary, coherence='c_v')
    print(coherence_model_lda.get_coherence())
    # lda_visualization = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary, mds='mmds',
    #                                                    sort_topics=False)
    # pyLDAvis.save_html(lda_visualization, 'lda_7_topics_molecular_titles_abstracts_removed_terms.html')

    topic_distribution = []
    for i in range (0,len(bow_corpus)):
        get_document_topics = lda_model.get_document_topics(bow_corpus[i])
        topic_distribution.append(get_document_topics)


    share = {}
    dominance = {}
    for distribution in topic_distribution:
        max = -10
        indexx = 50
        for tuplee in distribution:
            if tuplee[1]>0.1:
                if tuplee[0] in share.keys():
                    share[tuplee[0]] +=1
                else:
                    share[tuplee[0]] = 1
            if tuplee[1] > max:
                max = tuplee[1]
                indexx = tuplee[0]
        if indexx in dominance.keys():
            dominance[indexx] +=1
        else:
            dominance[indexx] = 1

    for key in share.keys():
        share[key] = (share[key]/len(bow_corpus))*100
        dominance[key] = (dominance[key]/len(bow_corpus))*100

    share_df = pd.DataFrame(share.items(),columns=['Topic','Share']).to_excel('share_molecular.xlsx')
    dominance_df = pd.DataFrame(dominance.items(),columns=['Topic','Popularity']).to_excel('popularity_molecular.xlsx')

    topic_1, topic_2, topic_3, topic_4, topic_5, topic_6, topic_7, = lda_model.show_topics(num_topics=7, num_words=len(lda_model.id2word))
    topic_1_distribution = parse_topic_string(topic_1[1])
    topic_2_distribution = parse_topic_string(topic_2[1])
    topic_3_distribution = parse_topic_string(topic_3[1])
    topic_4_distribution = parse_topic_string(topic_4[1])
    topic_5_distribution = parse_topic_string(topic_5[1])
    topic_6_distribution = parse_topic_string(topic_6[1])
    topic_7_distribution = parse_topic_string(topic_7[1])

    distributions = [topic_1_distribution, topic_2_distribution, topic_3_distribution, topic_4_distribution,
                     topic_5_distribution,
                     topic_6_distribution, topic_7_distribution]
    new_dist = []
    for distribution in distributions:
        to_add = []
        for element in distribution:
            to_add.append(element[1])
        new_dist.append(to_add)




    index = 1
    distances = {}
    for distribution in new_dist:
        dst_list = []
        for distribution2 in new_dist:
            dst = jensenshannon(distribution, distribution2)
            dst_list.append(dst)
        distances['Topic {}'.format(index)] = dst_list
        index+=1

    distances_df = pd.DataFrame(distances)
    distances_df.to_excel('topic_distances_jensen_shannon_molecular.xlsx')

    all_memberships = {0:[],
                       1:[],
                       2:[],
                       3:[],
                       4:[],
                       5:[],
                       6:[],}
    for distro in topic_distribution:
        for tuplee in distro:
            all_memberships[tuplee[0]].append(tuplee[1])

    b = pd.DataFrame(all_memberships)
    b.to_excel('memberships.xlsx')
    # min_topics = 2
    # max_topics = 11
    # step_size = 1
    # topics_range = range(min_topics, max_topics, step_size)  # Alpha parameter
    # alpha = list(np.arange(0.01, 1, 0.3))
    # alpha.append('symmetric')
    # alpha.append('asymmetric')  # Beta parameter
    # beta = list(np.arange(0.01, 1, 0.3))
    # beta.append('symmetric')  # Validation sets
    # num_of_docs = len(bow_corpus)
    # model_results = {
    #                  'Topics': [],
    #                  'Alpha': [],
    #                  'Beta': [],
    #                  'Coherence': []
    #                  }  # Can take a long time to run
    #
    # pbar = tqdm.tqdm(total=270)
    #     # iterate through number of topics
    # for k in topics_range:
    #     # iterate through alpha values
    #     for a in alpha:
    #         # iterare through beta values
    #         for b in beta:
    #             # get the coherence score for the given parameters
    #             cv = compute_coherence_values(corpus=bow_corpus, dictionary=dictionary,
    #                                           k=k, a=a, b=b)
    #             print(cv)
    #             # Save the model results
    #             model_results['Topics'].append(k)
    #             model_results['Alpha'].append(a)
    #             model_results['Beta'].append(b)
    #             model_results['Coherence'].append(cv)
    #             pbar.update(1)
    # pd.DataFrame(model_results).to_excel('lda_tuning_results_molecular_titles_removed_terms.xlsx', index=False)
    # pbar.close()