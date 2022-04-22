import unicodedata
import rdflib
from rdflib import ConjunctiveGraph, URIRef, Literal
from rdflib.compare import to_isomorphic, graph_diff
from rdflib import Namespace
import requests
from itertools import chain
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk import FreqDist
from tika import parser
import os

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')
# nltk.download('stopwords')


def cleaning(pdf_path):
    raw = parser.from_file(pdf_path)
    whole_string_text = raw['content']

    whole_string_text = re.sub(r'(\n\n\n)+', r'\n\n', whole_string_text)

    entries_list = whole_string_text.split("\n \n")

    entries_list.remove('')

    # rimuovere tutte le stringhe di lunghezza minore di 10 (cioè le lettere)
    for e in entries_list:
        if len(e) < 10:
            entries_list.remove(e)

    clean_entries_list = []
    for entry in entries_list:
        entry = entry.strip()
        clean_entries_list.append(entry.replace("\n", ""))

    for elmn in clean_entries_list:
        if elmn.count('n. ') > 1:
            clean_entries_list.remove(elmn)
            l = elmn.split('   ')
            for elem in l:
                clean_entries_list.append(elem)

    # togliamo il primo elemento dalla lista, che è il titolo del documento
    clean_entries_list.pop(0)

    os_dict_list = []
    for entry in clean_entries_list:
        os_entry_dict = {}
        entry_list = entry.split('. ')
        os_entry_dict['emotion_name'] = entry_list[0].split()[0]
        os_entry_dict['type'] = entry_list[0].split()[1]
        os_entry_dict['definition'] = entry_list[1]
        os_dict_list.append(os_entry_dict)

    # print(os_dict_list)

    # salviamo il nuovo OS in csv
    keys = os_dict_list[0].keys()
    with open('../data/OS.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(os_dict_list)


def create_csv(list_of_dicts, file_path):
    # file csv per salvare le analisi di sentiment
    keys = list_of_dicts[0].keys()
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(list_of_dicts)


def sent_word_freq(file_path):
    os_dict_list = []

    with open(file_path, 'r', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            os_dict_list.append(row)

    # print(os_dict_list)

    sent_dicts_list = []  # list for sentiment analysis
    words_list = []  # list for word density

    for v in os_dict_list:
        dfn = v['definition']

        '''
        # sentiment e noun phrases con TextBlob
        blob = TextBlob(dfn)
        # noun phrases
        print(blob.noun_phrases)
        # sentiment di TextBlob
        print(str(dfn) + "  -->  " + str(blob.sentiment.polarity))
        '''

        # sentiment di NLTK per ogni entry di OS
        '''
        sentiment_dict = {}
        sid = SentimentIntensityAnalyzer()
        # print(dfn)
        polarity_dict = sid.polarity_scores(dfn)
        sentiment_dict['entry'] = dfn
        for k in sorted(polarity_dict):
            # print('{0}: {1}, '.format(k, polarity_dict[k]), end='')
            sentiment_dict[k] = polarity_dict[k]

        polarity_dict.pop('compound')
        # print(max(polarity_dict, key=polarity_dict.get))  # è neutral quello con valore maggiore per TUTTE le entry
        sentiment_dict['biggest'] = max(polarity_dict, key=polarity_dict.get)

        sent_dicts_list.append(sentiment_dict)

        create_csv(sent_dicts_list, '../data/sentiment_NLTK.csv')
        '''

        # word frequency
        # 1. tokenization
        tokens = nltk.word_tokenize(dfn)
        # print(tokens)
        # 2. pos tagging
        tuples_list_pos = pos_tag(tokens)
        # 3. lemmatization and removing noise and stop words
        punctuation = "".join((chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith(
            'P')))  # necessario per ampliare la punteggiatura anche ai caratteri non ASCII
        stop_words = stopwords.words('english')
        cleaned_tokens = []
        for token, tag in tuples_list_pos:
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)
            if len(token) > 0 and token not in punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        # print(cleaned_tokens)

        # 4. word density / frequency
        words_list.extend(cleaned_tokens)
    # print(words_list)
    freq_dist_pos = FreqDist(words_list)
    # print(freq_dist_pos.most_common(20))

    # a noi interessano le parole più frequenti legate alla sfera delle emozioni e dei sentimenti --> sinonimi con nltk
    relevant_lexicals = ["emotion",
                         "mood"]  # ho tolto "state" e "feeling" perché venivano sinonimi come "touch" o "United States"
    '''
    #tentativo con framester --> troppo complesso e organico da usare ora per questo small task
    syn_uri_list = []
    related_relevant_lexicals = []

    for lex in relevant_lexicals:
        ssl._create_default_https_context = ssl._create_unverified_context
        framester_endpoint = "http://etna.istc.cnr.it/framester2/sparql"  # get the endpoint API
        my_SPARQL_query = """
            PREFIX wn30schema: <https://w3id.org/framester/wn/wn30/schema/> 
            SELECT DISTINCT ?lexical ?syn WHERE { 
              ?syn wn30schema:containsWordSense ?wordSense . 
              ?wordSense wn30schema:word ?word . 
              ?word wn30schema:lexicalForm ?lexical FILTER (str(?lexical) = """+lex+""") . 
            }
            """
        sparql_wd = SPARQLWrapper(framester_endpoint)  # set the endpoint
        sparql_wd.setQuery(my_SPARQL_query)  # set the query
        sparql_wd.setReturnFormat(JSON)  # set the returned format
        results = sparql_wd.query().convert()  # get the results
        for item in results['results']['bindings']:
            syn_uri_list.append(item['syn']['value'])

    # alla fine delle quattro iterazioni del for loop, la lista syn_uri_list dovrebbe avere 4 elementi,
    # cioè stringhe degli URI dei lexicals rilevanti (es. emotion --> "https://w3id.org/framester/wn/wn30/instances/synset-emotion-noun-1")

    # query per recuperare le forme lessicali partendo dallo URI del synset
    SELECT DISTINCT ?syn ?lexical WHERE { 
      ?syn wn30schema:containsWordSense ?wordSense FILTER (str(?syn) = "https://w3id.org/framester/wn/wn30/instances/synset-emotion-noun-1") . 
      ?wordSense wn30schema:word ?word . 
      ?word wn30schema:lexicalForm ?lexical . 
    }
    # problema: per questo tipo di parole, tipo "emotion", c'è solo una lexical unit --> bisogna trovare un altro
    # modo per trovare le parole semanticamente collegate a emotion. Quindi bisogna trovare altre relazioni 
    '''
    # invece che framester o framenet, per questo task usiamo wordnet con nltk --> iperonimi, iponimi, forme relate e pertanimi
    related_lexicals = []
    for lxc in relevant_lexicals:
        for i, j in enumerate(wordnet.synsets(lxc)):  # for synset in emotion_synsets:
            related_lexicals.extend(list(chain(*[l.lemma_names() for l in j.hypernyms()])))
            related_lexicals.extend(list(chain(
                *[l.lemma_names() for l in j.hyponyms()])))  # for hypo in synset.hyponyms(): print(hypo.lemma_names())
            for x, lemma in enumerate(j.lemmas()):
                related_lexicals.extend(list(chain(*[l.name() for l in lemma.derivationally_related_forms()])))
                related_lexicals.extend(list(chain(*[l.name() for l in lemma.pertainyms()])))
    related_lexicals_clean = []
    for word in related_lexicals:
        if len(word) > 1 and word not in related_lexicals_clean:
            related_lexicals_clean.append(word)

    # ora cerchiamo se queste parole sono tra quelle più frequenti in OS
    emotion_words_in_os = []
    for w in related_lexicals_clean:
        for tuple in freq_dist_pos.most_common(100):
            if w in tuple[0]:
                emotion_words_in_os.append(tuple)
    print(emotion_words_in_os)  # [('feeling', 8), ('desire', 6)]


def nouns_verbs_inspection(f_p):
    os_dict_list = []
    def_dict = {}
    words_count = {}
    list_view = []

    with open(f_p, 'r', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            os_dict_list.append(row)

    # per ogni entry di OS prendi la definizione ma solo la stringa fino al primo segno di interpunzione,
    # poi tokenizzala e puliscila (vedi funzione "indagini") e prendi i noun o noun phrases che ci sono
    # dovrebbero essere quelli che determinano il tipo di entità di cui si sta dando una definizione: se è un'emozione,
    # un'esperienza, uno stato ecc.
    for entry in os_dict_list:
        complete_dfn = entry['definition']
        dfn = re.split('[!()*+.:;?~]', complete_dfn)[0]  # stringa fino al primo segno di interpunzione
        tokens = nltk.word_tokenize(dfn)  # tokenization
        tuples_list_pos = pos_tag(tokens)  # pos tagging
        # a noi interessano: NN noun, singular; NNS noun plural; NNP proper noun, singular; NNPS proper noun, plural;
        # VB verb, base form take; VBD verb, past tense took; VBN verb, past participle taken;
        # VBG verb, gerund/present participle taking --> caso particolare, a volte sono aggettivi altre volte sono verbi che descrivono un'azione
        POS_tags_nouns = ['NN', 'NNS', 'NNP']
        POS_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBN']
        vb_and_nn = []
        for t in tuples_list_pos:
            pos = tuples_list_pos.index(t)
            if t[1] in POS_tags:
                vb_and_nn.append(t)
            elif t[1] == 'VBG' and pos + 1 < len(tuples_list_pos):
                if tuples_list_pos[pos + 1][
                    1] not in POS_tags_nouns:  # se il token che segue il token in -ing NON E' un nome, allora il token che finisce in -ing non è un aggettivo
                    vb_and_nn.append(t)
        # per ogni definizione, salviamo in un dizionario la prima parola che occorre (che sia un nome o un verbo)
        # a definire quell'emozione
        def_dict[entry['emotion_name']] = vb_and_nn[0][0]

    # vediamo se ce ne sono di uguali
    for k, v in def_dict.items():
        if v not in words_count.keys():
            words_count[v] = 1
        else:
            words_count[v] = words_count[v] + 1
    for i in words_count:
        k = (i, words_count[i])
        list_view.append(k)
    list_view.sort(key=lambda x: x[1], reverse=True)
    print(list_view)

    # facciamo un json per salvare ste info
    json_dict = {'entries': def_dict, 'definitions': words_count, 'sorted_def': list_view}
    json_object = json.dumps(json_dict, indent=4)
    with open("../data/words_analysis.json", "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def fred(file_p):
    os_dict_list = []
    with open(file_p, 'r', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            os_dict_list.append(row)

    out = open('../data/FRED_OS_graphs.ttl', 'w')

    headers = {
        'accept': 'text/turtle',
        'Authorization': 'Bearer a7727b8c-aa1e-39d4-8b34-3977ec1c73f5',
    }

    for entry in os_dict_list:
        params = (
            ('text', entry['definition']),
            ('wfd_profile', 'b'),
            ('textannotation', 'earmark'),
            ('wfd', True),
            ('roles', False),  # use FramenNet roles
            ('alignToFramester', True),
            ('semantic-subgraph', True)
        )
        response = requests.get('http://wit.istc.cnr.it/stlab-tools/fred', headers=headers, params=params)
        out.write("#### knowledge graph for: " + entry['emotion_name'] + ": " + entry['definition'] + '\n')
        out.write(response.text + '\n')
        out.write("# -------------------------" + '\n')


def fred_separate(file_p):
    os_dict_list = []
    with open(file_p, 'r', encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            os_dict_list.append(row)

    selected = os_dict_list[:15]

    for entry in selected:
        headers = {
            'accept': 'text/turtle',
            'Authorization': 'Bearer a7727b8c-aa1e-39d4-8b34-3977ec1c73f5',
        }
        params = (
            ('text', entry['definition']),
            ('wfd_profile', 'b'),
            ('textannotation', 'earmark'),
            ('wfd', True),
            ('roles', False),  # use FramenNet roles
            ('alignToFramester', True),  # align to Framester
            ('semantic-subgraph', True)
        )
        response = requests.get('http://wit.istc.cnr.it/stlab-tools/fred', headers=headers, params=params)
        if response.status_code == requests.codes.ok:
            out = open('../data/graphs/{0}_FRED_graph.ttl'.format(entry['emotion_name'].replace(' ', '_')), 'w',
                       encoding='utf-8')
            out.write("#### knowledge graph for: " + entry['emotion_name'] + ": " + entry['definition'] + '\n')
            out.write(response.text + '\n')
            out.write("# -------------------------" + '\n')
            print('knowledge graph for: ' + entry['emotion_name'])
            print(response.text)
            print()
            print()


def compare_graphs(file_p1, file_p2):
    framestercore = Namespace("https://w3id.org/framester/data/framestercore/")
    fred = Namespace("http://www.ontologydesignpatterns.org/ont/fred/domain.owl#")
    DUL = Namespace("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#")

    g1 = rdflib.ConjunctiveGraph()  # create an empty Graph
    g2 = rdflib.ConjunctiveGraph()  # create an empty Graph
    g1.parse(file_p1, format='ttl')  # parse a local RDF file by specifying the format into the graph
    g2.parse(file_p2, format='ttl')  # parse a local RDF file by specifying the format into the graph

    iso1 = to_isomorphic(g1)
    iso2 = to_isomorphic(g2)

    in_both, in_first, in_second = graph_diff(iso1,
                                              iso2)  # Returns three sets of triples: “in both”, “in first” and “in second”

    relevant_classes = set()
    subclass_of = URIRef('http://www.w3.org/2000/01/rdf-schema#subClassOf')
    rdf_type = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
    for s, p, o in in_both.triples((None, None, None)):
        if p == subclass_of or p == rdf_type:
            relevant_classes.add(o)

    print(relevant_classes)


def compare_multiple_graphs(dir_path):
    general_list = []
    for filename in os.listdir(dir_path):
        with open(os.path.join(dir_path, filename).replace("\\", "/")) as f:
            g = rdflib.ConjunctiveGraph()  # create an empty Graph
            g.parse(f, format='ttl')  # parse file as graph
            subclass_of = URIRef('http://www.w3.org/2000/01/rdf-schema#subClassOf')
            rdf_type = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
            for s, p, o in g.triples((None, None, None)):
                if p == subclass_of or p == rdf_type:
                    if len([i for i, v in enumerate(general_list) if v[0] == o]) > 0:
                        r_l = [i for i, v in enumerate(general_list) if v[0] == o]
                        general_list[r_l[0]][1] += 1
                    else:
                        general_list.append([o, 1])

    sorted_list = sorted(general_list, key=lambda x: x[1], reverse=True)

    li_json = list()
    for element in sorted_list:
        li_json.append([element[0].split('/')[-1], element[0], element[1]])

    print(li_json)
    # facciamo un json per salvare ste info
    json_object = json.dumps(li_json, indent=4)
    with open("../data/graphs_analysis.json", "w") as outfile:
        outfile.write(json_object)


# compare_graphs("../data/graphs/aimonomia_FRED_graph.ttl", "../data/graphs/Altschmerz_FRED_graph.ttl")
# compare_multiple_graphs('../data/graphs')
# os_entries = cleaning('../The_Dictionary_of_Obscure_Sorrows.pdf')
# sent_word_freq('../data/OS.csv')
# fred('../data/OS.csv')
# fred_separate('../data/OS.csv')
# nouns_verbs_inspection('../data/OS.csv')