import re
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

stopword_list = stopwords.words('english')
'''
["c'mon", 'tb', 'one', 'cn', 'indicated', 'course', 'thanks', 'inasmuch', 'av', 'ge', 'six', 'sd', 'owing', 
'hereupon', 'fifteen', 'gr', 'rq', 'let', 'changes', 'bd', 'thru', 'contain', 'sometimes', 'primarily', 'sixty', 'e3', 'vj', "here's", 'ca', 'stop', '3a', 
'dx', 'gone', 'io', 'please', '0o', "he'll", 'resulted', 'nobody', 'gave', 'lj', 'various', 'insofar', 'ever', 'pf', 'fix', 'xv', 'k', 'pages', 'whole', 
'sc', 'fl', 'im', 'y2', 'wherein', 'adj', 'regarding', 'eight', 'known', 'xx', 'clearly', 'nl', 'b2', 'itd', 'sent', 'date', 'per', 'vd', 'mt', 'kept', 
'taking', 'hy', 'end', 'whats', 'though', 'cl', 'inward', 'best', 'i7', 'certain', 'eg', 'xs', 'everyone', 'noted', 'az', 'substantially', 'p3', 'ignored', 
'eleven', 'hu', 'ej', 'promptly', 'never', 'ln', 'pd', 'immediately', 'regards', 'wed', 'whereupon', 'pas', 'interest', 'research-articl', 'um', 'js', 'c3',
'cit', 'sensible', 'eo', 'a2', 'qu', 'concerning', 'detail', 'information', "c's", 'u201d', 'nearly', 'whatever', 'th', 'importance', 'du', 'inc', 'rf', 
'instead', 'hid', 'took', 'mill', 'c1', 'ns', 'en', 'jt', 'novel', "she'd", 'dp', 'part', 'usually', 'rt', 'cant', 'regardless', 'whereas', 'either', 'lest',
'must', 'ba', 'appear', 'fill', 'di', '3b', 'apparently', 'i8', 'refs', 'cc', 'seriously', 'gotten', 'iy', 'r', 'ko', 'co', 'cp', 'showed', 'ten', 'line', 
'fn', "they'll", 'iv', 'know', 'namely', 'whoever', 'e', 'xl', 'far', 'specifying', 'amongst', 'definitely', 'sufficiently', 'outside', 'already', 'came', 
'si', 'whether', 'alone', 'dj', 'werent', 'ue', 'cg', 'ord', 'sec', 'mean', 'might', 'ls', 'behind', 'bp', 'least', 'ds', '6o', 'b3', 'act', 'three', 'fr', 
'arent', 'fj', 'yet', 'yj', "t's", 'r2', 'usefulness', 'oz', 'truly', 'latterly', 'somethan', 'dl', 'want', 'second', 'v', "we're", 'obtained', 'indicate',
'con', 'pj', 'st', 'ev', 'via', 'welcome', 'giving', 'df', 'shown', 'rs', 'words', 'zi', 'couldnt', 'los', 'pl', 'third', 'nonetheless', 'unlikely', 
'resulting', 'away', 'et', 'got', 'sz', 'ce', 'soon', 'hes', 'iz', 'normally', 'hed', 'ol', 'le', 'da', 'les', 'new', 'else', 'xk', 'used', 'unlike',
'lo', 'em', 'ia', 'anyhow', 'said', 'x1', 'mo', 'a4', 'rh', 'wants', 'say', 'anything', 'theyre', "there'll", 'sorry', 'us', 'top', 'cry', 'ok',
'widely', 'de', 'thereto', 'vols', 'ru', 'tends', 'bottom', "that's", 'especially', 'xf', 'contains', 'mn', 'move', 'everything', 'aside', 'twelve', 
'begins', 'ending', 'help', 'someone', 'indicates', 'lb', 'nt', 'therere', 'two', 'whence', 'therefore', 'give', 'lets', 'td', 'becoming', 'allow', 'xo',
'sa', 'somewhere', 'containing', 'ti', 'uj', 'pe', 'i4', 'pn', 'arise', 'previously', 'jj', 'also', 'anyways', 'seeming', 'fu', 'begin', 'among', 'apart',
'mine', 'fa', 'uo', 'f', 'successfully', "there've", 'thousand', 'tm', 'done', 'wasnt', 'way', 'whereby', 'ot', 'yes', 'uses', 'forth', 'come', 'ask',
'elsewhere', "why's", 'ups', 'usefully', 'ml', 'better', 'briefly', 'miss', 'amount', 'sure', 'towards', 'bx', 'ry', 'inner', 'auth', 'consequently',
'dy', 'self', "how's", 'run', 'hereafter', 'accordingly', "where's", 'significant', 'made', 'reasonably', "i'm", 't2', 'gi', 'ig', 'specified',
'particular', 'well', 'section', 'accordance', 'beginnings', 'kj', 'seven', 'whenever', "they've", 'ui', 'ei', 'id', "what'll", 'whod', 'rm',
'serious', 'back', 'forty', 'i2', 'tq', 'first', "they'd", 'ft', 'certainly', 'neither', 'mg', 'kg', 'nos', 'nc', 'ow', 'possible', 'wherever',
'twice', 'pr', 'currently', "who'll", 'lt', 'fire', 'hr', 'relatively', 'readily', 'tc', 'described', 'ff', 'l2', 'm2', 'nj', 'thickv', 
'thereof', 'try', 'cm', 'however', 'mr', '6b', 'etc', 'somehow', 'trying', 'largely', 'afterwards', 'la', 'rather', 'sj', 'mu', 'within',
'able', 'non', 'wheres', 'fify', 'hj', 'need', 'upon', 'unto', 'sincere', 'z', 'several', 'zero', 'tx', 'tv', 'brief', 'un', 'considering',
 "we'll", 'ibid', 'mug', 'lr', 'thoroughly', 'seem', 'volumtype', 'although', 'whomever', 'ones', 'needs', 'ke', 'hardly', 'anymore', 'entirely',
'til', 'useful', "he's", 'twenty', 'gets', "she'll", 'oh', 'op', 'following', 'cy', 'ho', 'much', 'since', 'use', 'po', 'sq', 'become', 'probably',
'pu', 'related', 'com', 'ee', 'ny', 'aj', "we've", 'sn', 'bk', 'affected', 'call', 'former', 'el', 'rr', 'thered', 'ao', 'hh', 'anyone',
'oq', 'ic', 'saw', 't3', 'tries', 'beside', 'none', 'ob', 'selves', '0s', 'iq', 'vu', 'ps', 'meantime', 'xt', 'according', 'describe', 
'presumably', 'research', 'due', 'thus', 'na', 'nowhere', 'corresponding', 'likely', 'nr', 'shows', 'ef', 'ec', 'another', "that've", 'sub',
'ib', 'oa', 'u', 'sr', 'toward', 'bs', 'cx', 'significantly', 'beyond', 'bu', 'tl', 'wi', 'b1', 'specify', 'something', 'mrs', 'causes', 
'howbeit', 'near', 'ran', 'bl', 'whos', 'dc', 'overall', 'jr', 'plus', 'added', 'ju', 'awfully', 'others', 'ap', 'similarly', 'par', 'theyd',
'ci', 'latter', 'eighty', 'ph', 'ts', 'os', 'au', 'makes', 'ah', 'fy', 'approximately', 'wo', 'lf', 'a3', "when's", 'old', 'h', 'vq', 'b', 
'throughout', 'found', 'cq', 'biol', 'g', 'beginning', "ain't", 'therein', 'past', 'rj', 'que', 'f2', 'anyway', 'would', 'sup', 'allows', 
'always', 'vt', 'vs', 'respectively', 'comes', 'al', 'amoungst', 'wa', '3d', 'nd', 'est', 'p1', 'side', 'important', 'omitted', 'fi', 'x', 
'go', 'effect', 'suggest', "a's", 'sometime', 'h2', 'world', 'little', 'yl', 'va', 'consider', 'necessarily', 'ninety', 'ac', 'using', 'tell',
'cs', 'front', 'formerly', 'pm', 'similar', 'n2', 'sf', 'sy', 'whither', 'od', 'ie', 'secondly', 'ra', 'enough', 'eq', 'thence', 'q', 
'becomes', 'om', 'even', 'l', 'tip', 'predominantly', 'happens', 'uk', 'ri', 'says', 'thoughh', 'cannot', "i've", 'gl', 'seen', 'hasnt', 
'tr', 'despite', 'shall', 'thin', 'xj', "i'd", 'seemed', 'put', 'sl', 'maybe', 'anywhere', "we'd", 'available', 'qv', 'i3', 'oc', 'fs',
'youd', 'c', 'going', 'right', 'looks', 'think', 'furthermore', 'slightly', 'less', 'specifically', 'tn', 'p2', 'km', 'possibly', 'thereby',
'j', 't1', 'cd', 'bill', 'get', 'thereupon', 'ur', 'bt', 'dk', 'may', 'necessary', 'mostly', 'went', "who's", 'ys', 'thanx', 'viz', 'nay', 
'knows', 'seeing', 'means', 'cause', 'showns', 'dt', 'noone', 'still', 'ox', 'keeps', 'oi', 'tt', 'www', 's2', 'keep', 'hither', 'cr', 'given', 
'unless', 'full', 'pt', 'somewhat', 'e2', 'xn', 'ut', 'obtain', 'many', 'like', 'anybody', 'pi', 'cv', 'ad', 'ag', 'hopefully', 'ii', 'rv', 'cj', 
"they're", 'ni', 'tj', 'vo', 'otherwise', 'ep', 'bn', 'cu', 'placed', 'throug', 'ro', 'find', "can't", 'looking', 'c2', 'wish', 'er', 'nevertheless', 
'willing', 'show', 'i6', 'obviously', 'make', 'whim', 'taken', 'perhaps', 'yr', 'actually', 'ou', 'af', 'thats', 'ix', 'tried', 'potentially', 'gives', 
'hello', 'along', 'asking', 'ed', 'gs', 'home', 'almost', 'et-al', 'associated', 'vol', 'ir', 'provides', 'thank', 'wont', 'nine', 'ltd', 'edu', 'okay', 
'liked', 'youre', 'h3', 'everybody', "it'll", 'announce', 'og', 'bi', 'ea', 'last', 'really', 'moreover', 'seems', 'greetings', 'd2', 'rn', 'without', 
'qj', 'bc', 'affects', "he'd", 'rl', 'proud', 'together', 'except', 'herein', 'everywhere', 'often', 'merely', 'il', 'oj', 'dd', 'four', 'recent', 
'somebody', 'sp', 'lc', 'thou', 'onto', 'system', 'affecting', 'tf', 'x2', 'goes', 'rc', 'saying', 'ar', 'ch', 'five', 'eu', 'million', 'x3', 'value', 
'es', 'zz', 'downwards', 'index', 'w', 'py', 'a1', 'whereafter', 'meanwhile', 'immediate', 'recently', 'became', 'hi', 'pk', 'ay', 'ij', 'quickly', 
'wouldnt', 'beforehand', 'hs', 'cf', 'later', 'ab', 'shed', 'ct', 'followed', 'results', 'ne', 'ng', 'poorly', 'te', 'bj', 'different', 'unfortunately', 
'br', 'oo', 'tp', 'shes', 'name', 'next', 'cz', 'pq', 'pc', 'well-b', 'present', 'appropriate', "i'll", 'page', 'every', 'theres', 'exactly', 'hundred', 
'getting', 'ss', 'example', 'particularly', 'appreciate', "let's", 'thereafter', 'ref', 'around', 'wonder', 'ought', 'hence', 'nn', 'hereby', 'pagecount', 
'rd', 'believe', 'se', 'invention', 'fo', 'heres', 'pp', 'could', 'ga', 'nothing', 'besides', "it'd", 'aw', 'ey', 'ms', 'xi', 'thorough', 'gy', "there's", 
'mainly', 'p', 'gj', 'take', 'across', 'ip', "what's", 'empty', 'lately', 'indeed', 'sm', 'strongly', 'yt', 'ax', 'dr', 'ae', 'fc', 'ih', 'fifth', 'abst', 
'see', 'http', 'whose', 'quite', 'n', 'look', 'ex', 'follows']
'''

def convert_to_lowercase(sentence):
    lowercase = sentence.lower()
    return lowercase


def word_tokenizer(sentence):
    tokenized_word = word_tokenize(sentence)
    return tokenized_word

def sent_tokenizer(sentence):
    tokenized_sent = sent_tokenize(sentence)
    return tokenized_sent


regex = re.compile('[%s]' % re.escape(string.punctuation+"“”’"))

def punctuation_removal(tokens):
    tokenized_no_punc = [] #list of tokenized words
    for token in tokens:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            tokenized_no_punc.append(new_token)
    return tokenized_no_punc

def stopwords_removal(tokens):
    tokenized_no_stopwords = []
    for token in tokens:
        if not token.lower() in stopword_list:
            tokenized_no_stopwords.append(token)
    return tokenized_no_stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
wordnet = WordNetLemmatizer()

preprocessed_docs = []
def stemming(tokens):
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(wordnet.lemmatize(token))
    return stemmed_tokens

def hashtag_extractions(sentence):
    hashtag_list = []
    for token in sentence.split():
        if token[0] == '#':
            hashtag_list.append(token)
    return hashtag_list
