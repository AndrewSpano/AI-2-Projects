import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')


def remove_urls(sentence):
    # remove urls that start with http, https
    no_https = re.sub(r"https?:?[\/]?[\/]?[\S]*", ' ', sentence)
    # remove urls that have extensions, e.g. myspace.com
    no_extensions = re.sub(r"[\S]*\.(com|org|net)[\S]*", ' ', no_https)
    # remove urls that are in the form "www.somepath"
    no_wwws = re.sub(r"www\.[\S]+", ' ', no_extensions)

    # return the result
    result = no_wwws
    return result


def remove_twitter_tags(sentence):
    # remove tags of the form: @jason @ maria, etc
    result = re.sub(r"@[\s]*[\S]*", '', sentence)
    return result


def remove_retweet_token(sentence):
    # remove retweet text "RT"
    result = re.sub(r"(rt|RT)[\s]+", '', sentence)
    return result


def remove_tickers(sentence):
    # remove tickers like $GE
    result = re.sub(r"\$\w*", '', sentence)
    return result


def remove_most_punctuation(sentence):
    # remove most punctuation, except for some that either are emojis or help understand the sentiment (e.g. !, ?)
    result = re.sub(r"(#|\$|%|\^|&|\*|-|_|\+|=|,|\.|<|>|\/|;|\"|`|~|\[|\]|{|})+", '', sentence)
    # separate ":@" from text
    result = re.sub(r":[\s]*@+", " :@ ", result)
    # separate ":)" from text
    result = re.sub(r":[\s]*\)+", " :) ", result)
    # separate ":(" from text
    result = re.sub(r":[\s]*\(+", " :( ", result)
    # separate ":D" from text
    result = re.sub(r":[\s]*D+", " :D ", result)
    # substitute apostrophes (') with empty string (e.g.: don't -> dont)
    result = re.sub(r"'+", '', result)
    # now substitute groups of exclamation marks (!) with one exclamation mark ( ! ) separated by spaces
    result = re.sub(r"!+", " ! ", result)
    # now substitute groups  question marks (?)  with one question mark (?) separated by spaces
    result = re.sub(r"\?+", " ? ", result)
    return result
    

def remove_non_alphanumerical(sentence):
    # remove all non alphanumerical characters
    result = re.sub(r"[^A-Za-z0-9]+", ' ', sentence)
    return result


def remove_numbers(sentence):
    # substitute all numbers with a space
    result = re.sub(r"[0-9]+", ' ', sentence)
    return result


def remove_multiple_whitespace(sentence):
    # substitutes groups of whitespaces with just a space
    result = re.sub(r"[\s]+", ' ', sentence)
    return result


def strip_whitespaces(sentence):
    # removes leading and trailing whitespace
    result = sentence.strip()
    return result


def convert_to_lowercase(sentence):
    # convert every sentence to lowercase
    result = sentence.lower()
    return result


def remove_stopwords(sentence, stop_words):
    # remove stopwrods
    result = ' '.join(list(filter(lambda word: word not in stop_words, sentence.split())))
    return result


def stem_words(sentence, stemmer):
    # stem words
    result = ' '.join(list(map(lambda word: stemmer.stem(word), sentence.split())))
    return result


def main_pipeline(sentence, language="english"):
    """ The main pipeline every sentence should follow """
    # stop_words = stopwords.words(language)
    # stemmer = PorterStemmer()

    result = remove_urls(sentence)
    result = remove_twitter_tags(result)
    result = remove_retweet_token(result)
    result = remove_tickers(result)
    result = remove_most_punctuation(result)
    # result = remove_non_alphanumerical(result)
    result = remove_numbers(result)
    result = remove_multiple_whitespace(result)
    result = strip_whitespaces(result)
    result = convert_to_lowercase(result)
    # result = remove_stopwords(result, stop_words)
    # result = stem_words(result, stemmer)

    return result