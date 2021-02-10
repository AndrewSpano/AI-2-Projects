import re


def remove_urls(corpus):
    # remove urls that start with http, https
    no_https = re.sub(r'https?:?[\/]?[\/]?[\S]*', ' ', corpus)
    # remove urls that have extensions, e.g. myspace.com
    no_extensions = re.sub(r'[\S]*\.(com|org|net)[\S]*', ' ', no_https)
    # remove urls that are in the form 'www.somepath'
    no_wwws = re.sub(r'www\.[\S]+', ' ', no_extensions)

    # return the result
    result = no_wwws
    return result


def remove_references(corpus):
    # remove references to bibliography of the fort [1, [2], [3], etc.
    result = re.sub('(\[[0-9]+(\]|,)|[0-9]+\])', '', corpus)
    return result


def remove_multiple_full_stops(corpus):
    # remove mutliple consecutive full stops, as they break sentence tokenization
    result = re.sub('\.+', '.', corpus)
    return result


def remove_et_al(corpus):
    # remove the 'et al.' strings, as the dot breaks tokenization
    result = re.sub('et al. ', '', corpus)
    return result


def remove_figure_references(corpus):
    # remove references to figures, of the form: (graph showed in Fig 1)
    result = re.sub('\([^\)]*?(?=Fig)Fig.*?(?=\))\)', '', corpus)
    return result


def remove_multiple_whitespace(corpus):
    # substitutes groups of whitespaces with just a space
    result = re.sub(r"[\s]+", ' ', corpus)
    return result
