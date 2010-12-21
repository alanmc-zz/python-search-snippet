#!/usr/bin/python

import sys
from itertools import izip

PREFERRED_SNIPPET_LENGTH = 40
MAX_SNIPPET_LENGTH = 60
SNIPPET_MATCH_WINDOW_SIZE = 5

def normalize_term(term):
    """Strips leading/trailing punctuation and lowercases terms. 

    >>> normalize_term("Hello!!!")
    'hello'

    >>> normalize_term("My nAme is Alan.")
    'my name is alan'

    >>> normalize_term("Normalize!")
    'normalize'

    """

    return term.strip('!,.?').lower()

def get_normalized_terms(query):
    """Gets the list of normalized query terms. Does some
    extremely naive stemming. Assumption is that in a real system the
    search engine would return back the matched mispelled term set

    >>> get_normalized_terms("Hello, GoodBye!!")
    ['hello', 'goodbye', 'hellos', 'goodbyes', 'helloes', 'goodbyees']

    """

    terms = [normalize_term(term) for term in query.split()] + \
            [normalize_term(term) + 's' for term in query.split()] + \
            [normalize_term(term) + 'es' for term in query.split()]

    return terms

def list_range(x):
    """Returns the range of a list.                                                                                                                                                                      

    >>> list_range([9, 7, 8, 6])
    3

    """

    return max(x) - min(x)

def get_window(positions, indices):
    """Given a list of lists and an index for each of those lists,                                                                                                                                        
    this returns a list of all of the corresponding values for those                                                                                                                                      
    lists and their respective index.                                                                                                                                       
              
    >>> get_window([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [0, 1, 2]) 
    [1, 5, 9]

    """

    return [word_positions[index] for word_positions, index in \
            izip(positions, indices)]

def get_min_index(positions, window):
    """Given a list of lists representing term positions in a corpus,
    this returns the index of the min term, or nothing if none left

    >>> get_min_index([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 5, 9]) 
    0

    >>> get_min_index([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [3, 5, 9])
    1

    >>> get_min_index([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [3, 6, 9]) 

    """

    min_indices = (window.index(i) for i in sorted(window))
    for min_index in min_indices:
        if window[min_index] < positions[min_index][-1]:
            return min_index

def get_shortest_term_span(positions):
    """ Given a list of positions in a corpus, returns the shortest span
    of words that contain all query terms

    >>> get_shortest_term_span([[0, 5, 10, 15], [1, 3, 6, 9], [4, 8, 16, 21]])
    [5, 3, 4]

    """

    # Initialize our list of lists where each list corresponds to the
    # locations within the document for a term
    indices = [0] * len(positions)

    min_window = window = get_window(positions, indices)

    # Iteratively moving the minimum index forward finds us our
    # minimum span
    while True:

        min_index = get_min_index(positions, window)

        if min_index == None:
            return min_window

        indices[min_index] += 1

        window = get_window(positions, indices)

        if list_range(min_window) > list_range(window):
            min_window = window

        if list_range(min_window) == len(positions):
            return min_window

def get_highest_density_window(doc, query, size=PREFERRED_SNIPPET_LENGTH):
    """ Given a list of positions in a corpus, returns the shortest span
    of words that contain all query terms

    >>> get_highest_density_window('this case is a good test', 'test case', size=5)
    (1, 6)

    >>> get_highest_density_window('this case is a good test case to test', 'test case', size=5)
    (4, 9)

    >>> get_highest_density_window('good test case', 'test case', size=2)
    (1, 3)

    >>> get_highest_density_window('no terms match', 'this case', size=2)
    (0, 2)

    >>> get_highest_density_window('size is too big', 'this case', size=10)
    (0, 4)

    """

    terms = get_normalized_terms(query)

    document_words = doc.split()
    document_size = len(document_words)

    count = high_count = 0
    high_count = 0
    start_index = read_start = 0
    end_index = read_end = start_index + size - 1

    # If the document is shorter than the desired window, just return the doc
    if document_size < size:
        return (start_index, document_size)

    # calculate the # of term occurances in the initial window
    for i in xrange(read_start, read_end):
        if normalize_term(document_words[i]) in terms:
            count += 1
            high_count += 1

    read_start += 1
    read_end += 1

    # Use a "sliding window" technique to count occurences
    # Move the window one work at a time. After each iteration if we've
    # picked up a matched term term increment. Decrement if we just dropped
    # a matched term
    while read_end < document_size:
        if document_words[read_start - 1] in terms:
            count -= 1
            
        if document_words[read_end] in terms:
            count += 1
            
        if count > high_count:
            high_count = count
            start_index = read_start
            end_index = read_end

        read_start += 1
        read_end += 1

    # Return end_index + 1 so that callers can more intuitively use
    # non-inclusive range operaters
    return (start_index, end_index + 1)
    
def generate_term_positions(doc, query):
    """ Iterates over the words in the corpus and stores the locations of
    each matched query term. This data is structured as a list of lists, where
    each sub-list contains all of the positions for a matched query term

    This is a fairly expensive process, and would more ideally be returned
    as doc data by the search engine

    """

    terms = query.split()

    positions = [[] for j in range(len(terms))]

    for i, word in enumerate(doc.split()):
        for term in terms:
            if normalize_term(word) in get_normalized_terms(term):
                positions[terms.index(term)].append(i)
                break


    # Sometimes a term doesn't appear at all in the returned document
    # Remove the location list for any terms that has no locations within
    # the document
    positions = [x for x in positions if x]

    return positions

def shorten_snippet(doc, query):
    """ Iterates over the words in the snippet and attempts to "close the
    gap" between matched terms in an overly long snippet.  Naive implementation.

    >>> shorten_snippet("test blah blah blah blah blah blah case", "test case")
    'test blah blah blah blah blah ... case'

    """

    flattened_snippet_words = []
    normalized_terms = get_normalized_terms(query)
    
    last_term_appearence = 0
    skipping_words = False

    for i, word in enumerate(doc.split()):

        # Spotted a matched term, set our state flag to false and update
        # the "time" of our last term appearance 
        if word in normalized_terms:
            last_term_appearence = i
            skipping_words = False

        # If it's been too long since our last match, start dropping words
        if i - last_term_appearence > SNIPPET_MATCH_WINDOW_SIZE:

            # Only want to add "..." once between terms, so check our state flag first
            if not skipping_words:
                flattened_snippet_words.append("...")
                skipping_words = True

            continue

        flattened_snippet_words.append(word)

    return ' '.join(flattened_snippet_words)

def highlight_query_terms(doc, 
                          query, 
                          highlight_start='[[HIGHLIGHT]]', 
                          highlight_end='[[ENDHIGHLIGHT]]'):
    """ Iterates over the words in the corpus first stores the start and end index
    of each "span" of matched query terms, where a "span" is one or more 
    consecutive matched query terms

    Then iterates over all spans and prefixes the starts and ends with the supplied
    highlight strings.

    """

    normalized_terms = get_normalized_terms(query)

    highlight_spans = []
    document_words = doc.split()

    start_span = None
    
    for i, word in enumerate(document_words):

        # If the word we're inspecting matches the given query terms
        # we want to do one of two things. Either start a new span if
        # we haven't already, or just keep moving along.
        #
        # If the word we're inspecting DOESN'T match, then check
        # to see if we've started a span.  If so, end it with the previous
        # word.  If not, move along.
        if normalize_term(word) in get_normalized_terms(query):       
            if start_span is None:
                start_span = i
        else:
            if start_span is not None:
                highlight_spans.append( (start_span, i - 1) )
                start_span = None

    # Our span algorithm completes spans by checking for non-matching terms
    # If the document ends with matching terms we'll never hit a non-matching
    # term to pick them up, so we do it here. If we're in the middle of a span
    # when the document ends, add the current span.
    if start_span is not None:
        highlight_spans.append( (start_span, i) )

    for span in highlight_spans:
        document_words[span[0]] = highlight_start + document_words[span[0]]
        document_words[span[1]] = document_words[span[1]] + highlight_end

    return ' '.join(document_words)

def highlight_doc_shortest_span(doc, query, max_length=PREFERRED_SNIPPET_LENGTH):
    """Returns the highlighted snippet using the "shortest span" method
    which assumes users want to see the snippet which incorporates the most
    distinct matched query terms in the shortest span.

    A potential side effect of this function is that various whitespace may be
    removed from the doc snippet, making the snippet more readable.

    Args:
    doc - String that is a document to be highlighted
    query - String that contains the search query
    
    Returns:
    The the most relevant snippet with the query terms highlighted.

    """
    
    positions = generate_term_positions(doc, query)

    if not positions:
        return highlight_query_terms(' '.join(doc.split()[0: max_length]).strip(), query)

    span = get_shortest_term_span(positions)
    
    start = max(0, span[0] - (PREFERRED_SNIPPET_LENGTH / 2))
    end = min(len(doc.split()), span[len(positions) - 1] + (PREFERRED_SNIPPET_LENGTH / 2))
    
    snippet = ' '.join(doc.split()[start:end+1])

    if (end - start > MAX_SNIPPET_LENGTH):
        snippet = flatten_snippet(snippet, query)

    return highlight_query_terms(snippet.strip(), query)


def highlight_doc_density(doc, query, size=PREFERRED_SNIPPET_LENGTH):
    """Returns the highlighted snippet using the "highest density" method,
    which assumes that users want to see the snippet with the highest density of
    matched search terms.

    A potential side effect of this function is that various whitespace may be
    removed from the doc snippet, making the snippet more readable.

    Args:
    doc - String that is a document to be highlighted
    query - String that contains the search query
    
    Returns:
    The the most relevant snippet with the query terms highlighted.

    """

    window = get_highest_density_window(doc, query, size=size)
    return highlight_query_terms(' '.join(doc.split()[window[0]:window[1]]).strip(), query)


def main():
    print highlight_doc_shortest_span(open(sys.argv[1]).read(), 'deep dish pizza')
    print '--- '
    print highlight_doc_density(open(sys.argv[1]).read(), 'deep dish pizza')

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
