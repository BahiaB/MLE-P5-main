import pytest
from final_model import TagsCleaner, HtmlCleaner, TextTokenizer, TextLower,TextLemmatizer, TextStopWordRemover, select_top_n_tags

def test_tags_cleaner():
    cleaner = TagsCleaner()
    text = "<html><body><p>Test</p></body></html>"
    expected_output = [['html', 'body', 'p']]
    assert cleaner.transform([text]) == expected_output

def test_html_cleaner():
    cleaner = HtmlCleaner()
    text = "<html><body><p>Test</p></body></html>"
    expected_output = ["Test"]
    assert cleaner.transform([text]) == expected_output

def test_text_tokenizer():
    tokenizer = TextTokenizer()
    text = "This is a test."
    expected_output = ['This', 'is', 'a', 'test', '.']
    assert tokenizer.transform([text]) == [expected_output]

def test_text_lower():
    lower = TextLower()
    text = ['This', 'Is', 'A', 'Test']
    expected_output = ['this', 'is', 'a', 'test']
    assert lower.transform([text]) == [expected_output]

def test_text_stop_word_remover():
    remover = TextStopWordRemover()
    text = ['this', 'is', 'a', 'test']
    expected_output = ['test']  # assuming 'this', 'is', 'a' are in the stop words list
    assert remover.transform([text]) == [expected_output]

def test_text_lemmatizer():
    lemmatizer = TextLemmatizer()
    text = ['running', 'tests']
    expected_output = ['run', 'test']  # assuming the lemmatizer converts 'running' to 'run' and 'tests' to 'test'
    assert lemmatizer.transform([text]) == [expected_output]

def test_select_top_n_tags():
    import numpy as np
    probabilities = np.array([[0.1, 0.3, 0.5, 0.2, 0.4],
                             [0.6, 0.8, 0.7, 0.9, 0.5]])
    threshold = 0.4
    top_n = 3
    expected_output = np.array([[0, 0, 1, 0, 1],
                               [1, 1, 1, 1, 0]])
    assert select_top_n_tags(probabilities, threshold, top_n) == expected_output

'''def test_vectorize_data(data):
    # create and instanciate Vectorisation
    data = ['This is a test']
    vectorized_data = pipeline_tfidf.transform(data)
    return vectorized_data'''