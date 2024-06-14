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

