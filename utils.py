import string
import re
from collections import Counter

def get_answer_txt(start, end, contxt):
	'''
	start: start index int
	end: end index int ##Note: 'end' is inclusive
	contxt: context sentence
	'''
	c_words = contxt.split()
	return " ".join(c_words[start:end+1])

def count_tokens(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    return num_same, len(prediction_tokens), len(ground_truth_tokens)

def f1_score(common_len, pred_len, grnd_len):
    if common_len == 0: return common_len
    precision = 1.0 * common_len / pred_len
    recall = 1.0 * common_len / grnd_len
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))
