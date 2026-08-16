def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
print(f"P(sick|positive) = {result:.4f}")

import math
from collections import defaultdict


class NaiveBayes:
    def __init__(self, smoothing=1.0, short_threshold=3):
        self.smoothing = smoothing
        self.short_threshold = short_threshold
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.short_counts = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            if len(words) <= self.short_threshold:
                self.short_counts[label] += 1
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)
            is_short = len(words) <= self.short_threshold
            short_count = self.short_counts[cls]
            long_count = self.class_counts[cls] - short_count
            if is_short:
                score += math.log(
                    (short_count + self.smoothing)
                    / (self.class_counts[cls] + self.smoothing * 2)
                )
            else:
                score += math.log(
                    (long_count + self.smoothing)
                    / (self.class_counts[cls] + self.smoothing * 2)
                )
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                score += math.log(
                    (count + self.smoothing) / (total + self.smoothing * vocab_size)
                )
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class


train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
]

classifier = NaiveBayes()
classifier.train(train_docs, train_labels)

test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]

for msg in test_messages:
    print(f"  '{msg}' -> {classifier.predict(msg)}")


def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]
    probs = {}
    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)
        probs[word] = (count + classifier.smoothing) / (
            total + classifier.smoothing * vocab_size
        )
    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for word, prob in sorted_words[:n]:
        print(f"    {word}: {prob:.4f}")


print("\nTop spam words:")
show_top_words(classifier, "spam")
print("\nTop ham words:")
show_top_words(classifier, "ham")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_docs)
clf = MultinomialNB()
clf.fit(X_train, train_labels)

X_test = vectorizer.transform(test_messages)
predictions = clf.predict(X_test)
for msg, pred in zip(test_messages, predictions):
    print(f"  '{msg}' -> {pred}")


# ex1
secondPosResult = bayes(prior=result, likelihood=0.99, false_positive_rate=0.01)
print(f"P(sick|positive^2) = {secondPosResult:.4f}")

# ex2
classifier_001 = NaiveBayes(smoothing=0.01)
classifier_001.train(train_docs, train_labels)
for msg in test_messages:
    print(f"  001:'{msg}' -> {classifier_001.predict(msg)}")
print("\nTop spam words:")
show_top_words(classifier_001, "spam")
print("\nTop ham words:")
show_top_words(classifier_001, "ham")

classifier_01 = NaiveBayes(smoothing=0.1)
classifier_01.train(train_docs, train_labels)
for msg in test_messages:
    print(f"  01:'{msg}' -> {classifier_01.predict(msg)}")
print("\nTop spam words:")
show_top_words(classifier_01, "spam")
print("\nTop ham words:")
show_top_words(classifier_01, "ham")

classifier_1 = NaiveBayes(smoothing=1)
classifier_1.train(train_docs, train_labels)
for msg in test_messages:
    print(f"  1:'{msg}' -> {classifier_1.predict(msg)}")
print("\nTop spam words:")
show_top_words(classifier_1, "spam")
print("\nTop ham words:")
show_top_words(classifier_1, "ham")

# classifier_0 = NaiveBayes(smoothing=0)
# classifier_0.train(train_docs, train_labels)
# for msg in test_messages:
#     print(f"  0:'{msg}' -> {classifier_0.predict(msg)}")
# print("\nTop spam words:")
# show_top_words(classifier_0, "spam")
# print("\nTop ham words:")
# show_top_words(classifier_0, "ham")
