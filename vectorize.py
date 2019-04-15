from sklearn.feature_extraction.text import TfidVectorizer


def vectorize(X_train['review_text']):
	vectorizer = TfidVectorizer()
	vectorized_review = vectorizer.fit_transform(X_train['review_text'])
	return vectorized_review

vectorize()