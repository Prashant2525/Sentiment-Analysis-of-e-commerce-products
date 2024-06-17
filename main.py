import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
nltk.download('punkt')

file = 'input_data.csv'
dfs = pd.read_csv(file)

product_sentiments = {}

# Load spaCy model after downloading it
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("SpaCy model 'en_core_web_sm' not found. Please download the model using 'python -m spacy download en_core_web_sm'.")

def analyze_product_sentiment(product_name):
    matching_products = dfs[dfs['productTitle'].str.lower().str.contains(product_name.lower())]['productTitle'].unique()

    if len(matching_products) == 0:
        print(f"No products found matching '{product_name}'.")
    elif len(matching_products) == 1:
        product_reviews = dfs[(dfs['productTitle'].str.lower() == matching_products[0].lower()) & (dfs['reviewDescription'].notnull())]['reviewDescription'].tolist()
        calculate_sentiment(matching_products[0], product_reviews)
        lda_output = topic_modeling(product_reviews)
        # aspect_sentiments = aspect_sentiment_analysis(product_reviews)
    else:
        print("Multiple products found matching your input:")
        for idx, product in enumerate(matching_products, start=1):
            print(f"{idx}. {product}")
        choice = input("Enter the number of the specific product you want to analyze: ")
        if choice.isdigit() and int(choice) in range(1, len(matching_products) + 1):
            chosen_product = matching_products[int(choice) - 1]
            product_reviews = dfs[(dfs['productTitle'].str.lower() == chosen_product.lower()) & (dfs['reviewDescription'].notnull())]['reviewDescription'].tolist()
            calculate_sentiment(chosen_product, product_reviews)
            lda_output = topic_modeling(product_reviews)
            # aspect_sentiments = aspect_sentiment_analysis(product_reviews)
        else:
            print("Invalid choice.")

def display_reviews(reviews):
    print("Here are some of the reviews of the product you searched for:\n")
    for review in reviews:
        print("+" + "-" * 78 + "+")  # Border
        print("| {:^76} |".format(review))  # Review content
        print("+" + "-" * 78 + "+")  # Border
    print("\n")

def calculate_sentiment(product_name, product_reviews):
    sid = SentimentIntensityAnalyzer()
    product_scores = []

    print(f"\nSentiment analysis for product '{product_name}':\n")

    for review in product_reviews:
        tokens = word_tokenize(review)
        print(f"Tokens: {tokens}")

        ss = sid.polarity_scores(review)
        print(f"Review: {review}")
        print(f"Negative Score: {ss['neg']:.4f}")
        print(f"Neutral Score: {ss['neu']:.4f}")
        print(f"Positive Score: {ss['pos']:.4f}")
        print(f"Compound Score: {ss['compound']:.4f}")
        print('-' * 111)

        product_scores.append(ss['compound'])

    avg_score = sum(product_scores) / len(product_scores)
    product_sentiments[product_name] = avg_score

    print(f"Average Sentiment Score: {avg_score:.4f}")

    if avg_score >= 0.05:
        comment = "It's recommended to buy this product."
    elif avg_score <= -0.05:
        comment = "It's not recommended to buy this product."
    else:
        comment = "You may consider buying this product based on other factors."

    print(f"Conclusion of Product: {comment}\n")


def topic_modeling(reviews):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(reviews)

    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_output = lda_model.fit_transform(tfidf)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    topic_probabilities = lda_model.transform(tfidf)

    return lda_output, feature_names, topic_probabilities

# def aspect_sentiment_analysis(reviews):
#     aspect_sentiments = {}
#
#     for review in reviews:
#         aspects = extract_aspects(review)
#         for aspect in aspects:
#             aspect_sentiment = analyze_sentiment_aspect(review, aspect)
#             if aspect in aspect_sentiments:
#                 aspect_sentiments[aspect].append(aspect_sentiment)
#             else:
#                 aspect_sentiments[aspect] = [aspect_sentiment]
#
#     return aspect_sentiments

# def extract_aspects(review):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(review)
#     entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG']]
#     aspects = [entity[0] for entity in entities]
#     return aspects

def analyze_sentiment_aspect(review, aspect):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(review)
    aspect_sentiment = ss['compound']
    return aspect_sentiment

def analyze_all_products_sentiment():
    sid = SentimentIntensityAnalyzer()

    for product_name in dfs['productTitle'].unique():
        product_reviews = dfs[(dfs['productTitle'] == product_name) & (dfs['reviewDescription'].notnull())]['reviewDescription'].tolist()

        if product_reviews:
            product_scores = []

            for review in product_reviews:
                ss = sid.polarity_scores(review)
                product_scores.append(ss['compound'])

            avg_score = sum(product_scores) / len(product_scores)
            product_sentiments[product_name] = avg_score

def plot_all_products_sentiments():
    total_products = len(product_sentiments)
    print(f"Total number of Unique products: {total_products}")

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(product_sentiments)), list(product_sentiments.values()), c=list(product_sentiments.values()), cmap='coolwarm')
    plt.xlabel('')  # Empty x-label
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis of Products')
    plt.colorbar(label='Sentiment Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


product_name = input("Enter the product name for which you want reviews: ")

analyze_product_sentiment(product_name)

analyze_all_products_sentiment()
matching_products = dfs[dfs['productTitle'].str.lower().str.contains(product_name.lower())]['productTitle'].unique()
product_reviews = dfs[(dfs['productTitle'].str.lower() == matching_products[0].lower()) & (dfs['reviewDescription'].notnull())]['reviewDescription'].tolist()

# Call topic modeling and aspect sentiment analysis
lda_output, feature_names, topic_probabilities = topic_modeling(product_reviews)
# aspect_sentiments = aspect_sentiment_analysis(product_reviews)

def display_topic_modeling_results(lda_output, feature_names, topic_probabilities):
    print("Topic Modeling Results:")
    for topic_idx, topic in enumerate(lda_output):
        top_words_indices = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices if i < len(feature_names)]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
        print("Topic Probability Scores:")
        print(topic_probabilities[topic_idx])
        print()


print("Topic Modeling Results:")
print(lda_output)
display_topic_modeling_results(lda_output, feature_names, topic_probabilities)
# print("\nAspect Sentiment Analysis Results:")
# print(aspect_sentiments)

# You can also call extract_aspects for individual reviews
# for review in product_reviews:
#     aspects = extract_aspects(review)
#     print("Extracted Aspects:")
#     print(aspects)

plot_all_products_sentiments()


