import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

file = 'amazon_sentiment.xlsx'
xl = pd.ExcelFile(file)
dfs = xl.parse(xl.sheet_names[0])


unique_products = dfs['product_name'].unique()

product_sentiments = {}

product_name = input("Enter the product name for which you want reviews: ")

product_reviews = dfs[dfs['product_name'] == product_name]['review_text'].tolist()

if not product_reviews:
    print(f"No reviews found for product '{product_name}'.")
else:
    sid = SentimentIntensityAnalyzer()
    product_scores = []
    for review in product_reviews:
        ss = sid.polarity_scores(review)
        print(f"Review of {product_name}", review)
        product_scores.append(ss['compound'])

    avg_score = sum(product_scores) / len(product_scores)
    product_sentiments[product_name] = avg_score

    print(f"\nSentiment analysis for product '{product_name}':")
    print(f"Average Sentiment Score: {avg_score:.4f}")
    print(f"Negative Score: {ss['neg']:.4f}")
    print(f"Neutral Score: {ss['neu']:.4f}")
    print(f"Positive Score: {ss['pos']:.4f}")
    print(f"Compound Score: {ss['compound']:.4f}")

    if avg_score >= 0.05:
        comment = "It's recommended to buy this product."
    elif avg_score <= -0.05:
        comment = "It's not recommended to buy this product."
    else:
        comment = "You may consider buying this product based on other factors."

    print(f"Conclusion of Product: {comment}")


for product_name in unique_products:
    product_reviews = dfs[dfs['product_name'] == product_name]['review_text'].tolist()

    if not product_reviews:
        print(f"No reviews found for product '{product_name}'.")
    else:
        sid = SentimentIntensityAnalyzer()
        product_scores = []
        for review in product_reviews:
            ss = sid.polarity_scores(review)
            product_scores.append(ss['compound'])
        # Calculate average sentiment score for the product
        avg_score = sum(product_scores) / len(product_scores)
        product_sentiments[product_name] = avg_score



highest_product = max(product_sentiments, key=product_sentiments.get)
lowest_product = min(product_sentiments, key=product_sentiments.get)

print(f"\nProduct with highest sentiment score: {highest_product}")
print(f"Product with lowest sentiment score: {lowest_product}")

plt.figure(figsize=(10, 6))
plt.bar(product_sentiments.keys(), product_sentiments.values(), color=['green' if score >= 0 else 'red' for score in product_sentiments.values()])
plt.xlabel('Product')
plt.ylabel('Average Sentiment Score')
plt.title('Sentiment Analysis of Products')
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()



