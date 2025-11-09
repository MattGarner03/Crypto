import feedparser
# Use a pipeline as a high-level helper
from transformers import pipeline

ticker = "BTC-USD"
keyword = 'BTC'
keywords = ["BTC", "Bitcoin"]

pipe = pipeline("text-classification", model="ProsusAI/finbert")

rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'

feed = feedparser.parse(rss_url)

total_score = 0

num_articles = 0

for i, entry in enumerate (feed.entries):



    # skip if NONE of the keywords appear
#    if not any(k in text for k in keywords):
    found = False
    for k in keywords:
        if k.lower() in entry.summary.lower():
            found = True
            break  # stop checking once we find one

    if not found:
        continue

#    if not any(k.lower() in entry.summary.lower() for k in keywords):
#        continue

#    if keyword.lower() not in entry.summary.lower():
#        continue

    print(f'Title: {entry.title}')
    print(f'Link: {entry.link}')
    print(f'Published: {entry.published}')
    print(f'Summary: {entry.summary}')

    sentiment = pipe(entry.summary)[0]

    print(f'Sentiment {sentiment["label"]}, Score: {sentiment["score"]}')
    print("-"*40)

    if sentiment['label'] == 'positive':
        total_score += sentiment['score']
        num_articles +=1
    elif sentiment['label'] == 'negative':
        total_score -= sentiment['score']
        num_articles +=1

final_score = total_score / num_articles
print(f'Overall Sentiment: {"Positive" if total_score >= 0.15 else "Negative" if total_score <= -0.15 else 'Neutral'} {final_score}')