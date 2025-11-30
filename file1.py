'''
# imported the Yelp Business dataset with all of the information of each business
import json
with open("yelp_academic_dataset_review.json", "r") as f:
  data = json.load(f)

for line in data:
  print(json.dumps(line, indent=2))

'''
'''
import json
import re

df = []
with open("yelp_academic_dataset_review.json", "r") as f:
    for line in f:
        df.append(json.loads(line))
#print(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned"] = df["text"].apply(clean_text)
#df
'''

import json
import re
import pandas as pd

# Loading data
data = []
with open("yelp_academic_dataset_review.json", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Selecting the columns we want (confirm are these the only ones we need?)
df = pd.DataFrame(data)[['business_id', 'stars', 'text']]

# Clean Function (as in our google doc)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Group reviews by business_id (this code shows first 5 groups as example)
business_groups = df.groupby('business_id')
print("Number of businesses:", len(business_groups))
print("\nFirst 5 business groups:")
for business_id, group in list(business_groups)[:5]:
    print(f"\nBusiness ID: {business_id}")
    print(f"Number of reviews: {len(group)}")
    print(f"Average stars: {group['stars'].mean():.2f}")
    print("Sample cleaned texts:")
    for i, text in enumerate(group['cleaned_text'].head(2)):
        print(f"  {i+1}: {text[:100]}...")

# Optional: Save cleaned data to CSV (can think about this later)
#df.to_csv('yelp_cleaned_reviews.csv', index=False)


