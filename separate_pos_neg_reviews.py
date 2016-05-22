import nltk
import json

pos = []
neg = []
f = open("yelp_academic_dataset_business.json", "r")
lines = f.readlines()

for i in range(0, len(lines)):
    business_example = json.loads(lines[i])
    if business_example['stars'] <= 3 and 'Restaurants' in business_example['categories'] and business_example['city'] == 'Las Vegas':
        neg.append(business_example['business_id'])
    elif business_example['stars'] > 3 and 'Restaurants' in business_example['categories'] and business_example['city'] == 'Las Vegas':
        pos.append(business_example['business_id'])

f = open("yelp_academic_dataset_review.json", "r")
lines = f.readlines()

for i in range(0, len(lines)):
    business_example = json.loads(lines[i])
    if business_example['business_id'] in pos:
        with open('pos_reviews', 'a') as pos_reviews:
            try:
                pos_reviews.write(business_example['text'])
            except:
                pass
    elif business_example['business_id'] in neg:
        with open('neg_reviews', 'a') as neg_reviews:
            try:
                neg_reviews.write(business_example['text'])
            except:
                pass
