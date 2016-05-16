import json


f = open("yelp_academic_dataset_business.json", "r")
lines = f.readlines()

for i in range(0, len(lines)):
    business_example = json.loads(lines[i])
    if 'Restaurants' in business_example['categories']:
        print business_example['name']
