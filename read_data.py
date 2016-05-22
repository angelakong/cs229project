import json


f = open("yelp_academic_dataset_business.json", "r")
lines = f.readlines()

for i in range(0, 1):
    business_example = json.loads(lines[i])
    print business_example
