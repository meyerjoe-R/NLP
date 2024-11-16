# example for video games dataset

with open(file, 'r') as fp:
    for line in fp:
        pprint(json.loads(line.strip()))
        break