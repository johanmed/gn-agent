import json
import os
import string
from copy import copy

dir = "/home/johannesm/all_corpus/"

collection = {}

files = [os.path.join(dir, file) for file in os.listdir(dir)]


def clean(text) -> str:
    return text.strip(string.punctuation).strip()


for file in files:
    with open(file) as f:
        lines = f.readlines()
        mem = ""
        for ind, line in enumerate(lines):
            if (not line.startswith("@")) and len(line) > 1:
                if line.startswith("<"):
                    line = line.strip()
                    contents = line.split(" ")
                    key = (
                        clean(contents[0].split("/")[-1])
                        if "/" in contents[0]
                        else clean(contents[0].split(":")[-1])
                    )
                    mem = copy(key)
                    value = " ".join(
                        (
                            clean(content.split("/")[-1])
                            if "/" in content
                            else clean(content.split(":")[-1])
                        )
                        for content in contents[1:]
                    )
                    key = clean(key)
                    value = clean(value)
                    if key not in collection:
                        collection[key] = [value]
                    else:
                        collection[key].append(value)
                else:
                    line = line.strip()
                    key = copy(mem)
                    contents = line.split(" ")
                    value = " ".join(
                        (
                            clean(content.split("/")[-1])
                            if "/" in content
                            else clean(content.split(":")[-1])
                        )
                        for content in contents
                    )
                    key = clean(key)
                    value = clean(value)
                    if key not in collection:
                        collection[key] = [value]
                    else:
                        collection[key].append(value)

with open(f"{dir}aggr_rdf.txt", "w") as new:
    new.write(json.dumps(collection))
