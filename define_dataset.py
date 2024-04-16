import pandas as pd
from datetime import datetime
from nltk.tokenize import word_tokenize
import re, csv

df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin1", header=None)
cols_to_drop = df.columns[[1, 3]]
df = df.drop(columns=cols_to_drop)
df.columns = ["target", "date", "user", "text"]

new_csv = open("project_dataset.csv", "w", newline="", encoding="utf-8")
csv_writer = csv.writer(new_csv)
csv_writer.writerow(["target", "date", "user", "text"])

for index, row in df.iterrows():
    date = datetime.strptime(row["date"], "%a %b %d %H:%M:%S PDT %Y").strftime("%d.%m.%Y %H:%M:%S")
    text = row["text"]
    text = re.sub(r"http\S+|www.\S+", "", text, flags=re.MULTILINE)
    line = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-\\]]", "", text.lower())
    tokens = word_tokenize(line)
    words = [word for word in tokens if word.isalpha()]
    csv_writer.writerow([row["target"], date, row["user"], words])

new_csv.close()
