import csv
import json
from itertools import islice
import zipfile
from lightfm.data import Dataset

def get_data():

    with zipfile.ZipFile("./data/BX-CSV-Dump.zip") as archive:
        return (
            csv.DictReader(
                (x.decode("utf-8", "ignore") for x in archive.open("BX-Book-Ratings.csv")),
                delimiter=";",
            ),
            csv.DictReader(
                (x.decode("utf-8", "ignore") for x in archive.open("BX-Books.csv")), delimiter=";"
            ),
        )


def get_ratings():

    return get_data()[0]


def get_book_features():

    return get_data()[1]

ratings, book_features = get_data()

for line in islice(ratings, 2):
    print(json.dumps(line, indent=4))

dataset = Dataset()

dataset.fit((x['User-ID'] for x in get_ratings()),
            (x['ISBN'] for x in get_ratings()))

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

dataset.fit_partial(items=(x['ISBN'] for x in get_book_features()),
                    item_features=(x['Book-Author'] for x in get_book_features()))

(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN'], float(x['Book-Rating']))
                                                      for x in get_ratings()))
print(repr(interactions))

item_features = dataset.build_item_features(((x['ISBN'], [x['Book-Author']])
                                              for x in get_book_features()))
print(repr(item_features))
from lightfm import LightFM

model = LightFM(loss='bpr')
model.fit(interactions, item_features=item_features)
# f = open("./data/BX-CSV-Dump/BX-Users.csv","rb")#二进制格式读文件
# while True:
#     line = f.readline()
#     if not line:
#         break
#     else:
#         try:
#             #print(line.decode('utf8'))
#             line.decode('utf8', "ignore")
#             #为了暴露出错误，最好此处不print
#         except:
#             print(str(line))
# with open("./data/BX-CSV-Dump/BX-Book-Ratings.csv","rb") as finput:
#     for x in finput:
#         print(x.decode("utf-8", "ignore"))

# finput = open("./data/BX-CSV-Dump/BX-Book-Ratings.csv","rb")
# a = [x.decode("utf-8", "ignore") for x in finput]
# print(a)