import numpy as np

with open("/Users/Patron/Desktop/Advanced-NLP/hw1/glove.6B.50d.txt", "r") as f:
        embeddings=f.read()

    
emb_list=embeddings.strip().split("\n")
print(emb_list[0].split(" "))

emb_dict={}

for i in emb_list:
    emb_dict[i.split()[0]]=np.array(i.split()[1:], dtype=np.float32)


print(emb_dict['the'])