from deep_translator import MyMemoryTranslator
from multiprocessing.dummy import Pool as ThreadPool
import time
import pandas as pd
import tqdm
import json
f = open(r'C:/Users/enesm/projects/fsvqa_augmented_train_questions.json')


data = json.load(f)
data=list(data["questions"])
data_list=[]
translator=MyMemoryTranslator(source='en', target='tr')
for i in range(len(data)//20):
    data_list.append(data[i*20:i*20+20])
def request_batch(text_dict_batch):
    text_batch=list([d["question"] for d in text_dict_batch])
    translated_texts= translator.translate_batch(text_batch) 
    for i in range(20):
        text_dict_batch[i]["questions"]=translated_texts[i]
    return text_dict_batch

for t in tqdm.tqdm(range(int(len(data_list)//100))):
    pool = ThreadPool(20) # Threads
    time1 = time.time()
    tr_data=list(data_list[t*100:t*100+100])
    
    try:
          results = pool.map(request_batch, tr_data)
    except Exception as e:
          raise e
    flat_list = [item for sublist in results for item in sublist]
    pd.DataFrame(flat_list).to_csv("tr.csv",mode="a", index=False, header=False)
    pool.close()
    pool.join()

    time2 = time.time()
    print("Translating %sth batch of sentence, a total of %s s, %s iter/s"%(t,time2 - time1,len(tr_data)/time2 - time1))
