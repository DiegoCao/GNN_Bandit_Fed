import json as json
import numpy as np 
import networkx as nx
from networkx.readwrite import json_graph
# with open('./example_data/toy-ppi-feats.npy') as load_f:
def t1():
    with open('./example_data/toy-ppi-G.json') as f:
        data = json.load(f)

    for i in data:
        print(i)

    print(data['directed'])
    print(data['graph'])
    print(data['multigraph'])
    # print(data['nodes'])
    print(type(data['graph']))
    print(type(data['links']))
    print(type(data['nodes']))
    cnt = 0
    itr = 0
    testcnt = 0

    train_set = set()
    val_set = set()
    test_set = set()

    for i in data['nodes']:
        mid = i['id']
        if itr == 0:
            print(i)
            itr +=1
        if i['val'] == True:
            cnt += 1
            val_set.add(mid)
        elif i['test'] ==  True:
            testcnt += 1
            test_set.add(mid)
        else:
            train_set.add(mid)

        if i['val'] == True and i['test']== True:
            print('wtf?')


    itr = 0
    for link in data['links']:
        
        if itr == 0:
            print(i)
            itr +=1    
            print(link)
            print(type(link))

        
        if link['train_removed'] == True:
            # print('wtf')
            target = link['target']
            source = link['source']
            if (target not in val_set or source not in val_set) and link['test_removed'] == False:
                print('damn!!')
            pass

        if link['test_removed'] == True:
            target = link['target']
            source = link['source']
            assert( (target in test_set) and (source in test_set))

# print(data['links'][3])
# val_cnt = cnt
# train_cnt = len(data['nodes']) - cnt - testcnt
# print('the test cnt', testcnt)
# print('the val cnt', val_cnt)
# print('the total ', len(data['nodes']))
# print('the train ', train_cnt)
# print('the train/total', train_cnt/len(data['nodes']))

# print(cnt)
# print(len(data['nodes'])- cnt)



# res = np.load('./unsup_example_data/graphsage_mean_small_0.000010/val.npy')
# # print(res[0])


# print(len(res))

# feats = np.load('./example_data/toy-ppi-feats.npy')
# print(type(feats))
# print(type(feats[0]))
# print(feats[0])

def t2():
    with open('./fljson/sto-G.json', 'r') as fp:
        file = json.load(fp)
    itr = 0


    # for key, items in file.items():
    #     if itr == 0:

    #         itr +=1
    #     print(key)


    G = json_graph.node_link_graph(file)
    print(G.nodes[0]['val'])
    for edge in G.edges():
        print(edge)
if __name__ == "__main__":

    t2()