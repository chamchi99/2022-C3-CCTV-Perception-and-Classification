import pickle

# hard coded path
with open('reid-data/cdataset/train_ids.txt', 'r') as f:
    train_ids = f.readline().split(',')[:-1]
    train_ids = sorted(list(map(int, train_ids)))
    
# hard coded path
with open('reid-data/cdataset/test_ids.txt', 'r') as f:
    test_ids = f.readline().split(',')[:-1]
    test_ids = sorted(list(map(int, test_ids)))


train_idx_con = {}
for i, train_id in enumerate(train_ids):
    train_idx_con[train_id] = i # ID should start with zero (becaue of torchreid library setting)

test_idx_con = {}
for i, test_id in enumerate(test_ids):
    test_idx_con[test_id] = i # ID should start with zero (becaue of torchreid library setting)


with open('reid-data/cdataset/train_idx_con.pickle', 'wb') as f:
    pickle.dump(train_idx_con, f)

with open('reid-data/cdataset/test_idx_con.pickle', 'wb') as f:
    pickle.dump(test_idx_con, f)
