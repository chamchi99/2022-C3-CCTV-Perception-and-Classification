python build_cdataset_split.py && # split train-test data 
python build_cdataset.py && # crop image using label data
python build_cdataset_query.py &&
python idx_convert.py
