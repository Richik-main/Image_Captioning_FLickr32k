from datasets import load_dataset, load_from_disk
#%%
# ds = load_dataset("nlphuji/flickr30k")
# ds.save_to_disk("flickr30k_dataset_backup")
# print("Dataset saved to disk in Arrow format")

#%%
data = load_from_disk("/home/ubuntu/Training_img_cap/Caption_module/flickr30k_dataset_backup")
#%%
data = data['test']
print(data)
# train_data = data['test'].filter(lambda x: x['split'] == 'train')
# test_data = data['test'].filter(lambda x: x['split'] =='test')
# val_data= data['test'].filter(lambda x: x['split'] == 'val')
