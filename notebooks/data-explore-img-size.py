# /!\ VS code : Télécharger l'extension jupyter
# /!\ replacer dossier images dans '\bootcamp-mle24-rakuten\data'

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Infos + visu rapide X_train
X_train = pd.read_csv('../data/X_train_update.csv', index_col = 0)
X_train.head()
X_train.info()

# %%
# Infos + visu rapide y_train
y_train = pd.read_csv('../data/Y_train_CVw08PX.csv', index_col = 0)
y_train.head()
y_train.info()
prdtypecodes = set(y_train['prdtypecode'].tolist())
print(len(prdtypecodes))

# %%
# Analyse taille des images
import os

# liste des path images
train_images_path = '../data/images/image_train'
test_images_path = '../data/images/image_test'
images_list = os.listdir(train_images_path) + os.listdir(test_images_path)
print(f"Nombre total d'images: {len(images_list)}")

# Recuperation size des images
img_size_list = []
img_name_list = []

# recuperation size img train
with os.scandir(train_images_path) as img_path:
    for elem in img_path:
        info = elem.stat()
        img_size_list.append(info.st_size)
        img_name_list.append(elem.name)

# recuperation size img test
with os.scandir(test_images_path) as img_path:
    for elem in img_path:
        info = elem.stat()
        img_size_list.append(info.st_size)
        img_name_list.append(elem.name)

# Dataframe nom / size img
img_size_df = pd.DataFrame(img_size_list, columns = ['image size in bits'])
img_name_df = pd.DataFrame(img_name_list,columns = ['image name'])
img_df = pd.concat([img_size_df, img_name_df], axis = 1)
img_df.head()
img_size_df.describe()

# %%
# Analyse distribution size / fréquence des img
import matplotlib.pyplot as plt

img_size_df.plot.hist(y = ['image size in bits'], bins = 40, rwidth = 0.8 , alpha = 0.5, legend = False)
plt.xlabel('Taille (bits)')
plt.ylabel('Fréquence')
plt.title('Fréquence des images par taille en bits')
plt.show()

#%%
# Ajout d'une colonne qui contient le nom complet de l'image
X_train['image name'] = 'image_' + X_train['imageid'].map(str) + '_product_' + X_train['productid'].map(str) + '.jpg'
X_train.head()

#%%
# Ajout d'une colonne qui contient la catégorie finale
X_train = X_train.merge(img_df, 'left', 'image name')
X_train['y_train'] = y_train
X_train.head()

# %%
# Vérification taille img / type de produit
plt.figure(figsize = (20, 8))
sns.boxplot(x = 'y_train', y = 'image size in bits', data = X_train)
plt.xlabel('prdtypecode')
plt.ylabel('img size (bits)')
plt.title("Distribution de la taille des image par prdtypecode")
plt.show()

# %%
# test ANOVA influence prdtypecode sur size img
# H0 : pas d'influence, moyennes egales
# H1 : influence
import statsmodels.api

X_train = X_train.rename(columns = {'image size in bits': 'image_size_bits'})
df = pd.concat([X_train , y_train], axis = 1)
result = statsmodels.formula.api.ols('image_size_bits ~ prdtypecode', data = df).fit()
table = statsmodels.api.stats.anova_lm(result)
table
# %%
