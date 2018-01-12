import numpy as np
import pandas as pd
import math
df_train = pd.read_csv('../data/dogs-breed/labels.csv')
breedTypes = df_train['breed'].drop_duplicates()
breedCounts = pd.Series(data=0, index=breedTypes, dtype=np.int16)

validation_percentage = 0.05
total_number = df_train.shape[0]
retry = 5
tried = 0
while tried < retry:
    sample = np.random.permutation(total_number)[1:math.floor(validation_percentage*total_number)]

    train_flag = np.repeat(True, total_number)
    # train_flag[sample] = False
    np.put(train_flag, sample, False)

    df_train['train'] = train_flag

    for index in sample:
        breedCounts[df_train.ix[index]['breed']] = breedCounts[df_train.ix[index]['breed']] + 1

    if breedCounts.min() != 0:
        print('Good (min is {}).'.format(breedCounts.min()))
        df_train.to_csv('labels_enhanced.csv', index=False)
        break
    else:
        print('Some breed is out of selection. We will try again.')

    tried = tried + 1
else:
    print('Oh we have failed.')


