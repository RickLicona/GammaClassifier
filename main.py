import numpy as np
import pandas as pd
from gamma.utils import Gamma


training = pd.read_csv('/Users/ricklicona/PycharmProjects/GammaClassifier/data/Train.csv')
test = pd.read_csv('/Users/ricklicona/PycharmProjects/GammaClassifier/data/Test.csv')
df_training = training
df_test = test

train_X = df_training.drop(['Class'], 1)
train_y = df_training['Class']

test_X = df_test.drop(['Class'], 1)
test_y = df_test['Class']

class1_n, class2_n, class3_n = Gamma().get_number_of_patterns_by_class(train_y)

train_elements_class_1, train_elements_class_2, train_elements_class_3 = Gamma().get_patterns_by_class(train_y,
                                                                                                       train_X)
train_patterns, X_test, number_of_patterns = Gamma().prepare_data(class1_n,
                                                                  train_elements_class_1,
                                                                  class2_n,
                                                                  train_elements_class_2,
                                                                  class3_n,
                                                                  train_elements_class_3,
                                                                  test_X,
                                                                  test_y)

data = {'X_test': X_test, 'train_patterns': train_patterns, 'number_of_patterns': number_of_patterns, 'theta': 5}    #the keyword argument container

acc, preds = Gamma().predict(**data)

print("\nGAMMA ACC: ", acc)#92.5925
names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

