import numpy as np
from gamma.rules import rules_gamma_operator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
class Gamma(object):

    @staticmethod
    def get_number_of_patterns_by_class(y):
        class1_number = np.count_nonzero(y == 1)
        class2_number = np.count_nonzero(y == 2)
        class3_number = np.count_nonzero(y == 3)
        total = []
        total.extend([class1_number, class2_number, class3_number])
        print("Total elements of Class 1:", class1_number)
        print("Total elements of Class 2:", class2_number)
        print("Total elements of Class 3:", class3_number)
        print("Total: ", sum(total))
        # set figure size
        plt.figure(figsize=(14, 6))

        # add title
        plt.title("Countplot of #Patterns by Class")

        # make countplot
        sns.countplot(x=y)
        plt.show()

        return class1_number, class2_number, class3_number

    @staticmethod
    def get_patterns_by_class(y, X):
        y_y = y.copy()
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            y = np.array(y)
            X = np.array(X)

        indexes_class1 = np.array(())
        indexes_class2 = np.array(())
        indexes_class3 = np.array(())

        # Obtenemos los indices por clase
        for w in range(0, y.shape[0]):  # 10240  [1, 0, 1, 0 .., 10240 VERTICAL!!]

            if y[w] == 1:
                indexes_class1 = np.append(indexes_class1, w)

            if y[w] == 2:
                indexes_class2 = np.append(indexes_class2, w)

            if y[w] == 3:
                indexes_class3 = np.append(indexes_class3, w)

        X_elements_class_1 = np.zeros((len(indexes_class1), X.shape[1]))  # [1,1,1,,1,1,1..., 768 Horizontal)
        X_elements_class_2 = np.zeros((len(indexes_class2), X.shape[1]))
        X_elements_class_3 = np.zeros((len(indexes_class3), X.shape[1]))

        # Adquirimos los valores de esos indices
        for z in range(0, y.shape[0]):
            if z < len(indexes_class1):
                X_elements_class_1[z] = X[int(indexes_class1[z])]

            if z < len(indexes_class2):
                X_elements_class_2[z] = X[int(indexes_class2[z])]

            if z < len(indexes_class3):
                X_elements_class_3[z] = X[int(indexes_class3[z])]

        return X_elements_class_1, X_elements_class_2, X_elements_class_3

    @staticmethod
    def detect_missing_values(X, y):
        print('Missing X data? ', X.isnull().any().any())
        print('Missing y Label? ', y.isnull().any())
        msg='Done'
        return msg

    @staticmethod
    def separate_data_and_labels(X):
        X.dropna(axis=0, subset=['class'], inplace=True)
        y = X['class']
        X.drop(['class'], axis=1, inplace=True)
        return X, y

    @staticmethod
    def logarithmic_scale(X, exponent):
        X = X*(10**(exponent+1))
        X = X.astype(int)
        return X

    @staticmethod
    def total_patterns(X):
        print("Total patterns: ", X.shape[0])
        msg = 'Done'
        return msg

    @staticmethod
    def read_dataset(website, dataset):
        if website == 'UCI_Machine_Learning_Repository':
            if dataset == 'iris':
                names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
                X = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', sep=',',
                                names=names)

        return X

    @staticmethod
    def hold_out(X, y, train_size, test_size):
        X_train, X_valid, y_train, y_valid = train_test_split(X.values, y.values, train_size=train_size, test_size=test_size,
                                                              random_state=42)
        return X_train, X_valid, y_train, y_valid

    @staticmethod
    def hold_out_stratified(X, y, train_size, test_size):
        sss = StratifiedShuffleSplit(train_size=train_size, test_size=test_size, random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
        return X_train, X_valid, y_train, y_valid


    @staticmethod
    def feature_correlation(X):
        df_train = X.copy()
        train_cols = df_train.columns.tolist()
        train_cols = train_cols[-1:] + train_cols[:-1]
        df_train = df_train[train_cols]
        corr = df_train.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(10, 10))

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()
        msg='Done'
        return msg
    @staticmethod
    def prepare_data(class1_n, patterns_c1, class2_n, patterns_c2, class3_n, patterns_c3, X_valid, y_valid):

        if isinstance(X_valid, pd.DataFrame) and isinstance(y_valid, pd.Series):
            y_valid = np.array(y_valid)
            X_valid = np.array(X_valid)

        train_patterns = {'patterns_c1': patterns_c1, 'patterns_c2': patterns_c2, 'patterns_c3': patterns_c3,}  # the dict you want to pass to func
        X_test = {'X_valid': X_valid, 'y_valid': y_valid}
        number_of_patterns = {'class1_n': class1_n, 'class2_n': class2_n, 'class3_n': class3_n}

        return train_patterns, X_test, number_of_patterns

    @staticmethod
    def predict(**data):
        X_y = data.get('X_test')
        number_of_patterns = data.get('number_of_patterns')
        train_patterns = data.get('train_patterns')
        paro = data.get('theta')
        test_X = X_y['X_valid']
        test_y = X_y['y_valid']
        X_elements_class_1 = train_patterns['patterns_c1']
        X_elements_class_2 = train_patterns['patterns_c2']
        X_elements_class_3 = train_patterns['patterns_c3']
        clase1_number = number_of_patterns['class1_n']

        predicted = []
        tetha = 0
        for i in range(0, test_y.shape[0]):  # 27
            while tetha < paro:
                clase1sum = 0
                clase2sum = 0
                clase3sum = 0
                for j in range(0, clase1_number):  # 123
                    for k in range(0, len(test_X[0])):
                        clase1sum = clase1sum + rules_gamma_operator(X_elements_class_1[j][k], test_X[i][k], tetha)
                        clase2sum = clase2sum + rules_gamma_operator(X_elements_class_2[j][k], test_X[i][k], tetha)
                        clase3sum = clase3sum + rules_gamma_operator(X_elements_class_3[j][k], test_X[i][k], tetha)
                clase1 = clase1sum / 2
                clase2 = clase2sum / 2
                clase3 = clase3sum / 2
                iguales = (
                        clase1 == clase2 and clase1 == clase3 and clase2 == clase3)
                iguales = int(iguales)
                if iguales:
                    tetha = tetha + 1
                    if tetha == paro - 1:
                        maxi = np.zeros(3)
                        maxi[0] = clase1
                        maxi[1] = clase2
                        maxi[2] = clase3
                        maximo = maxi.argmax() + 1
                        class_predicted = maximo
                        predicted.append(class_predicted)
                diferentes = (clase1 != clase2 or clase1 != clase3 or clase2 != clase3)
                diferentes = int(diferentes)
                if diferentes == 1 and iguales == 0:
                    maxi = np.zeros(3)
                    maxi[0] = clase1
                    maxi[1] = clase2
                    maxi[2] = clase3
                    maximo = maxi.argmax() + 1
                    class_predicted = maximo
                    predicted.append(class_predicted)
                    break
        acc = np.count_nonzero(np.array(predicted) == test_y)
        acc = (acc * 100) / len(test_y)
        return acc, np.array(predicted)

    @staticmethod
    def plot_confusion_matrix(targets, predictions, target_names,
                              title='Confusion matrix', cmap="Blues"):

        targets = np.array(targets)
        predictions = np.array(predictions)
        """Plot Confusion Matrix."""
        cm = confusion_matrix(targets, predictions)
        cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
        g = sns.heatmap(df, annot=True, fmt=".1f", linewidths=.5, vmin=0, vmax=100,
                        cmap=cmap)
        g.set_title(title)
        g.set_ylabel('True label')
        g.set_xlabel('Predicted label')
        plt.show()
        return g
    @staticmethod
    def mean_absolute_error(targets, y):
        mae = mean_absolute_error(targets, y)
        print("GMean Asbolute Error: ", mae)
        msg = 'Done'
        return msg

    @staticmethod
    def generate_descriptive_statistics(X):

        print(X.describe())
        msg = 'Done'
        return msg

