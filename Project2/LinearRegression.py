import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import ANN
import torch


class LR:
    def predict(self, X_test, w):
        y = np.dot(X_test, w)
        return y

    def mean_squared_error(self, X, w, y):
        return np.mean((np.dot(X, w) - y) ** 2)

    def train(self, train):
        x_train, y_train = ann.get_train(train)
        x = x_train
        y = y_train
        X = np.column_stack((np.ones(x.shape[0]), x))
        w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        train_mse = self.mean_squared_error(X, w, y_train)
        return w, train_mse

    def evaluate(self, test, w):
        x_test, y_test = ann.get_train(test)
        X_test = np.column_stack((np.ones(x_test.shape[0]), x_test))
        return self.mean_squared_error(X_test, w, y_test)

    def train_evaluation(self, train, test):
        w, train_mse = self.train(train)
        return self.evaluate(test, w)

    def train_evaluate(self, train_x, train_y, test_x, test_y):
        x = train_x
        y = train_y
        X = np.column_stack((np.ones(x.shape[0]), x))
        w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        train_mse = self.mean_squared_error(X, w, train_y)

        X_test = np.column_stack((np.ones(test_x.shape[0]), test_x))
        return self.mean_squared_error(X_test, w, test_y)

    def TGTONLY(self, file_paths, target_domain, target_size):
        female_train = pd.read_csv(file_paths["female_train"]).sample(target_size)
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"]).sample(target_size)
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"]).sample(target_size)
        mixed_test = pd.read_csv(file_paths["mixed_test"])

        if target_domain == "female":
            mse = self.train_evaluation(female_train, female_test)
        elif target_domain == "male":
            mse = self.train_evaluation(male_train, male_test)
        else:
            mse = self.train_evaluation(mixed_train, mixed_test)

        return mse

    def SRCONLY(self, file_paths, target_domain, target_size):
        female_train = pd.read_csv(file_paths["female_train"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])

        if target_domain == "female":
            train_female = male_train.copy()
            train_female = train_female.append(mixed_train, ignore_index=True)
            train_female = train_female.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_female, female_test)

        elif target_domain == "male":
            train_male = female_train.copy()
            train_male = train_male.append(mixed_train, ignore_index=True)
            train_male = train_male.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_male, male_test)

        else:
            train_mixed = male_train.copy()
            train_mixed = train_mixed.append(female_train, ignore_index=True)
            train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_mixed, mixed_test)

        return mse

    def ALL(self, file_paths, target_domain, target_size):
        female_train = pd.read_csv(file_paths["female_train"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])

        if target_domain == "female":
            train_female = male_train.copy()
            train_female = train_female.append(mixed_train, ignore_index=True)
            train_female = train_female.append(female_train.sample(target_size), ignore_index=True)
            train_female = train_female.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_female, female_test)

        elif target_domain == "male":
            train_male = female_train.copy()
            train_male = train_male.append(mixed_train, ignore_index=True)
            train_male = train_male.append(male_train.sample(target_size), ignore_index=True)
            train_male = train_male.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_male, male_test)

        else:
            train_mixed = male_train.copy()
            train_mixed = train_mixed.append(female_train, ignore_index=True)
            train_mixed = train_mixed.append(mixed_train.sample(target_size), ignore_index=True)
            train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_mixed, mixed_test)

        return mse

    def WEIGHTED(self, file_paths, target_domain, target_size):
        female_train = pd.read_csv(file_paths["female_train"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])

        if target_domain == "female":
            train_female = male_train.copy()
            train_female = train_female.append(mixed_train, ignore_index=True)
            ratio = train_female.shape[0] / target_size
            female_train_split = female_train.sample(target_size)
            temp = female_train_split
            for i in range(int(ratio)):
                female_train_split = female_train_split.append(temp, ignore_index=True)
            train_female = train_female.append(female_train_split, ignore_index=True)
            train_female = train_female.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_female, female_test)

        elif target_domain == "male":
            train_male = female_train.copy()
            train_male = train_male.append(mixed_train, ignore_index=True)
            ratio = train_male.shape[0] / target_size
            male_train_split = male_train.sample(target_size)
            temp = male_train_split
            for i in range(int(ratio)):
                male_train_split = male_train_split.append(temp, ignore_index=True)
            train_male = train_male.append(male_train_split, ignore_index=True)
            train_male = train_male.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_male, male_test)

        else:
            train_mixed = male_train.copy()
            train_mixed = train_mixed.append(female_train, ignore_index=True)
            ratio = train_mixed.shape[0] / target_size
            mixed_train_split = mixed_train.sample(target_size)
            temp = mixed_train_split
            for i in range(int(ratio)):
                mixed_train_split = mixed_train_split.append(temp, ignore_index=True)
            train_mixed = train_mixed.append(mixed_train_split, ignore_index=True)
            train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
            mse = self.train_evaluation(train_mixed, mixed_test)

        return mse

    def add_feature(self, data, new_feature):
        result = data
        result["SRCONLY"] = new_feature
        return result

    def PRED_train(self, src, tgt, test):
        w, train_mse = self.train(src)

        train_x, train_y = ann.get_train(tgt)
        train_x = np.column_stack((np.ones(train_x.shape[0]), train_x))
        new_train_feature = self.predict(train_x, w)
        new_train_x = self.add_feature(tgt, new_train_feature)

        test_x, test_y = ann.get_train(test)
        test_x = np.column_stack((np.ones(test_x.shape[0]), test_x))
        new_test_feature = self.predict(test_x, w)
        new_test_x = self.add_feature(test, new_test_feature)
        return self.train_evaluate(new_train_x, train_y, new_test_x, test_y)

    def PRED(self, file_paths, target_domain, target_size):
        female_train = pd.read_csv(file_paths["female_train"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])

        if target_domain == "female":
            train_female = male_train.copy()
            train_female = train_female.append(mixed_train, ignore_index=True)
            source_domain = train_female.sample(frac=1).reset_index(drop=True)
            target_domain = female_train.sample(target_size).reset_index(drop=True)
            test = female_test
            mse = self.PRED_train(source_domain, target_domain, test)

        elif target_domain == "male":
            train_male = female_train.copy()
            train_male = train_male.append(mixed_train, ignore_index=True)
            source_domain = train_male.sample(frac=1).reset_index(drop=True)
            target_domain = male_train.sample(target_size).reset_index(drop=True)
            test = male_test
            mse = self.PRED_train(source_domain, target_domain, test)

        else:
            train_mixed = male_train.copy()
            train_mixed = train_mixed.append(female_train, ignore_index=True)
            source_domain = train_mixed.sample(frac=1).reset_index(drop=True)
            target_domain = mixed_train.sample(target_size).reset_index(drop=True)
            test = mixed_test
            mse = self.PRED_train(source_domain, target_domain, test)

        return mse

    def LININT_train(self, src_domain, tat_domain, dev, test):
        src_w, src_mse = self.train(src_domain)
        dev_x, dev_y = ann.get_train(dev)
        dev_x = np.column_stack((np.ones(dev_x.shape[0]), dev_x))
        src_feature = self.predict(dev_x, src_w).flatten().tolist()

        tgt_w, tgt_mse = self.train(tat_domain)
        tgt_feature = self.predict(dev_x, tgt_w).flatten().tolist()
        index = [i for i in range(100)]

        data = {'SRCONLY': src_feature, 'TGTONLY': tgt_feature}
        new_dev_x = pd.DataFrame(data=data, index=index)

        test_x, test_y = ann.get_train(test)
        test_x = np.column_stack((np.ones(test_x.shape[0]), test_x))
        src_feature = self.predict(test_x, src_w).flatten().tolist()
        tgt_feature = self.predict(test_x, tgt_w).flatten().tolist()
        data = {'SRCONLY': src_feature, 'TGTONLY': tgt_feature}
        new_test_x = pd.DataFrame(data=data, index=index)

        mse = self.train_evaluate(new_dev_x, dev_y, new_test_x, test_y)
        return mse

    def LININT(self, file_paths, target_domain, target_size):
        female_train = pd.read_csv(file_paths["female_train"])
        female_dev = pd.read_csv(file_paths["female_dev"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_dev = pd.read_csv(file_paths["male_dev"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_dev = pd.read_csv(file_paths["mixed_dev"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])

        if target_domain == "mixed":
            train_female = male_train.copy()
            train_female = train_female.append(mixed_train, ignore_index=True)
            src_domain = train_female.sample(frac=1).reset_index(drop=True)
            tat_domian = female_train.sample(target_size)
            mse = self.LININT_train(src_domain, tat_domian, female_dev, female_test)

        elif target_domain == "male":
            train_male = female_train.copy()
            train_male = train_male.append(mixed_train, ignore_index=True)
            src_domain = train_male.sample(frac=1).reset_index(drop=True)
            tat_domian = male_train.sample(target_size)
            mse = self.LININT_train(src_domain, tat_domian, male_dev, male_test)

        else:
            train_mixed = female_train.copy()
            train_mixed = train_mixed.append(male_train, ignore_index=True)
            src_domain = train_mixed.sample(frac=1).reset_index(drop=True)
            tat_domian = mixed_train.sample(target_size)
            mse = self.LININT_train(src_domain, tat_domian, mixed_dev, mixed_test)
        return mse


if __name__ == '__main__':
    ann = ANN.ANN()
    lr = LR()
    female_path = "Data/FEMALE.csv"
    female_train_path = "Data/FEMALE_train.csv"
    female_dev_path = "Data/FEMALE_dev.csv"
    female_test_path = "Data/FEMALE_test.csv"
    male_path = "Data/MALE.csv"
    male_train_path = "Data/MALE_train.csv"
    male_dev_path = "Data/MALE_dev.csv"
    male_test_path = "Data/MALE_test.csv"
    mixed_path = "Data/MIXED.csv"
    mixed_train_path = "Data/MIXED_train.csv"
    mixed_dev_path = "Data/MIXED_dev.csv"
    mixed_test_path = "Data/MIXED_test.csv"

    file_paths = {}
    file_paths["female_train"] = female_train_path
    file_paths["female_test"] = female_test_path
    file_paths["female_dev"] = female_dev_path
    file_paths["male_train"] = male_train_path
    file_paths["male_test"] = male_test_path
    file_paths["male_dev"] = male_dev_path
    file_paths["mixed_train"] = mixed_train_path
    file_paths["mixed_test"] = mixed_test_path
    file_paths["mixed_dev"] = mixed_dev_path

    feda_file_paths = {}
    feda_file_paths["female_train"] = "FEDA DATA/FEMALE_train.csv"
    feda_file_paths["female_test"] = "FEDA DATA/FEMALE_test.csv"
    feda_file_paths["female_dev"] = "FEDA DATA/FEMALE_dev.csv"
    feda_file_paths["male_train"] = "FEDA DATA/MALE_train.csv"
    feda_file_paths["male_test"] = "FEDA DATA/MALE_test.csv"
    feda_file_paths["male_dev"] = "FEDA DATA/MALE_dev.csv"
    feda_file_paths["mixed_train"] = "FEDA DATA/MIXED_train.csv"
    feda_file_paths["mixed_test"] = "FEDA DATA/MIXED_test.csv"
    feda_file_paths["mixed_dev"] = "FEDA DATA/MIXED_dev.csv"

    mse = {}
    target_domain = "mixed"
    target_size = 100
    mse["Domain"] = target_domain + " " + str(target_size)
    mse["SRCONLY"] = lr.SRCONLY(file_paths, target_domain, target_size)
    mse["TGTONLY"] = lr.TGTONLY(file_paths, target_domain, target_size)
    mse["ALL"] = lr.ALL(file_paths, target_domain, target_size)
    mse["WEIGHTED"] = lr.WEIGHTED(file_paths, target_domain, target_size)
    mse["PRED"] = lr.PRED(file_paths, target_domain, target_size)
    mse["LINTINT"] = lr.LININT(file_paths, target_domain, target_size)
    mse["FEDA"] = lr.ALL(feda_file_paths, target_domain, target_size)
    # for i in range(sample_times):
    #     mse["SRCONLY"] += lr.SRCONLY(file_paths)
    #     mse["TGTONLY"] += lr.TGTONLY(file_paths)
    #     mse["ALL"] += lr.ALL(file_paths)
    #     mse["WEIGHTED"] += lr.WEIGHTED(file_paths)
    #     mse["PRED"] += lr.PRED(file_paths)
    #     mse["LINTINT"] += lr.LININT(file_paths)
    # for key, values in mse.items():
    #     mse[key] = values/sample_times
    print(mse)
