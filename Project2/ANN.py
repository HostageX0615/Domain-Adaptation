import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import random
import json


class ANN(nn.Module):

    def train_dev_test_split(self, input_path, train_path, dev_path, test_path):
        df = pd.read_csv(input_path)
        total = df.sample(200)
        dev = total[100:]
        test = total[:100]
        df = df.drop(dev.index)
        train = df.drop(test.index)
        train.to_csv(train_path, index=0)
        dev.to_csv(dev_path, index=0)
        test.to_csv(test_path, index=0)

    def label_formatting(self, train_path):
        df = pd.read_csv(train_path)
        label = df.pop("Exam Score")
        dataset = tf.data.Dataset.from_tensor_slices((df.values, label.values))
        train_dataset = dataset.shuffle(len(df)).batch(1)
        return train_dataset

    def get_compiled_model(self, input_size, units1, units2):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            units=units1,
            activation='relu',
            input_shape=(input_size,)
        ))

        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(
            units=units2,
            activation='relu'
        ))

        model.add(tf.keras.layers.Dense(
            units=1,
            activation='linear'
        ))
        # print(model.summary())

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='mse')
        return model

    def simple_model(self, input_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            units=1,
            activation='linear',
            input_shape=(input_size,)
        ))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='mse')
        return model

    def get_train(self, df):
        y = df.pop("Exam Score")
        x = df
        y = torch.tensor(y.values)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    def add_feature(self, data, new_feature):
        result = data
        result["SRCONLY"] = new_feature

        return result

    def model_train(self, model, train, patience, delta):
        train_x, train_y = self.get_train(train)
        min_delta = pow(10, delta)
        call_back = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=patience, min_delta=min_delta)
        model.fit(train_x, train_y,
                  epochs=100000,
                  batch_size=128,
                  callbacks=[call_back])
        return model

    def model_train_evaluation(self, model, train, test, patience, delta):
        self.model_train(model, train, patience, delta)
        test_x, test_y = self.get_train(test)
        return model.evaluate(test_x, test_y)

    def model_train_evaluate(self, model, train_x, train_y, test_x, test_y):
        train_y = np.asarray(train_y)
        call_back = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20, min_delta=0.01)
        model.fit(train_x, train_y, epochs=10000, batch_size=128, callbacks=[call_back])
        return model.evaluate(test_x, test_y)

    def SRCONLY(self, model, file_paths, target_domain, target_size, patience, delta, dev=False):
        female_train = pd.read_csv(file_paths["female_train"])
        female_dev = pd.read_csv(file_paths["female_dev"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_dev = pd.read_csv(file_paths["male_dev"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_dev = pd.read_csv(file_paths["mixed_dev"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])
        if dev == False:
            if target_domain == "female":
                train_female = male_train.copy()
                train_female = train_female.append(mixed_train, ignore_index=True)
                train_female = train_female.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_female, female_test, patience, delta)

            elif target_domain == "male":
                train_male = female_train.copy()
                train_male = train_male.append(mixed_train, ignore_index=True)
                train_male = train_male.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_male, male_test, patience, delta)

            else:
                train_mixed = male_train.copy()
                train_mixed = train_mixed.append(female_train, ignore_index=True)
                train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_mixed, mixed_test, patience, delta)
        else:
            if target_domain == "female":
                train_female = male_train.copy()
                train_female = train_female.append(mixed_train, ignore_index=True)
                train_female = train_female.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_female, female_dev, patience, delta)

            elif target_domain == "male":
                train_male = female_train.copy()
                train_male = train_male.append(mixed_train, ignore_index=True)
                train_male = train_male.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_male, male_dev, patience, delta)

            else:
                train_mixed = male_train.copy()
                train_mixed = train_mixed.append(female_train, ignore_index=True)
                train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_mixed, mixed_dev, patience, delta)
        return mse

    def TGTONLY(self, model, file_paths, target_domain, target_size, patience, delta, dev=False):
        female_train = pd.read_csv(file_paths["female_train"]).sample(target_size)
        female_dev = pd.read_csv(file_paths["female_dev"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"]).sample(target_size)
        male_dev = pd.read_csv(file_paths["male_dev"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"]).sample(target_size)
        mixed_dev = pd.read_csv(file_paths["mixed_dev"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])
        if dev == False:
            if target_domain == "female":
                mse = self.model_train_evaluation(model, female_train, female_test, patience, delta)
            elif target_domain == "male":
                mse = self.model_train_evaluation(model, male_train, male_test, patience, delta)
            else:
                mse = self.model_train_evaluation(model, mixed_train, mixed_test, patience, delta)
        else:
            if target_domain == "female":
                mse = self.model_train_evaluation(model, female_train, female_dev, patience, delta)
            elif target_domain == "male":
                mse = self.model_train_evaluation(model, male_train, male_dev, patience, delta)
            else:
                mse = self.model_train_evaluation(model, mixed_train, mixed_dev, patience, delta)
        return mse

    def ALL(self, model, file_paths, target_domain, target_size, patience, delta, dev=False):
        female_train = pd.read_csv(file_paths["female_train"])
        female_dev = pd.read_csv(file_paths["female_dev"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_dev = pd.read_csv(file_paths["male_dev"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_dev = pd.read_csv(file_paths["mixed_dev"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])
        if dev == False:
            if target_domain == "female":
                train_female = male_train.copy()
                train_female = train_female.append(mixed_train, ignore_index=True)
                train_female = train_female.append(female_train.sample(target_size), ignore_index=True)
                train_female = train_female.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_female, female_test, patience, delta)

            elif target_domain == "male":
                train_male = female_train.copy()
                train_male = train_male.append(mixed_train, ignore_index=True)
                train_male = train_male.append(male_train.sample(target_size), ignore_index=True)
                train_male = train_male.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_male, male_test, patience, delta)

            else:
                train_mixed = male_train.copy()
                train_mixed = train_mixed.append(female_train, ignore_index=True)
                train_mixed = train_mixed.append(mixed_train.sample(target_size), ignore_index=True)
                train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_mixed, mixed_test, patience, delta)
        else:
            if target_domain == "female":
                train_female = male_train.copy()
                train_female = train_female.append(mixed_train, ignore_index=True)
                train_female = train_female.append(female_train.sample(target_size), ignore_index=True)
                train_female = train_female.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_female, female_dev, patience, delta)

            elif target_domain == "male":
                train_male = female_train.copy()
                train_male = train_male.append(mixed_train, ignore_index=True)
                train_male = train_male.append(male_train.sample(target_size), ignore_index=True)
                train_male = train_male.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_male, male_dev, patience, delta)

            else:
                train_mixed = male_train.copy()
                train_mixed = train_mixed.append(female_train, ignore_index=True)
                train_mixed = train_mixed.append(mixed_train.sample(target_size), ignore_index=True)
                train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
                mse = self.model_train_evaluation(model, train_mixed, mixed_dev, patience, delta)
        return mse

    def WEIGHTED(self, model, file_paths, target_domain, target_size, patience, delta, dev=False):
        female_train = pd.read_csv(file_paths["female_train"])
        female_dev = pd.read_csv(file_paths["female_dev"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_dev = pd.read_csv(file_paths["male_dev"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_dev = pd.read_csv(file_paths["mixed_dev"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])
        if dev == False:
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
                mse = self.model_train_evaluation(model, train_female, female_test, patience, delta)

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
                mse = self.model_train_evaluation(model, train_male, male_test, patience, delta)

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
                mse = self.model_train_evaluation(model, train_mixed, mixed_test, patience, delta)
        else:
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
                mse = self.model_train_evaluation(model, train_female, female_dev, patience, delta)

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
                mse = self.model_train_evaluation(model, train_male, male_dev, patience, delta)

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
                mse = self.model_train_evaluation(model, train_mixed, mixed_dev, patience, delta)
        return mse

    def PRED(self, model, file_paths, new_model, target_domain, target_size, patience, delta, dev=False):
        female_train = pd.read_csv(file_paths["female_train"])
        female_dev = pd.read_csv(file_paths["female_dev"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_dev = pd.read_csv(file_paths["male_dev"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_dev = pd.read_csv(file_paths["mixed_dev"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])
        if dev == False:
            if target_domain == "female":
                train_female = male_train.copy()
                train_female = train_female.append(mixed_train, ignore_index=True)
                train_female = train_female.sample(frac=1).reset_index(drop=True)
                model1 = self.model_train(model, train_female, patience, delta)

                target_domain = female_train.sample(target_size).reset_index(drop=True)
                train_x, train_y = self.get_train(target_domain)
                new_train_feature = model1.predict(train_x)
                new_train_x = self.add_feature(target_domain, new_train_feature)

                test_x, test_y = self.get_train(female_test)
                new_test_feature = model1.predict(test_x)
                new_test_x = self.add_feature(female_test, new_test_feature)
                mse = self.model_train_evaluate(new_model, new_train_x, train_y, new_test_x, test_y)

            elif target_domain == "male":
                train_male = female_train.copy()
                train_male = train_male.append(mixed_train, ignore_index=True)
                train_male = train_male.sample(frac=1).reset_index(drop=True)
                model2 = self.model_train(model, train_male, patience, delta)

                target_domain = male_train.sample(target_size).reset_index(drop=True)
                train_x, train_y = self.get_train(target_domain)
                new_train_feature = model2.predict(train_x)
                new_train_x = self.add_feature(target_domain, new_train_feature)

                test_x, test_y = self.get_train(male_test)
                new_test_feature = model2.predict(test_x)
                new_test_x = self.add_feature(male_test, new_test_feature)
                mse = self.model_train_evaluate(new_model, new_train_x, train_y, new_test_x, test_y)

            else:
                train_mixed = male_train.copy()
                train_mixed = train_mixed.append(female_train, ignore_index=True)
                train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
                model3 = self.model_train(model, train_mixed, patience, delta)

                target_domain = mixed_train.sample(target_size).reset_index(drop=True)
                train_x, train_y = self.get_train(target_domain)
                new_train_feature = model3.predict(train_x)
                new_train_x = self.add_feature(target_domain, new_train_feature)

                test_x, test_y = self.get_train(mixed_test)
                new_test_feature = model3.predict(test_x)
                new_test_x = self.add_feature(mixed_test, new_test_feature)
                mse = self.model_train_evaluate(new_model, new_train_x, train_y, new_test_x, test_y)
        else:
            if target_domain == "female":
                train_female = male_train.copy()
                train_female = train_female.append(mixed_train, ignore_index=True)
                train_female = train_female.sample(frac=1).reset_index(drop=True)
                model1 = self.model_train(model, train_female, patience, delta)

                target_domain = female_train.sample(target_size).reset_index(drop=True)
                train_x, train_y = self.get_train(target_domain)
                new_train_feature = model1.predict(train_x)
                new_train_x = self.add_feature(target_domain, new_train_feature)

                test_x, test_y = self.get_train(female_dev)
                new_test_feature = model1.predict(test_x)
                new_test_x = self.add_feature(female_dev, new_test_feature)
                mse = self.model_train_evaluate(new_model, new_train_x, train_y, new_test_x, test_y)

            elif target_domain == "male":
                train_male = female_train.copy()
                train_male = train_male.append(mixed_train, ignore_index=True)
                train_male = train_male.sample(frac=1).reset_index(drop=True)
                model2 = self.model_train(model, train_male, patience, delta)

                target_domain = male_train.sample(target_size).reset_index(drop=True)
                train_x, train_y = self.get_train(target_domain)
                new_train_feature = model2.predict(train_x)
                new_train_x = self.add_feature(target_domain, new_train_feature)

                test_x, test_y = self.get_train(male_dev)
                new_test_feature = model2.predict(test_x)
                new_test_x = self.add_feature(male_dev, new_test_feature)
                mse = self.model_train_evaluate(new_model, new_train_x, train_y, new_test_x, test_y)

            else:
                train_mixed = male_train.copy()
                train_mixed = train_mixed.append(female_train, ignore_index=True)
                train_mixed = train_mixed.sample(frac=1).reset_index(drop=True)
                model3 = self.model_train(model, train_mixed, patience, delta)

                target_domain = mixed_train.sample(target_size).reset_index(drop=True)
                train_x, train_y = self.get_train(target_domain)
                new_train_feature = model3.predict(train_x)
                new_train_x = self.add_feature(target_domain, new_train_feature)

                test_x, test_y = self.get_train(mixed_dev)
                new_test_feature = model3.predict(test_x)
                new_test_x = self.add_feature(mixed_dev, new_test_feature)
                mse = self.model_train_evaluate(new_model, new_train_x, train_y, new_test_x, test_y)
        return mse

    def LININT_train(self, model, src_domain, tat_domain, dev, test, patience, delta):
        src_model = self.model_train(model, src_domain, patience, delta)
        dev_x, dev_y = self.get_train(dev)
        src_feature = src_model.predict(dev_x).flatten().tolist()

        tat_model = self.model_train(model, tat_domain, patience, delta)
        tat_feature = tat_model.predict(dev_x).flatten().tolist()
        index = [i for i in range(100)]

        data = {'SRCONLY': src_feature, 'TGTONLY': tat_feature}
        new_dev_x = pd.DataFrame(data=data, index=index)

        test_x, test_y = self.get_train(test)
        src_feature = src_model.predict(test_x).flatten().tolist()
        tat_feature = tat_model.predict(test_x).flatten().tolist()
        data = {'SRCONLY': src_feature, 'TGTONLY': tat_feature}
        new_test_x = pd.DataFrame(data=data, index=index)

        model1 = self.simple_model(2)
        mse = self.model_train_evaluate(model1, new_dev_x, dev_y, new_test_x, test_y)
        return mse

    def LININT(self, model, file_paths, target_domain, target_size, patience, delta, dev=False):
        female_train = pd.read_csv(file_paths["female_train"])
        female_dev = pd.read_csv(file_paths["female_dev"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_dev = pd.read_csv(file_paths["male_dev"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_dev = pd.read_csv(file_paths["mixed_dev"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])
        if dev == False:
            if target_domain == "female":
                train_female = male_train.copy()
                train_female = train_female.append(mixed_train, ignore_index=True)
                src_domain = train_female.sample(frac=1).reset_index(drop=True)
                tat_domian = female_train.sample(target_size)
                mse = self.LININT_train(model, src_domain, tat_domian, female_dev, female_test, patience, delta)

            elif target_domain == "male":
                train_male = female_train.copy()
                train_male = train_male.append(mixed_train, ignore_index=True)
                src_domain = train_male.sample(frac=1).reset_index(drop=True)
                tat_domian = male_train.sample(target_size)
                mse = self.LININT_train(model, src_domain, tat_domian, male_dev, male_test, patience, delta)

            else:
                train_mixed = female_train.copy()
                train_mixed = train_mixed.append(male_train, ignore_index=True)
                src_domain = train_mixed.sample(frac=1).reset_index(drop=True)
                tat_domian = mixed_train.sample(target_size)
                mse = self.LININT_train(model, src_domain, tat_domian, mixed_dev, mixed_test, patience, delta)
        else:
            if target_domain == "female":
                train_female = male_train.copy()
                train_female = train_female.append(mixed_train, ignore_index=True)
                src_domain = train_female.sample(frac=1).reset_index(drop=True)
                tat_domian = female_train.sample(target_size)
                mse = self.LININT_train(model, src_domain, tat_domian, female_dev, female_dev, patience, delta)

            elif target_domain == "male":
                train_male = female_train.copy()
                train_male = train_male.append(mixed_train, ignore_index=True)
                src_domain = train_male.sample(frac=1).reset_index(drop=True)
                tat_domian = male_train.sample(target_size)
                mse = self.LININT_train(model, src_domain, tat_domian, male_dev, male_dev, patience, delta)

            else:
                train_mixed = female_train.copy()
                train_mixed = train_mixed.append(male_train, ignore_index=True)
                src_domain = train_mixed.sample(frac=1).reset_index(drop=True)
                tat_domian = mixed_train.sample(target_size)
                mse = self.LININT_train(model, src_domain, tat_domian, mixed_dev, mixed_dev, patience, delta)
        return mse

    def domain_names(self, names, domain):
        return [attributes + "_" + domain for attributes in names]

    def FEDA_dataframe(self, general, target, names, target_names):

        if target is None:
            df = general.copy()
            df.pop("Exam Score")
            target = pd.DataFrame(np.zeros_like(df), columns=target_names)
        else:
            temp = {}
            for i in range(len(names)):
                temp[names[i]] = target_names[i]
            target = target.rename(columns=temp)
            target.pop('Exam Score')
        return target

    def FEDA_concat(self, general, female, male, mixed):
        names = ["Year", "FSM", "VR1 Band", "VR Band of Student", "Ethnic group of student", "School denomination"]
        female_names = self.domain_names(names, "FEMALE")
        male_names = self.domain_names(names, "MALE")
        mixed_names = self.domain_names(names, "MIXED")

        female = self.FEDA_dataframe(general, female, names, female_names)
        male = self.FEDA_dataframe(general, male, names, male_names)
        mixed = self.FEDA_dataframe(general, mixed, names, mixed_names)

        result = pd.concat([general, female, male, mixed], axis=1)
        return result

    def FEDA(self, file_paths):
        FEDA_file_paths = {}
        # the new vector is (general, female, male, mixed)
        female_train = pd.read_csv(file_paths["female_train"])
        female_dev = pd.read_csv(file_paths["female_dev"])
        female_test = pd.read_csv(file_paths["female_test"])
        male_train = pd.read_csv(file_paths["male_train"])
        male_dev = pd.read_csv(file_paths["male_dev"])
        male_test = pd.read_csv(file_paths["male_test"])
        mixed_train = pd.read_csv(file_paths["mixed_train"])
        mixed_dev = pd.read_csv(file_paths["mixed_dev"])
        mixed_test = pd.read_csv(file_paths["mixed_test"])

        female_train_FEDA_path = "FEDA DATA/FEMALE_train.csv"
        female_dev_FEDA_path = "FEDA DATA/FEMALE_dev.csv"
        female_test_FEDA_path = "FEDA DATA/FEMALE_test.csv"
        male_train_FEDA_path = "FEDA DATA/MALE_train.csv"
        male_dev_FEDA_path = "FEDA DATA/MALE_dev.csv"
        male_test_FEDA_path = "FEDA DATA/MALE_test.csv"
        mixed_train_FEDA_path = "FEDA DATA/MIXED_train.csv"
        mixed_dev_FEDA_path = "FEDA DATA/MIXED_dev.csv"
        mixed_test_FEDA_path = "FEDA DATA/MIXED_test.csv"

        female_train_FEDA = self.FEDA_concat(female_train, female_train, None, None)
        female_train_FEDA.to_csv(female_train_FEDA_path, index=0)
        female_dev_FEDA = self.FEDA_concat(female_dev, female_dev, None, None)
        female_dev_FEDA.to_csv(female_dev_FEDA_path, index=0)
        female_test_FEDA = self.FEDA_concat(female_test, female_test, None, None)
        female_test_FEDA.to_csv(female_test_FEDA_path, index=0)

        male_train_FEDA = self.FEDA_concat(male_train, None, male_train, None)
        male_train_FEDA.to_csv(male_train_FEDA_path, index=0)
        male_dev_FEDA = self.FEDA_concat(male_dev, None, male_dev, None)
        male_dev_FEDA.to_csv(male_dev_FEDA_path, index=0)
        male_test_FEDA = self.FEDA_concat(male_test, None, male_test, None)
        male_test_FEDA.to_csv(male_test_FEDA_path, index=0)

        mixed_train_FEDA = self.FEDA_concat(mixed_train, None, None, mixed_train)
        mixed_train_FEDA.to_csv(mixed_train_FEDA_path, index=0)
        mixed_dev_FEDA = self.FEDA_concat(male_dev, None, None, mixed_dev)
        mixed_dev_FEDA.to_csv(mixed_dev_FEDA_path, index=0)
        mixed_test_FEDA = self.FEDA_concat(male_test, None, None, mixed_test)
        mixed_test_FEDA.to_csv(mixed_test_FEDA_path, index=0)

        FEDA_file_paths["female_train"] = female_train_FEDA_path
        FEDA_file_paths["female_test"] = female_test_FEDA_path
        FEDA_file_paths["female_dev"] = female_dev_FEDA_path
        FEDA_file_paths["male_train"] = male_train_FEDA_path
        FEDA_file_paths["male_test"] = male_test_FEDA_path
        FEDA_file_paths["male_dev"] = male_dev_FEDA_path
        FEDA_file_paths["mixed_train"] = mixed_train_FEDA_path
        FEDA_file_paths["mixed_test"] = mixed_test_FEDA_path
        FEDA_file_paths["mixed_dev"] = mixed_dev_FEDA_path

        return FEDA_file_paths

    def dict_update(self, dict, mse, para, label):
        if dict.get(label) is not None:
            temp = dict.get(label)
            if mse < temp.get("mse"):
                temp["mse"] = mse
                temp["parameters"] = para
        else:
            temp = {}
            temp["mse"] = mse
            temp["parameters"] = para
            dict[label] = temp
        return dict

    def parameter_generate(self):
        units1 = random.randint(10, 20)
        units2 = random.randint(units1,30)
        patience = random.randint(1, 10)
        delta = random.randint(-5, -1)
        return units1, units2, patience, delta

    def FEDA_para_generate(self):
        units1 = random.randint(30, 40)
        units2 = random.randint(units1, 60)
        patience = random.randint(1, 10)
        delta = random.randint(-5, -1)
        return units1, units2, patience, delta

if __name__ == '__main__':
    ann = ANN()

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

    ann.train_dev_test_split(female_path, female_train_path, female_dev_path, female_test_path)
    ann.train_dev_test_split(male_path, male_train_path, male_dev_path, male_test_path)
    ann.train_dev_test_split(mixed_path, mixed_train_path, mixed_dev_path, mixed_test_path)

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

    FEDA_file_paths = ann.FEDA(file_paths)
    # model = ann.get_compiled_model(6)
    Domains = ["female", "male", "mixed"]
    Functions = ["SRCONLY", "TGTONLY", "ALL", "WEIGHTED", "PRED", "FEDA"]
    # target_domain = "mixed"
    target_size = 100
    mses = {}
    # mse["Domain"] = target_domain + " " + str(target_size)
    # mse["SRCONLY"] = ann.SRCONLY(model, file_paths, target_domain, target_size)
    # mse["TGTONLY"] = ann.TGTONLY(model, file_paths, target_domain, target_size)
    # mse["ALL"] = ann.ALL(model, file_paths, target_domain, target_size)
    # mse["WEIGHTED"] = ann.WEIGHTED(model, file_paths, target_domain, target_size)
    # mse["PRED"] = ann.PRED(model, file_paths, 6, target_domain, target_size)
    # mse["LINTINT"] = ann.LININT(model, file_paths, target_domain, target_size)
    # feda_model = ann.FEDA_model(24)
    # mse["FEDA"] = ann.ALL(feda_model, FEDA_file_paths, target_domain, target_size)

    # hyper parameter tunning
    dev = True
    # count = 0
    # total = len(Functions) * len(Domains) * 100
    for i in Functions:
        for j in Domains:
            if i == "ALL":
                for k in range(100):
                    units1, units2, patience, delta = ann.parameter_generate()
                    model = ann.get_compiled_model(6, units1, units2)
                    mse = ann.ALL(model, file_paths, j, target_size, patience, delta, dev=dev)
                    para = [units1, units2, patience, delta]
                    labels = i + "_" + j
                    mses = ann.dict_update(mses, mse, para, labels)
                with open("Data/hyper-parameter_tuning.json", 'w') as fp:
                    json.dump(mses, fp)

            if i == "SRCONLY":
                for k in range(100):
                    units1, units2, patience, delta = ann.parameter_generate()
                    model = ann.get_compiled_model(6, units1, units2)
                    mse = ann.SRCONLY(model, file_paths, j, target_size, patience, delta, dev=dev)
                    para = [units1, units2, patience, delta]
                    labels = i + "_" + j
                    mses = ann.dict_update(mses, mse, para, labels)
                with open("Data/hyper-parameter_tuning.json", 'w') as fp:
                    json.dump(mses, fp)

            if i == "TGTONLY":
                for k in range(100):
                    units1, units2, patience, delta = ann.parameter_generate()
                    model = ann.get_compiled_model(6, units1, units2)
                    mse = ann.TGTONLY(model, file_paths, j, target_size, patience, delta, dev=dev)
                    para = [units1, units2, patience, delta]
                    labels = i + "_" + j
                    mses = ann.dict_update(mses, mse, para, labels)
                with open("Data/hyper-parameter_tuning.json", 'w') as fp:
                    json.dump(mses, fp)

            if i == "WEIGHTED":
                for k in range(100):
                    units1, units2, patience, delta = ann.parameter_generate()
                    model = ann.get_compiled_model(6, units1, units2)
                    mse = ann.WEIGHTED(model, file_paths, j, target_size, patience, delta, dev=dev)
                    para = [units1, units2, patience, delta]
                    labels = i + "_" + j
                    mses = ann.dict_update(mses, mse, para, labels)
                with open("Data/hyper-parameter_tuning.json", 'w') as fp:
                    json.dump(mses, fp)


            if i == "FEDA":
                for k in range(100):
                    units1, units2, patience, delta = ann.FEDA_para_generate()
                    model = ann.get_compiled_model(24, units1, units2)
                    mse = ann.ALL(model, FEDA_file_paths, j, target_size, patience, delta, dev=dev)
                    para = [units1, units2, patience, delta]
                    labels = i + "_" + j
                    mses = ann.dict_update(mses, mse, para, labels)
                with open("Data/hyper-parameter_tuning.json", 'w') as fp:
                    json.dump(mses, fp)

            if i == "PRED":
                for k in range(100):
                    units1, units2, patience, delta = ann.parameter_generate()
                    model_1 = ann.get_compiled_model(6, units1, units2)
                    model_2 = ann.get_compiled_model(7, units1, units2)
                    mse = ann.PRED(model_1, file_paths, model_2, j, target_size, patience, delta, dev=dev)
                    para = [units1, units2, patience, delta]
                    labels = i + "_" + j
                    mses = ann.dict_update(mses, mse, para, labels)
                with open("Data/hyper-parameter_tuning.json", 'w') as fp:
                    json.dump(mses, fp)
    # for i in Domains:
    #     for k in range(100):
    #         units1, units2, patience, delta = ann.FEDA_para_generate()
    #         model = ann.get_compiled_model(24, units1, units2)
    #         mse = ann.ALL(model, FEDA_file_paths, i, target_size, patience, delta, dev=dev)
    #         para = [units1, units2, patience, delta]
    #         label = "FEDA_" + i
    #         mses = ann.dict_update(mses, mse, para, label)
    #     print("FEDA finished")
    #
    #     with open("Data/hyper-parameter_tuning_1.json", 'w') as fp:
    #         json.dump(mses, fp)


    print(mses)