import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import display, Markdown
from keras.callbacks import TensorBoard

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score

# from __future__ import division
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano

theano.config.floatX = 'float32'

import warnings
warnings.filterwarnings("ignore")

def load_data():
    features = pd.read_csv('data/features.csv',delimiter=";")
    names = list(features.Feature)
    X_train = pd.read_csv('data/train/X_train.txt', delim_whitespace=True, header=None, names=names)
    activity = {'0':'WALKING','1':'WALKING_UPSTAIRS','2':'WALKING_DOWNSTAIRS','3':'SITTING','4':'STANDING','5':'LAYING'}
    order = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']

    y_train = pd.read_csv('data/train/y_train.txt', delim_whitespace=True, header=None, names=['activity'])
    y_train.activity = y_train.activity - 1
    X_test = pd.read_csv('data/test/X_test.txt', delim_whitespace=True, header=None, names=names)
    y_test = pd.read_csv('data/test/y_test.txt', delim_whitespace=True, header=None, names=['activity'])
    y_test.activity = y_test.activity - 1

    names_EDA = list(names)
    names_EDA.extend(['activity'])
    EDA_train = np.hstack((X_train,y_train))
    EDA_test = np.hstack((X_test,y_test))
    EDA = np.vstack((EDA_train,EDA_test))
    EDA = pd.DataFrame(EDA,columns=names_EDA)
    EDA['activity_name'] = EDA['activity'].apply(lambda x: activity[str(int(x))])

    return features, names, activity, order, X_train, y_train, X_test, y_test, EDA, names_EDA


def violin_plot(EDA):
    fig, axes = plt.subplots(3,1, figsize=(20,30))
    feat = list(features.Feature[0:3])

    for c, ax in zip(feat,axes.flatten()):
        ax.set_title(c.upper(),fontsize = 20)
        ax.set_xticklabels(order, size=16)
        sns.violinplot(x='activity_name', y=c, data=EDA,width=1, order=order, ax=ax)
        ax.set_xlabel('Activities', size = 15)
        ax.set_ylabel('', size = 15)

    plt.savefig('images/acc_body_mean.png')

    fig, axes = plt.subplots(3,1, figsize=(20,30))
    feat = list(features.Feature[3:6])

    for c, ax in zip(feat,axes.flatten()):
        ax.set_title(c.upper(),fontsize = 20)
        ax.set_xticklabels(order, size=16)
        sns.violinplot(x='activity_name', y=c, data=EDA,width=1, order=order, ax=ax)
        ax.set_xlabel('Activities', size = 15)
        ax.set_ylabel('', size = 15)

    plt.savefig('images/acc_body_std.png')

    fig, axes = plt.subplots(3,1, figsize=(20,30))
    feat = list(features.Feature[40:43])

    for c, ax in zip(feat,axes.flatten()):
        ax.set_title(c.upper(),fontsize = 20)
        ax.set_xticklabels(order, size=16)
        sns.violinplot(x='activity_name', y=c, data=EDA,width=1, order=order, ax=ax)
        ax.set_xlabel('Activities', size = 15)
        ax.set_ylabel('', size = 15)

    plt.savefig('images/acc_grav_mean.png')

    fig, axes = plt.subplots(3,1, figsize=(20,30))
    feat = list(features.Feature[43:46])

    for c, ax in zip(feat,axes.flatten()):
        ax.set_title(c.upper(),fontsize = 20)
        ax.set_xticklabels(order, size=16)
        sns.violinplot(x='activity_name', y=c, data=EDA,width=1, order=order, ax=ax)
        ax.set_xlabel('Activities', size = 15)
        ax.set_ylabel('', size = 15)

    plt.savefig('images/acc_grav_std.png')

    fig, axes = plt.subplots(3,1, figsize=(20,30))
    feat = list(features.Feature[120:123])

    for c, ax in zip(feat,axes.flatten()):
        ax.set_title(c.upper(),fontsize = 20)
        ax.set_xticklabels(order, size=16)
        sns.violinplot(x='activity_name', y=c, data=EDA,width=1, order=order, ax=ax)
        ax.set_xlabel('Activities', size = 15)
        ax.set_ylabel('', size = 15)

    plt.savefig('images/gyro_body_mean.png')

    fig, axes = plt.subplots(3,1, figsize=(20,30))
    feat = list(features.Feature[123:126])

    for c, ax in zip(feat,axes.flatten()):
        ax.set_title(c.upper(),fontsize = 20)
        ax.set_xticklabels(order, size=16)
        sns.violinplot(x='activity_name', y=c, data=EDA,width=1, order=order, ax=ax)
        ax.set_xlabel('Activities', size = 15)
        ax.set_ylabel('', size = 15)

    plt.savefig('images/gyro_body_std.png')

    fig, axes = plt.subplots(3,1, figsize=(20,30))
    feat = list(features.Feature[558:562])

    for c, ax in zip(feat,axes.flatten()):
        ax.set_title(c.upper(),fontsize = 20)
        ax.set_xticklabels(order, size=16)
        sns.violinplot(x='activity_name', y=c, data=EDA,width=1, order=order, ax=ax)
        ax.set_xlabel('Activities', size = 15)
        ax.set_ylabel('', size = 15)

    plt.savefig('images/angles.png')

def correlation_plots(features,EDA):
    cond1 = features.Group == 'Accelerometer'
    cond2 = features.Part == 'Body'
    feat = list(features.Feature[cond1 & cond2])


    fig, ax = plt.subplots(1,1, figsize=(20,10))
    ax.set_title('Accelerometer Body Values',fontsize = 20)
    #feat = list(features.Feature[121:136])
    sns.set(style="white")
    mask = np.zeros_like(EDA[feat].corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(EDA[feat].corr(),mask=mask, annot=False, fmt=".2f",ax=ax, square = True, cmap=cmap);
    plt.savefig('images/correlation_Acc_Body.png')

    cond1 = features.Group == 'Accelerometer'
    cond2 = features.Part == 'Gravity'
    feat = list(features.Feature[cond1 & cond2])


    fig, ax = plt.subplots(1,1, figsize=(20,10))
    ax.set_title('Accelerometer Gravity Values',fontsize = 20)
    #feat = list(features.Feature[121:136])
    sns.set(style="white")
    mask = np.zeros_like(EDA[feat].corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(EDA[feat].corr(),mask=mask, annot=False, fmt=".2f",ax=ax, square = True, cmap=cmap);
    plt.savefig('images/correlation_Acc_Grav.png')

    cond1 = features.Group == 'Gyroscope'
    cond2 = features.Part == 'Body'
    feat = list(features.Feature[cond1 & cond2])


    fig, ax = plt.subplots(1,1, figsize=(20,10))
    ax.set_title('Gyroscope Body Values',fontsize = 20)
    #feat = list(features.Feature[121:136])
    sns.set(style="white")
    mask = np.zeros_like(EDA[feat].corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(EDA[feat].corr(),mask=mask, annot=False, fmt=".2f",ax=ax, square = True, cmap=cmap);
    plt.savefig('images/correlation_Gyro.png')

def pca(EDA, names_EDA):
    X_pca = pd.DataFrame(EDA,columns=names_EDA)
    y_pca = X_pca.pop('activity')
    #PCA
    pca = PCA(n_components=36)
    pca.fit(X_pca)
    #pca.explained_variance_ratio_
    #pca.singular_values_
    X_pca = pca.fit_transform(X_pca)
    X_train_pca = X_pca[0:7352]
    X_test_pca = X_pca[7352:]
    return pca, X_train_pca, X_test_pca, X_pca, y_pca

def scree_plot(pca):
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    cum_var = np.cumsum(vals)
    ax = plt.subplot(111)

    ax.plot(range(len(vals) + 1), np.insert(cum_var, 0, 0), color = 'r', marker = 'o')
    ax.bar(range(len(vals)), vals, alpha = 0.8)

    ax.axhline(0.9, color = 'g', linestyle = "--")
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    plt.title("Scree Plot", fontsize=16)
    plt.savefig('images/PCA_36.png')

    from IPython.display import display, Markdown
    texto  = "| Nº Component |     %    |\n"
    texto += "|--------------|----------|\n"
    var_acum = 0
    for i in range(pca.n_components_):
        var_acum += pca.explained_variance_ratio_[i]
        line = "| {0} |  {1:0.2%} |".format(int(i+1),var_acum)
        texto += line + "\n"
    display(Markdown("### PCA variance ratio"))
    display(Markdown(texto))

def plot_embedding(X, y, activity, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], activity[str(int(y[i]))], color=plt.cm.Set1(y[i] / 6.),
                 fontdict={'size': 10})

    ax.annotate("STATIC POSTURES",
            xy=(0.3, 0.3),
            xycoords='data',
            xytext=(0.2, 1),
            textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

    ax.annotate("DYNAMIC ACTIVITIES",
            xy=(0.8, 0.5),
            xycoords='data',
            xytext=(1, 0),
            textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

    plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,1.1])

    if title is not None:
        plt.title(title, fontsize=16)
    plt.savefig('images/embedding_pca.png')

def compare_predict(y_test,y_hat):
    y_test1 = np.array(y_test)
    compare = np.hstack((y_test1,y_hat.reshape(-1,1)))
    compare = pd.DataFrame(compare,columns=['Real','Predict'])
    temp = pd.DataFrame(pd.pivot_table(compare, values='Predict', index='Real',
                                    columns='Predict', aggfunc=len))
    temp = pd.DataFrame(np.array(temp.to_records()))
    temp.rename(columns=activity, inplace=True)
    temp['Real'] = temp['Real'].apply(lambda x: activity[str(x)])
    compare_predict = temp
    return compare_predict

def model_dt(X_train_pca, y_train, X_test_pca, y_test):
    pipeline_dt = Pipeline([
                    ('imputer', Imputer(strategy='median')),
                    ('std_scaler', StandardScaler()),
                    ('decisionTree', DecisionTreeClassifier()),
                    ])

    param_grid = {'decisionTree__min_samples_split':[2,5],
                 'decisionTree__presort':[True, False],
                 'decisionTree__splitter':['best','random'],
                 'decisionTree__max_depth':[None,3,5,10],
                 'decisionTree__min_samples_leaf':[1,2,3],
                 'decisionTree__max_leaf_nodes':[None,6,10]}

    dt = GridSearchCV(pipeline_dt, param_grid=param_grid)

    dt.fit(X_train_pca,y_train)
    y_hat_pdt = dt.predict(X_test_pca)
    print ('Decision Tree - Accuracy Score: {0:0.2%}'.format(accuracy_score(y_test,y_hat_pdt)))
    compare_predict(y_test,y_hat_pdt)
    return dt, y_hat_pdt

def model_bag(X_train_pca, y_train, X_test_pca, y_test):
    pipeline_bag = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler()),
        ('bagging', BaggingClassifier()),
        ])

    param_grid = {}

    bag = GridSearchCV(pipeline_bag, param_grid=param_grid)

    bag.fit(X_train_pca,y_train)
    y_hat_pbag = bag.predict(X_test_pca)
    print ('Bagging - Accuracy Score: {0:0.2%}'.format(accuracy_score(y_test,y_hat_pbag)))
    return bag, y_hat_pbag

def model_rf(X_train_pca, y_train, X_test_pca, y_test):
    pipeline_rf = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler()),
        ('randomforest', RandomForestClassifier()),
        ])

    param_grid = {'randomforest__n_estimators':[10,30,50],
                  'randomforest__min_samples_split':[2,5,10]}

    rf = GridSearchCV(pipeline_rf, param_grid=param_grid)

    rf.fit(X_train_pca,y_train)
    y_hat_prf = rf.predict(X_test_pca)
    print ('RandomForest - Accuracy Score: {0:0.2%}'.format(accuracy_score(y_test,y_hat_prf)))
    return rf, y_hat_prf

def model_kn(X_train_pca, y_train, X_test_pca, y_test):
  pipeline_kn = Pipeline([
      ('imputer', Imputer(strategy='median')),
      #('std_scaler', StandardScaler()),
      ('kneighbors', KNeighborsClassifier()),
      ])

  param_grid = {'kneighbors__n_neighbors':[5,10,30],
                'kneighbors__weights':['uniform','distance']}

  kn = GridSearchCV(pipeline_kn, param_grid=param_grid)

  kn.fit(X_train_pca,y_train)
  y_hat_pkn = kn.predict(X_test_pca)
  print ('KNeighbors - Accuracy Score: {0:0.2%}'.format(accuracy_score(y_test,y_hat_pkn)))
  return kn, y_hat_pkn

def model_ada(X_train_pca, y_train, X_test_pca, y_test):
  pipeline_ada = Pipeline([
      ('imputer', Imputer(strategy='median')),
      ('std_scaler', StandardScaler()),
      ('adaboost', AdaBoostClassifier()),
      ])

  param_grid = {'adaboost__learning_rate':[0.3, 0.5],
                'adaboost__n_estimators':[700, 900]}

  ada = GridSearchCV(pipeline_ada, param_grid=param_grid)

  ada.fit(X_train_pca,y_train)
  y_hat_pada = ada.predict(X_test_pca)
  print ('AdaBoost - Accuracy Score: {0:0.2%}'.format(accuracy_score(y_test,y_hat_pada)))
  return ada, y_hat_pada

def model_svc(X_train_pca, y_train, X_test_pca, y_test):
    pipeline_svc = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler()),
        ('svc', SVC()),
        ])

    param_grid = {'svc__C':[0.1,1,10],
                  'svc__kernel':['rbf','linear', 'poly', 'sigmoid']}

    svc = GridSearchCV(pipeline_svc, param_grid=param_grid)
    svc.fit(X_train_pca,y_train)
    y_hat_psvc = svc.predict(X_test_pca)
    print ('SVC - Accuracy Score: {0:0.2%}'.format(accuracy_score(y_test,y_hat_psvc)))
    return svc, y_hat_psvc

def model_gb(X_train_pca, y_train, X_test_pca, y_test):
    pipeline_gb = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler()),
        ('gradientboosting', GradientBoostingClassifier()),
        ])


    param_grid = {'gradientboosting__n_estimators':[100,300],
                  'gradientboosting__learning_rate':[0.1,0.5,1]}

    gb = GridSearchCV(pipeline_gb, param_grid=param_grid)
    gb.fit(X_train_pca,y_train)
    y_hat_pgb = gb.predict(X_test_pca)
    print ('Gradient Boosting - Accuracy Score: {0:0.2%}'.format(accuracy_score(y_test,y_hat_pgb)))
    return gb, y_hat_pgb

def define_nn_mlp_model(X_train, y_train_ohe):
    ''' defines multi-layer-perceptron neural network '''
    model = Sequential() # sequence of layers
    num_neurons_in_layer = 16 # number of neurons in a layer
    num_inputs = X_train.shape[1] # number of features
    num_classes = 6  # number of classes
    model.add(Dense(units=num_neurons_in_layer,
                    use_bias=True,
                    input_dim=num_inputs,
                    bias_initializer='zeros',
                    kernel_initializer='uniform',
                    activation='relu'))

    model.add(Dense(units=num_neurons_in_layer,
                use_bias=True,
                input_dim=num_neurons_in_layer,
                bias_initializer='zeros',
                kernel_initializer='uniform',
                activation='relu'))


    model.add(Dense(units=num_classes,
                    use_bias=True,
                    input_dim=num_neurons_in_layer,
                    bias_initializer='zeros',
                    kernel_initializer='uniform',
                    activation='softmax'))
    sgd = SGD(lr=0.002, decay=1e-7, momentum=.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] )
    return model

if __name__ == "__main__":
  features, names, activity, order, X_train, y_train, X_test, y_test, EDA, names_EDA = load_data()
  violin_plot(EDA)
  correlation_plots(features,EDA)
  pca, X_train_pca, X_test_pca, X_pca, y_pca = pca(EDA, names_EDA)
  scree_plot(pca)
  plot_embedding(X_pca,y_pca,activity, "PLOT PCA – 2 FIRSTS COMPONENTS (67.00% VARIANCE)")

  #Decision Tree
  dt, y_hat_pdt = model_dt(X_train_pca, y_train, X_test_pca, y_test)
  compare_predict(y_test,y_hat_pdt)

  #Bagging
  bag, y_hat_pbag = model_bag(X_train_pca, y_train, X_test_pca, y_test)
  compare_predict(y_test,y_hat_pbag)

  #RandomForest
  rf, y_hat_prf = model_rf(X_train_pca, y_train, X_test_pca, y_test)
  compare_predict(y_test,y_hat_prf)

  #KNeighbors
  kn, y_hat_pkn = model_kn(X_train_pca, y_train, X_test_pca, y_test)
  compare_predict(y_test,y_hat_pkn)

  #AdaBoost
  ada, y_hat_pada = model_ada(X_train_pca, y_train, X_test_pca, y_test)
  compare_predict(y_test,y_hat_pada)

  #SVC
  svc, y_hat_psvc = model_svc(X_train_pca, y_train, X_test_pca, y_test)
  compare_predict(y_test,y_hat_psvc)

  #Gradient Boosting
  gb, y_hat_pgb = model_gb(X_train_pca, y_train, X_test_pca, y_test)
  compare_predict(y_test,y_hat_pgb)


  #Neural Network
  X_train_nn = np.array(X_train).astype(theano.config.floatX)
  X_test_nn = np.array(X_test).astype(theano.config.floatX)
  X_train_nn.resize(len(y_train), 561)
  X_test_nn.resize(len(y_test), 561)
  y_train_ohe = np_utils.to_categorical(y_train)

  batch_size = X_train_nn.shape[0] # i.e. batch gradient descent

  tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)

  rng_seed = 0 # set random number generator seed
  np.random.seed(rng_seed)
  #model.load_weights('param_nn.txt')
  model = define_nn_mlp_model(X_train_nn, y_train_ohe)
  # Hmm, the fit uses 5 epochs with a batch_size of 5000.  I wonder if that's best?
  model.fit(X_train, y_train_ohe, epochs=20, batch_size=15, verbose=1,
            validation_split=0.1, callbacks = [tensorboard]) # cross val to estimate test error, can monitor overfitting


  y_hat_nn = model.predict_classes(X_test_nn, verbose=0)
  print ('Neural Network - Accuracy Score: {0:0.2%}'.format(accuracy_score(y_test,y_hat_nn)))
  compare_predict(y_test,y_hat_nn)
