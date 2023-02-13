import tensorflow as tf
import pandas as pd
import numpy as np
import xgboost
import optuna
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
from pyxtal import pyxtal
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, precision_score, recall_score, \
    roc_auc_score, roc_curve, mean_absolute_error
from tensorflow import keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.mixture import GaussianMixture


# two functions to see what are the dependencies between the elements

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            #check for the last word after the last _ in each columns

                pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop
def get_top_abs_correlations(df, n):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop)
    au_corr=au_corr.sort_values(ascending=False)

    return au_corr[0:n]
def resize(file):
    label_to_drop=[]
    for column in file:
        max=0
        max=file[column].max()
        if max==0:
         label_to_drop.append(column)
    return label_to_drop
def gaussian(matrix,plotbig=False,plotcluster=False):
 n_clusters=np.arange(1,30)
 modello=[GaussianMixture(n,covariance_type='full',random_state=42).fit(matrix) for n in n_clusters]
 bic=[m.bic(matrix) for m in modello]
 aic=[m.aic(matrix) for m in modello]
 if plotbig:
  plt.plot(n_clusters,[m.bic(matrix) for m in modello],label='BIC')
  plt.plot(n_clusters,[m.aic(matrix) for m in modello],label='AIC')
  plt.xlabel('n_cluster')
 modellogiusto=GaussianMixture(n_components=np.argmin(bic),covariance_type='full',random_state=42).fit(matrix)
 predict=modellogiusto.predict(matrix)
 if plotcluster:
  df_sub=matrix[['mean_ThermalConductivity','critical_temp']].values
  for n in range(np.argmin(bic)):
    plt.scatter(df_sub[predict==n, 0], df_sub[predict==n, 1], s=100, label ='Cluster ')
  plt.title('Cluster of Superconductors')
  plt.ylabel('temp')
  plt.show()
 clustermatrix=pd.DataFrame(predict,columns=['cluster'])
 matrix=matrix.join(clustermatrix)
 return matrix

def correlation(matrix):
#see for correlation with the critical temperature
 corr_mat=matrix.corr()
 corr_mat=corr_mat["critical_temp"].sort_values(ascending=False)
 print('this is the correlation matrix',corr_mat)
#plotting the scatter matrix we see what we expect looking at the results from get_top_abs_correlation
 scatter_matrix(matrix[corr_mat.index],figsize=(12,8))


caratteristiche = pd.read_csv(r"C:\Users\nicol\OneDrive\Documenti\python book\superconduct\train.csv")
elementi = pd.read_csv(r"C:\Users\nicol\OneDrive\Documenti\python book\superconduct\unique_m.csv")
# drop the columns with all zero values
elementi = elementi.loc[:, (elementi != 0).any(axis=0)]
# unique contains all the single elements the material is composed of so we unite this 2 dataframes
elementi = elementi.drop(["critical_temp", "material"], axis=1)
caratteristiche = caratteristiche.join(elementi)
# watch too much correlated matrix
print('lets look at the linear correlation between variables',get_top_abs_correlations(caratteristiche,10))
# drop too much correlated columns
caratteristiche = caratteristiche.drop(
    ["range_atomic_radius", "mean_fie", "mean_atomic_mass", "mean_atomic_radius", "wtd_gmean_ElectronAffinity",
     "range_atomic_mass", "number_of_elements", "range_fie", "entropy_atomic_radius", "entropy_atomic_mass",
     "entropy_fie", "wtd_mean_atomic_mass", "gmean_atomic_mass", "wtd_gmean_atomic_mass", "wtd_entropy_atomic_mass",
     "wtd_range_atomic_mass", "std_atomic_mass", "wtd_std_atomic_mass", "wtd_mean_fie", "gmean_fie", "wtd_gmean_fie",
     "wtd_entropy_fie", "wtd_range_fie", "std_fie", "wtd_std_fie", "wtd_mean_atomic_radius", "gmean_atomic_radius",
     "wtd_gmean_atomic_radius", "wtd_entropy_atomic_radius", "wtd_range_atomic_radius", "std_atomic_radius",
     "wtd_std_atomic_radius", "wtd_mean_Density", "gmean_Density", "wtd_gmean_Density", "entropy_Density",
     "wtd_entropy_Density", "range_Density", "wtd_range_Density", "std_Density", "wtd_std_Density",
     "wtd_mean_ElectronAffinity", "gmean_ElectronAffinity", "entropy_ElectronAffinity", "wtd_entropy_ElectronAffinity",
     "range_ElectronAffinity", "wtd_range_ElectronAffinity", "std_ElectronAffinity", "wtd_std_ElectronAffinity",
     "wtd_mean_FusionHeat", "gmean_FusionHeat", "wtd_gmean_FusionHeat", "entropy_FusionHeat", "wtd_entropy_FusionHeat",
     "range_FusionHeat", "wtd_range_FusionHeat", "std_FusionHeat", "wtd_std_FusionHeat", "wtd_std_ThermalConductivity",
     "wtd_mean_ThermalConductivity", "gmean_ThermalConductivity", "wtd_gmean_ThermalConductivity",
     "entropy_ThermalConductivity", "wtd_entropy_ThermalConductivity", "range_ThermalConductivity",
     "wtd_range_ThermalConductivity", "std_ThermalConductivity", "wtd_mean_Valence", "gmean_Valence",
     "wtd_gmean_Valence", "entropy_Valence", "wtd_entropy_Valence", "range_Valence", "wtd_range_Valence", "std_Valence",
     "wtd_std_Valence"], axis=1)
caratteristiche = caratteristiche.drop(resize(caratteristiche), axis=1)
#let's see if there are any better results
print('lets look at the linear correlation between variables',get_top_abs_correlations(caratteristiche,10))
# shuffle the data rows and columns
caratteristiche = caratteristiche.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)
# elimina righe con elementi nulli
caratteristiche.dropna()
# drop all equals rows
insurance = caratteristiche.drop_duplicates(keep='first')
correlation(caratteristiche)
# linearità tra range raggio atomo e entropia elletroaffinità più raggio è piccolo più la elettroaffinità è alta perchè è più difficile portargli via gli elettroni a differenza di un atomo con raggio grande
# essendoci la densità possiamo togliere la massa in quanto ridondante idem il raggio in quanto densità=mass/volume
# drop also fie since i don't know what it should be
# range atomic radius è già dentro la densità
# standardization
scaler = preprocessing.StandardScaler()
names = caratteristiche.columns
d = scaler.fit_transform(insurance)
caratteristiche = pd.DataFrame(d, columns=names)
# 28 è il numero di cluster ideale
caratteristiche = gaussian(caratteristiche)
# drop all clusters with only one value
caratteristiche = caratteristiche[caratteristiche.duplicated('cluster', keep=False)]
X = caratteristiche.drop(["critical_temp", "cluster"], axis=1)
Y = caratteristiche["critical_temp"]
# now let's apply dimensionality reduction into 2 dimension keeping the 95% of the variance
# pca=PCA(0.95)
# pca.fit(X)
# X=pca.transform(X)
# GaussianMixture

# dopo questo con np.argmin() vediamo che 17 sembrerebbe essere il numero di cluster ideale
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42,
                                                    stratify=caratteristiche['cluster'])
# OPTUNA is way better then Grid Search
# let's use a Optuna,Next we’ll use Optuna to tune the hyperparameters of the XGBRegressor model.
# Optuna lets you tune the hyperparameters of any model, not just XGBoost models.
# The first step in the process is to define an objective function.
# The objective function is the function that Optuna will try to optimize.
# In our case, we’re trying to minimize the mean squared error.

def create_model(trial):
    model_type = trial.suggest_categorical('model_type', ['linearegression', 'decisiontree', 'xgboost','mlpregressor'])

    if model_type == 'xgboost':
            max_depth=trial.suggest_int('max_depth', 1, 10)
            learning= trial.suggest_float('learning', 0.01, 1.0)
            n_estimators= trial.suggest_int('n_estimators', 50, 1000)
            min_child_weight= trial.suggest_int('min_child_weight', 1, 10)
            gamma= trial.suggest_float('gamma', 0.01, 1.0)
            subsample= trial.suggest_float('subsample', 0.01, 1.0)
            colsample_bytree= trial.suggest_float('colsample_bytree', 0.01, 1.0)
            reg_alpha= trial.suggest_float('reg_alpha', 0.01, 1.0)
            reg_lambda= trial.suggest_float('reg_lambda', 0.01, 1.0)
            random_state= trial.suggest_int('random_state', 1, 1000)
            model= XGBRegressor(max_depth=max_depth,learning_rate=learning,n_estimators=n_estimators,min_child_weight=min_child_weight,gamma=gamma,subsample=subsample,colsample_bytree=colsample_bytree,reg_alpha=reg_alpha,reg_lambda=reg_lambda,random_state=random_state)

    if model_type == 'linearegression':
        max_iter = trial.suggest_int('max_iter', 1,10)

        alpha = trial.suggest_categorical('alpha', [0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
        l1_ratio=trial.suggest_float('l1_ratio',0,1)
        model = ElasticNet(max_iter=max_iter, alpha=alpha, l1_ratio=l1_ratio)

    if model_type == 'decisiontree':
        splitter= trial.suggest_categorical('splitter',["best", "random"])
        max_depth = trial.suggest_int('max_depth', 1, X_train.shape[1])#numero di colonne di x_train
        min_samples_leaf= trial.suggest_int('min_samples_leaf',2,20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_weight_fraction_leaf= trial.suggest_float('min_weight_fraction_leaf',0.1, 0.5)
        max_features= trial.suggest_categorical("max_features", ["auto","log2", "sqrt", None])
        max_leaf_nodes= trial.suggest_int('max_leaf_nodes',10, 90)
        model = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,max_leaf_nodes=max_leaf_nodes,splitter=splitter )
    if model_type == 'mlpregressor':
        hidden_layers = trial.suggest_categorical("hidden_layer_sizes",[(50,100),(100,100),(50,75,100),(25,50,75,100)])
        activation = trial.suggest_categorical("activation", ["relu", "identity"])
        solver = trial.suggest_categorical("solver", ["sgd", "adam"])
        learning_rate = trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive'])
        learn = trial.suggest_float("learn", 0.001, 0.01)
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            learning_rate=learning_rate,
            learning_rate_init=learn,
            # early_stopping=True
        )
    #we can use pruning in case we use neptune
    #if trial.should_prune():
            #raise optuna.TrialPruned()

    return model
def objective(trial):
    model = create_model(trial)
    model.fit(X_train, y_train)
    return mean_squared_error(y_test, model.predict(X_test))
study = optuna.create_study(direction='minimize', study_name='supreme superconductor')
study.optimize(objective, n_trials=300)
best_model = create_model(study.best_trial)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print('r_2: ', r2_score(y_test, y_pred))
#here we have the best parameters
bestparam={'max_depth': 9,
'learning_rate': 0.16839770791382408,
 'n_estimators': 995,
 'min_child_weight': 6,
 'gamma': 0.010198379479515997,
 'subsample': 0.9178315622335436,
 'colsample_bytree': 0.6928868700170635,
 'reg_alpha': 0.27061387473469545,
 'reg_lambda': 0.8472439159558633,
 'random_state': 961}
#find best parameters for CatRegressor
def objectivecat(trial):

    param = {
        'iterations':trial.suggest_int("iterations", 1, 250),
        'od_wait':trial.suggest_int('od_wait', 50, 230),
        'learning_rate' : trial.suggest_float('learning_rate',0.01, 1),
        'reg_lambda': trial.suggest_categorical('reg_lambda',[1e-5,1e-4,1e-3,1e-2,1e-1,1,2,3,4,5]),
        'subsample': trial.suggest_float('subsample',0,1),
        'random_strength': trial.suggest_int('random_strength',10,50),
        'depth': trial.suggest_int('depth',1, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'bagging_temperature' :trial.suggest_categorical('bagging_temperature', [0.01, 0.1,1,10,100]),
        'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.4, 1.0),
    }


    model =CatBoostRegressor(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return mean_squared_error(y_test, y_pred)
studycat = optuna.create_study(direction='minimize', study_name='CatBoostregressor')
studycat.optimize(objectivecat, n_trials=17,n_jobs=-1)
#plot to see how good the model is
fig1=optuna.visualization.plot_optimization_history(studycat, target_name="MSE of Catboost")
fig2=optuna.visualization.plot_param_importances(studycat, target_name="MSE of Catboost")
fig1.show()
fig2.show()
#best parameters catboost
#{'iterations': 247,
# 'od_wait': 189,
# 'learning_rate': 0.1178128469313608,
# 'reg_lambda': 0.01,
# 'subsample': 0.4179973926537863,
# 'random_strength': 50,
# 'depth': 14,
# 'min_data_in_leaf': 19,
# 'leaf_estimation_iterations': 8,
# 'bagging_temperature': 0.01,
#'colsample_bylevel': 0.449236500979015}
#find best parameters for Decisiontree
def objectivedec(trial):

    param = {
        'splitter': trial.suggest_categorical('splitter',["best", "random"]),
        'max_depth' : trial.suggest_int('max_depth', 1, X_train.shape[1]),#numero di colonne di x_train
        'min_samples_leaf':trial.suggest_int('min_samples_leaf',2,20),
        'min_samples_split' :trial.suggest_int('min_samples_split', 2, 20),

        'max_features': trial.suggest_categorical('max_features', ['auto','log2', 'sqrt', None]),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes',10, 90)
    }


    model = DecisionTreeRegressor(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)
studydec = optuna.create_study(direction='minimize', study_name='Decisionregressor')
studydec.optimize(objectivedec, n_trials=40,n_jobs=-1)
#plot to see how good the model is
fig3=optuna.visualization.plot_optimization_history(studydec, target_name="MSE of Decisiontree")
fig4=optuna.visualization.plot_param_importances(studydec, target_name="MSE of Decisiontree")
fig3.show()
fig4.show()
#best parameters for decisiontree
#{'splitter':best,'max_depth':12,'min_samples_leaf':16,'min_samples_split':2,'auto':auto,'max_leaf_nodes':58}
tf.random.set_seed(42)
input_shape = X_train.shape[1:]


def build_model(n_hidden, n_neurons):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="selu", kernel_initializer="lecun_normal",
                        kernel_regularizer=keras.regularizers.l2(0.01))),
        keras.layers.BatchNormalization()
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss="mse")
    return model
#results of 3 layer and 52 neurons so...

#number of neurons for layers 31 426 230
param_distribs = {
 "n_hidden": np.arange(1, 4),
 "n_neurons": np.arange(1, 500),

}
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
 validation_data=(X_test, y_test),
 callbacks=[keras.callbacks.EarlyStopping(patience=10)])
print(rnd_search_cv.best_params_)
#{'n_neurons': 28, 'n_hidden': 2}
def Regressor():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    model.add(keras.layers.Dense(28, activation="selu", kernel_initializer="lecun_normal",
                             kernel_regularizer=keras.regularizers.l2(0.01))),
    keras.layers.BatchNormalization(),
    model.add(keras.layers.Dense(28, activation="selu", kernel_initializer="lecun_normal",
                        kernel_regularizer=keras.regularizers.l2(0.01))),
    keras.layers.BatchNormalization()
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss="mse")
    return model


def objectivemlp(trial):
    hidden_layers = trial.suggest_categorical("hidden_layer_sizes", [(50,100),(100,100),(50,75,100),(25,50,75,100)])
    activation = trial.suggest_categorical("activation", ["relu", "identity"])
    solver = trial.suggest_categorical("solver", ["sgd", "adam"])
    learning_rate = trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive'])
    learning_rate_init = trial.suggest_float("learning_rate_init", 0.001, 0.01)

    ## Create Model
    model = MLPRegressor(
                            hidden_layer_sizes=hidden_layers,
                            activation=activation,
                            solver=solver,
                            learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init,

                            )

    model.fit(X_train,y_train)

    mse = mean_squared_error(y_test, model.predict(X_test))
    return mse
studymlp = optuna.create_study(study_name="MLPRegressor")
studymlp.optimize(objectivemlp, n_trials=40,n_jobs=-1)
#plot to see how good the model is
fig5=optuna.visualization.plot_optimization_history(studymlp, target_name="MSE of MLP")
#plot feature importanti
fig6=optuna.visualization.plot_param_importances(studymlp, target_name="MSE of MLP")
#best parameters for mlp
fig5.show()
fig6.show()

#let's find optimal parameters for lgbmregressor and apply also pruning!!
def objectivelgbm(trial):

    param = {
        'metric':'rmse',
        'random_state': 48,
        'n_estimators':  trial.suggest_int('n_estimators', 2, 1000),
        'reg_alpha': trial.suggest_categorical('reg_alpha', [1e-3,1e-2,1e-1,1,2,3,4, 10.0]),
        'reg_lambda': trial.suggest_categorical('reg_lambda',[ 1e-3,1e-2,1e-1,1,2,3,4, 10.0]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10, 20, 100]),
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100)
    }
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial,"rmse",valid_name='valid_0')
    model = LGBMRegressor(**param)

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],early_stopping_rounds=10,
                      callbacks=[pruning_callback], verbose=True)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    return mse
studylgb = optuna.create_study(study_name="LGBMRegressor")
studylgb.optimize(objectivelgbm, n_trials=50,n_jobs=-1)
#plot to see how good the model is
fig7=optuna.visualization.plot_optimization_history(studylgb, target_name="MSE of LGBM")
#plot feature importanti
fig8=optuna.visualization.plot_param_importances(studylgb, target_name="MSE of LGBM")
#best parameters for mlp
fig7.show()
fig8.show()
#best parameters
#{'n_estimators':458,'reg_alpha':0.01,'reg_lambda':0.001,'colsample_bytree':0.6,'subsample':0.4,'learning_rate':0.02,'max_depth':100,'num_leaves':656,'min_child_samples':2,'min_data_per_groups':36}
#{'hidden_layer_sizes':(50,100),'activation':relu,'solver':adam,'learning_rate':adaptive,'learning_rate_init':0.00339504}
#NOTA BENE
#no need for pruning in random forest because,Roughly speaking, some of the potential over-fitting that might happen in a single tree (which is a reason you do pruning generally) is mitigated by two things in a Random Forest:
#The fact that the samples used to train the individual trees are "bootstrapped".
#The fact that you have a multitude of random trees using random features and thus the individual trees are strong but not so correlated with each other.

def objectiverandom(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
    }

    model = RandomForestRegressor(**params,oob_score=True)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse
studyrandom = optuna.create_study(study_name="RandomForest")
studyrandom.optimize(objectiverandom, n_trials=40,n_jobs=-1)
#plot to see how good the model is
fig9=optuna.visualization.plot_optimization_history(studyrandom, target_name="MSE of LGBM")
#plot feature importanti
fig10=optuna.visualization.plot_param_importances(studyrandom, target_name="MSE of LGBM")
#best parameters for mlp
fig9.show()
fig10.show()
#{'n_estimators': 935, 'max_depth': 18, 'min_samples_split': 4, 'min_samples_leaf': 2}

def objectiveextra(trial):
    n_estimators = trial.suggest_categorical('n_estimators', [25, 50, 75, 100, 125, 150, 175, 200, 225, 250])
    max_features = trial.suggest_float('max_features', 0.15, 1.0)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 14)
    max_samples = trial.suggest_float('max_samples', 0.6, 0.99)
    bootstrap=True
    model = ExtraTreesRegressor(n_estimators=n_estimators,
                                          max_features=max_features, min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf, max_samples=max_samples,
                                          bootstrap=bootstrap,oob_score=True, n_jobs=-1, verbose=0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse
studyextra = optuna.create_study(study_name="Extra")
studyextra.optimize(objectiveextra, n_trials=40,n_jobs=-1)
#plot to see how good the model is
fig11=optuna.visualization.plot_optimization_history(studyextra, target_name="MSE of LGBM")
#plot feature importanti
fig12=optuna.visualization.plot_param_importances(studyextra, target_name="MSE of LGBM")
#best parameters for mlp
fig11.show()
fig12.show()
extratree={'n_estimators':250,'max_features':0.9744,'min_samples_split':12,'min_samples_leaf':1,'max_samples':0.96168}
#last step we will make a Votingregressor
dec=DecisionTreeRegressor(**studydec.best_params)#r_2:  0.8570182333866809#mse: 0.14261704726333946
cat= CatBoostRegressor(**studycat.best_params)#r_2:  0.9314027430181999#mse: 0.06842227839838086
xgb=XGBRegressor(**bestparam)#mse: 0.06816613130492961#r_2:  0.9316595451651893
mlp=MLPRegressor(**studymlp.best_params)#r_2:  0.9022381299254715#mse: 0.09751249809829053
ela=ElasticNet()
reg=Regressor()#r_2:  0.7384143338009774#mse: 0.2609184108111514
extra=ExtraTreesRegressor(**studyextra.best_params,bootstrap=True)#r_2:  0.934328996607011#mse: 0.06550348912709829
lgbm = LGBMRegressor(**studylgb.best_params)#r_2:  0.9348643708444293#mse: 0.06496948052775035
rf = RandomForestRegressor(**studyrandom.best_params)#r_2:  0.9295414215038849#mse: 0.07027885203477487

#i have an r^2 score of 92% and as most important parameters in order of importance :Cu,O,mean_Valence,Ba,mean_thermalConductivity,mean_FusionHeat,Ca,mean_electronaffinity

stack = StackingCVRegressor(regressors=[cat, dec,xgb,lgbm,rf,extra,reg,mlp], cv=10,
                            meta_regressor=extra,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=1,n_jobs=-1)
stack.fit(X_train, y_train)
#since there is a bug in xgboost we will have to change the names of the test columns in this way
#X_test.columns=np.arange(X_test.shape[1])
#X_test=X_test.add_prefix('f')
y_predstack=stack.predict(X_test)
print('r_2: ', r2_score(y_test, y_predstack))#r_2:  0.94
print('mse:',mean_squared_error(y_test,y_predstack))#mse: 0.0605361436359339
#results from the stacking take us into a 0.5 improvement in r2 score and into a less overfit with a decrease of 0.8 in mse