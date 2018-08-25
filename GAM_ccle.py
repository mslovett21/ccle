import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr
import random
from pygam import LinearGAM






def find_parameters_evaluation(index_set,gene_expression, cell_count_aa):
    prediction=[]
    actual_value=[]
    n_splines_all=[]
    lam_all=[]

    # THIS IS OUTER LOOP: for VALIDATION/TESTING 
    #train n models and evaluate their average performance  
    gene_indexes=index_set
    y=cell_count_aa
    X=gene_expression[gene_expression.columns[gene_indexes]]
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    gam = LinearGAM()
    gam=gam.gridsearch(X, y,n_splines=np.arange(10,50),lam=[0.4,0.5,0.6,0.7,0.8])

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # THIS IS INNER LOOP: for TRAINING/VALIDATION
        #train model with given optimized parameters
        regr = gam.fit(X_train,y_train)
        #make a prediction on OUTER LOOP test set
        prediction_val=regr.predict(X_test)[0]
        # store predictions and actual values
        prediction.append(prediction_val)
        actual_value.append(y_test[0])
        # add optimal parameter values to arrays
        n_splines_all.append(regr.n_splines)
        lam_all.append(regr.lam)
        print(test_index)
        print(str(prediction_val)," ",str(y_test[0]))
    #calculate spearman correlation over all of the models
    rho, pval     =  spearmanr(actual_value,prediction)
    lams          =  np.array(lam_all)
    lams_mean     =  lams.mean()
    n_splines_all =  np.array(n_splines_all)
    n_splines_mean =  n_splines_all.mean()
    return lams_mean,n_splines_mean,rho,pval


def main():
    
    #variables for storing best performace

    best_rho=0
    best_pval=0
    best_feature_set=[]
    best_feature_set_size=0
    
    #load data
    gene_expression=pd.read_csv("lev_ccle_exp_dat.csv", index_col=False)
    targets=pd.read_csv("lev_ccle_target_dat.csv", index_col=False)
    gene_expression=gene_expression.drop(columns=['Unnamed: 0'])
    cell_count_aa=targets[targets.columns[-1]]
    cell_count_aa=np.array(cell_count_aa.astype(float))
    
    #standardize the data
 #   standarize = lambda x: (x-x.mean()) / x.std()
 #   gene_expression=gene_expression.pipe(standarize)
    
    results_file = open("Results_GAM_ccle.txt", "a+")
    results_file.write("set_size,lam,n_splines,spearmanr,pval")
    results_file.write('\n')
    # open file with with feature sets 

    with open("ccle_rest.txt", "r") as infile:
        for line in infile:
            line = line.strip()
            splitted = line.split(",")
            set_size  =int(splitted[0])
            fea_set = np.array(splitted[1:set_size+1])
            fea_set= fea_set.astype(int)
            lam,n_splines,rho,pval = find_parameters_evaluation(fea_set,gene_expression,cell_count_aa)
            if(rho > best_rho):
                best_rho  = rho
                best_pval = pval
                best_lam  = lam
                best_n_splines = n_splines
                best_feature_set = fea_set
                best_feature_set_size = set_size              
            results_file.write(str(set_size)+","+str(lam)+","+str(n_splines)+ "," + str(rho)+","+str(pval))
            results_file.write('\n')
            print(str(set_size),",",str(lam),",",str(n_splines),",",str(rho),",",str(pval))
    best_gene_names=gene_expression.columns[best_feature_set]
    print("Best result set_size: ",str(best_feature_set_size)," lam: ",str(best_lam)," n_splines: ",
          str(best_n_splines), " spearmanr: ",str(best_rho)," pval: ",str(best_pval),"\n")
    print("Best feature set: \n")
    print(str(best_feature_set))
    print(best_gene_names)



if __name__ == "__main__":
    main()        

