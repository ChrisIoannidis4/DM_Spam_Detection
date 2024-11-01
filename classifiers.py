import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold 
import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix



def create_descriptors(x_train, method = "bow", ngram_type = 'unigram', vocab_size=False, scale=False):
    '''
    Creates descriptors/feature vectors from the text which has to be in the column "review_text" of the df.
    inputs:
    df: pandas df -- has to have a column "review_text", based on which the model 'trains' (e.g. gathers histogram bins)
                     and this is the column fro which the descriptors will be extracted
    method: str -- "bow" or "tf-idf" specify which of the two methods to use to create the feature vectors
    ngram_type: str -- 'unigram' or 'bigram' specify what of histogram bins will be created
    vocab_size: bool, int -- if False, all histogram bins will be taken into account. if int, only the n=vocab_size most frequent terms
                             will be taken into account
    
    returns:
    feature_vectors: df, a column of the created descriptors (row index corresponds to text in initial df) 
    feature_model: model, the trained model            
    '''
    
    if ngram_type == 'unigram':
        ngram_range = (1, 1)
    elif ngram_type == 'bigram':
        ngram_range = (1, 2)
    
    if method == 'bow':
        model = CountVectorizer(ngram_range=ngram_range, max_features=vocab_size if vocab_size else None)
    elif method == 'tf-idf':
        model = TfidfVectorizer(ngram_range=ngram_range, max_features=vocab_size if vocab_size else None) 

    feature_vectors = model.fit_transform(x_train)
    print(feature_vectors.shape)
    if scale:
        # Ensure feature_vectors is in the correct format for scaling
        if isinstance(feature_vectors, csr_matrix):
            # Use StandardScaler directly on sparse matrix
            scaler = StandardScaler(with_mean=False)  # Use with_mean=False for sparse input
            feature_vectors = scaler.fit_transform(feature_vectors)  # Scale the feature vectors
        else:
            # Convert to dense array for scaling if necessary
            feature_vectors = feature_vectors.toarray()  # Convert to dense array for scaling
            scaler = StandardScaler()
            feature_vectors = scaler.fit_transform(feature_vectors)

    # print(f"rows x features: {feature_vectors.shape}")  
    # print(f"examples:\n {[model.get_feature_names_out()[i] for i in range(1, 200, 20)]}")
    return feature_vectors, model



def logreg_grid_search(feature_vectors, labels, name=None):
    '''
    hyperparameter tuning for logistic regression, for parameter C
    returns dict
    e.g. {'C' = 3000}
    '''
    log_reg = LogisticRegression(solver='saga' ,penalty='l1', max_iter=1000)
    param_grid = {'C': [1, 10, 500, 1000, 2000, 3000]}
    # log_reg = SGDClassifier(loss='log_loss', penalty = 'l1', max_iter=5000)
    # param_grid = {'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.1, 1]}
    grid_search = utils.perform_grid_search(log_reg, param_grid, feature_vectors, labels)
    utils.write_grid_search_results(f"grid_search_results/log_reg_results_{name}.txt", grid_search)
    return grid_search.best_params_


def tree_grid_search(feature_vectors, labels, name=None):
    '''
    hyperparameter tuning for tree classifier, for pruning parameter a
    returns dict
    e.g. {'ccp_alpha' = 0001}
    '''
    decision_tree_clf= DecisionTreeClassifier(random_state=42)
    param_grid = {'ccp_alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
    grid_search = utils.perform_grid_search(decision_tree_clf, param_grid, feature_vectors, labels)
    utils.write_grid_search_results(f"grid_search_results/tree_clf_results_{name}.txt", grid_search)
    return grid_search.best_params_


def rf_search(feature_vectors, labels, name=None):
    '''
    we dont do kfold cross validation for random forest fine tuning, we use oob score

    '''
    param_grid = {'n_estimators': [50, 100, 200, 400],
        'max_features': ['sqrt', 'log2', 0.2, 500, 1000, None]}
    best_score, best_params, best_model = -1, None, None

    with open(f"grid_search_results/rf_results_{name}", "w") as f:
        for n_estimators in param_grid['n_estimators']:
            for max_features in param_grid['max_features']: 
                rf_clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    oob_score=True,   
                    random_state=42
                )
                rf_clf.fit(feature_vectors, labels)
                oob_score = rf_clf.oob_score_
                if oob_score > best_score:
                    best_score = oob_score
                    best_params = {'n_estimators': n_estimators, 'max_features': max_features}
                    best_model = rf_clf

                f.write(f"OOB Score for n_estimators={n_estimators}, max_features={max_features}: {oob_score:.4f}\n")

        f.write(f"Best Params: {best_params}, Best OOB Score: {best_score:.4f}")
    return best_params



def kfold_no_features(df, model, method, n_gram, vocab_sizes=[500, 1000, 2000, None]):
    '''
    model: model object, it can test for any algorithm with cross validation the ideal vocabulary size/feature selection
    it tracks by f1 score.
    it is created mainly for the naive_bayes configuration option mentioned in the assignment
    '''
    best_f1_score = -1
    best_vocab_size = None
    all_results=[]
    for vocab_size in vocab_sizes:
        #for evey vocabulary size option, we initiate a kfold 
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        total_accuracy, total_precision, total_recall, total_f1_score = 0, 0, 0, 0

        # keep the feature extractor aka vectorizer that has kept the right bins from the dataset
        _, feature_extractor = create_descriptors(df["review_text"], method=method, ngram_type=n_gram, vocab_size=vocab_size)
        labels = df["deceptive_flag"].values.astype(int)

        for train_index, test_index in kf.split(df):
            #perform cross validation
            x_train, x_test = feature_extractor.transform(df["review_text"].iloc[train_index]), \
                            feature_extractor.transform(df["review_text"].iloc[test_index])
            y_train, y_test = labels[train_index], labels[test_index]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy, precision, recall, f1_score = utils.evaluate_model(y_test, y_pred)
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score


        avg_f1 = total_f1_score / kf.n_splits
        all_results.append({
            'vocab_size': vocab_size,
            'accuracy': total_accuracy/kf.n_splits,
            'precision': total_precision/kf.n_splits,
            'recall': total_recall / kf.n_splits,
            'f1_score': avg_f1
        })
        # Track the best performing vocab size
        if avg_f1 > best_f1_score:
            best_f1_score = avg_f1
            best_vocab_size = vocab_size
    return best_vocab_size, all_results


def naive_bayes_search(df, method, n_gram, vocab_sizes= [500, 1000, 2000, None], name=None):
    best_vocab_size, all_scores = kfold_no_features(df, MultinomialNB(), method, n_gram, vocab_sizes)
    utils.write_bayes_results(best_vocab_size=best_vocab_size, all_scores=all_scores, name=name )
    print(best_vocab_size)
    return best_vocab_size


def train_model(x_train, y_train, model_type, hyperparameters=None):  
    if hyperparameters is None:
        hyperparameters = {} 
    if model_type == 'logreg': 
        model = LogisticRegression(penalty='l1', solver='liblinear', **hyperparameters)
    elif model_type == 'tree': 
        model = DecisionTreeClassifier(**hyperparameters)
    elif model_type == 'rf':  
        model = RandomForestClassifier(**hyperparameters)
    elif model_type == 'bayes': 
        model = MultinomialNB()  
    model.fit(x_train, y_train)
    return model


def stat_test_mcnemar(pred_1, pred_2, test):

    #each vector has 1 in correct predictions and 0 in wrong ones 
    correct_1 = np.where(pred_1 == test, 1, 0)  
    correct_2 = np.where(pred_2 == test, 1, 0) 
    #calculate cont table terms and define cont table
    a = np.sum((correct_1 == 1) & (correct_2 == 1))  
    b = np.sum((correct_1 == 1) & (correct_2 == 0))  
    c = np.sum((correct_1 == 0) & (correct_2 == 1))  
    d = np.sum((correct_1 == 0) & (correct_2 == 0))   
    contingency_table = np.array([[a, b], [c, d]])
    result = mcnemar(contingency_table, exact=False, correction=True)

    return result



def main():
    
    #DESCRIPTOR HYPERPARAMS
    METHODS=['tf-idf', 'bow']
    NGRAM_TYPES=['unigram', 'bigram']
    VOCAB_SIZE = None

    reviews = pd.read_csv("reviews_data.csv")
    reviews['review_text'] = reviews['review_text'].apply(utils.preprocess_text)
    labels = reviews["deceptive_flag"].values.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(reviews['review_text'], labels, test_size=0.2)



    # for METHOD in METHODS:
    #     for NGRAM_TYPE in NGRAM_TYPES:
    #         print(METHOD, NGRAM_TYPE)
    #         feature_vectors_train, transformer = create_descriptors(x_train, method=METHOD, ngram_type=NGRAM_TYPE, vocab_size=VOCAB_SIZE, scale=False)  
    #         feature_vectors_test = transformer.transform(x_test)
    #         # grid searches, toggle on the one you want to perform
    #         best_vocab_size = naive_bayes_search(
    #                             reviews.loc[:len(x_train)], METHOD, NGRAM_TYPE, 
    #                             vocab_sizes= [500, 1000, 2000, 3000, None], name=f"{NGRAM_TYPE}_{METHOD}"
    #                             ) #have to change: take as input only x_train 
    #                             #   and not all dataframe
    #         logreg_best_params = logreg_grid_search(feature_vectors_train, y_train, name=f"{NGRAM_TYPE}_{METHOD}")
    #         print(logreg_best_params)
    #         tree_best_params = tree_grid_search(feature_vectors_train, y_train, name=f"{NGRAM_TYPE}_{METHOD}")
    #         rf_best_params = rf_search(feature_vectors_train, y_train, name=f"{NGRAM_TYPE}_{METHOD}")

    feature_vectors_train, model = create_descriptors(x_train,  'bow', 'bigram', None) #only for naive bayes
    feature_vectors_test = model.transform(x_test) #only for naive bayes too

    preds_bi = []

    for model_type, hyperparams in zip(['logreg', 'tree', 'rf'], [{'C': 3000}, {"ccp_alpha":0.01}, {"n_estimators": 400, "max_features": 'sqrt'}]):
        trained_model = train_model(feature_vectors_train, y_train, model_type, hyperparameters=hyperparams)
        y_pred = trained_model.predict(feature_vectors_test)
        accuracy, precision, recall, f1_score = utils.evaluate_model(y_test, y_pred)
        print(f'best {model_type}:: accuracy:{accuracy}, precision:{precision}, recall:{recall}, f1_score:{f1_score}')
        preds_bi.append(y_pred)

    feature_vectors_train, model = create_descriptors(x_train,  'bow', 'bigram', 2000) #only for naive bayes
    feature_vectors_test = model.transform(x_test) #only for naive bayes too
    bayes=train_model(feature_vectors_train, y_train, 'bayes', hyperparameters=None)
    y_pred = bayes.predict(feature_vectors_test)
    accuracy, precision, recall, f1_score = utils.evaluate_model(y_test, y_pred)
    print(f'best bayes:: accuracy:{accuracy}, precision:{precision}, recall:{recall}, f1_score:{f1_score}')
    preds_bi.append(y_pred)


#     # #Train full model. if you we have skipped the grid search by this point, hyperparameters should be specified as dict
#     # # e.g. {"C" : 3000}
#     # model=train_model(feature_vectors_train, y_train, "bayes", hyperparameters=None)
#     # y_pred = model.predict(feature_vectors_test)
#     # accuracy, precision, recall, f1_score = utils.evaluate_model(y_test, y_pred)
#     # print(accuracy, precision, recall, f1_score)

#     # #put the correct folder name
#     # with open("model_eval/bayes_tfidf_none.txt", "w") as f:
#     #     f.write(f"accuracy: {accuracy:.4f}\nprecision: {precision:.4f}\nrecall{recall:.4f}\nf1_score: {f1_score:.4f} ")


    logpred_bi, treepred_bi, rfpred_bi, bayespred_bi = preds_bi

    print("bigram bayes, logreg: ", stat_test_mcnemar(bayespred_bi, logpred_bi, y_test))
    print("bigram rf, log: ", stat_test_mcnemar(rfpred_bi, logpred_bi, y_test))
    print("bigram rf, bayes: ", stat_test_mcnemar(rfpred_bi, bayespred_bi, y_test))
    print("bigram tree, logreg: ", stat_test_mcnemar(treepred_bi, logpred_bi, y_test))


#     #################################################

    feature_vectors_train, model = create_descriptors(x_train,  'bow', 'unigram', None) #only for naive bayes
    feature_vectors_test = model.transform(x_test) #only for naive bayes too

    preds_uni = []

    for model_type, hyperparams in zip(['logreg', 'tree', 'rf'], [{'C': 3000}, {"ccp_alpha":0.01}, {"n_estimators": 400, "max_features": 'sqrt'}]):
        trained_model = train_model(feature_vectors_train, y_train, model_type, hyperparameters=hyperparams)
        y_pred = trained_model.predict(feature_vectors_test)
        accuracy, precision, recall, f1_score = utils.evaluate_model(y_test, y_pred)
        print(f'best {model_type}:: accuracy:{accuracy}, precision:{precision}, recall:{recall}, f1_score:{f1_score}')
        preds_uni.append(y_pred)

    feature_vectors_train, model = create_descriptors(x_train,  'bow', 'bigram', 1000) #only for naive bayes
    feature_vectors_test = model.transform(x_test) #only for naive bayes too
    bayes=train_model(feature_vectors_train, y_train, 'bayes', hyperparameters=None)
    y_pred = bayes.predict(feature_vectors_test)
    accuracy, precision, recall, f1_score = utils.evaluate_model(y_test, y_pred)
    print(f'best bayes:: accuracy:{accuracy}, precision:{precision}, recall:{recall}, f1_score:{f1_score}')
    preds_uni.append(y_pred)

    logpred_uni, treepred_uni, rfpred_uni, bayespred_uni = preds_uni

    print("unigram bayes, logreg: ", stat_test_mcnemar(bayespred_uni, logpred_uni, y_test))
    print("unigram rf, log: ", stat_test_mcnemar(rfpred_uni, logpred_uni, y_test))
    print("unigram rf, bayes: ", stat_test_mcnemar(rfpred_uni, bayespred_uni, y_test))
    print("unigram tree, logreg: ", stat_test_mcnemar(treepred_uni, logpred_uni, y_test))




    print("unigram vs bi: bayes ", stat_test_mcnemar(bayespred_uni, bayespred_bi, y_test))
    print("unigram vs bi: log ", stat_test_mcnemar(logpred_bi, logpred_uni, y_test))
    print("unigram vs bi: rf ", stat_test_mcnemar(rfpred_uni, rfpred_bi, y_test))
    print("unigram vs bi: tree: ", stat_test_mcnemar(treepred_uni, treepred_bi, y_test))







if __name__ == "__main__":
    main()

