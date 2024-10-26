import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold 
import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def create_descriptors(df, method = "bow", ngram_type = 'unigram', vocab_size=False):
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

    feature_vectors = model.fit_transform(df['review_text'])

    # print(f"rows x features: {feature_vectors.shape}")  
    # print(f"examples:\n {[model.get_feature_names_out()[i] for i in range(1, 200, 20)]}")
    return feature_vectors, model



def logreg_grid_search(feature_vectors, labels):
    '''
    hyperparameter tuning for logistic regression, for parameter C
    returns dict
    e.g. {'C' = 3000}
    '''
    log_reg = LogisticRegression(solver = 'liblinear', penalty='l1')
    param_grid = {'C': [500, 1000, 2000, 3000]}
    grid_search = utils.perform_grid_search(log_reg, param_grid, feature_vectors, labels)
    utils.write_grid_search_results("grid_search_results/log_reg_results.txt", grid_search)
    return grid_search.best_params_


def tree_grid_search(feature_vectors, labels):
    '''
    hyperparameter tuning for tree classifier, for pruning parameter a
    returns dict
    e.g. {'ccp_alpha' = 0001}
    '''
    decision_tree_clf= DecisionTreeClassifier(random_state=42)
    param_grid = {'ccp_alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
    grid_search = utils.perform_grid_search(decision_tree_clf, param_grid, feature_vectors, labels)
    utils.write_grid_search_results("grid_search_results/tree_clf_results.txt", grid_search)
    return grid_search.best_params_


def rf_search(feature_vectors, labels):
    '''
    we dont do kfold cross validation for random forest fine tuning, we use oob score

    '''
    param_grid = {'n_estimators': [50, 100, 200, 400],
        'max_features': ['sqrt', 'log2', 0.2, 500, 1000, None]}
    best_score, best_params, best_model = -1, None, None

    with open("grid_search_results/rf_results", "w") as f:
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
        _, feature_extractor = create_descriptors(df, method=method, ngram_type=n_gram, vocab_size=vocab_size)
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


def naive_bayes_search(df, method, n_gram, vocab_sizes= [500, 1000, 2000, None]):
    best_vocab_size, all_scores = kfold_no_features(df, MultinomialNB(), method, n_gram, vocab_sizes)
    utils.write_bayes_results(best_vocab_size=best_vocab_size, all_scores=all_scores)
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



def main():
    
    #DESCRIPTOR HYPERPARAMS
    METHOD="tf-idf"
    NGRAM_TYPE='unigram'
    VOCAB_SIZE = None

    reviews = pd.read_csv("reviews_data.csv")
    reviews['review_text'] = reviews['review_text'].apply(utils.preprocess_text)
    feature_vectors, _ = create_descriptors(reviews, method=METHOD, ngram_type=NGRAM_TYPE, vocab_size=VOCAB_SIZE)  
    labels = reviews["deceptive_flag"].values.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2)
    # grid searches, toggle on the one you want to perform
    # best_vocab_size = naive_bayes_search(
    #                     reviews, METHOD, NGRAM_TYPE, 
    #                     vocab_sizes= [500, 1000, 2000, None]
    #                     ) #have to change: take as input only x_train 
                          #and not all dataframe
    # logreg_best_params = logreg_grid_search(x_train, labels)
    # tree_best_params = tree_grid_search(x_train, labels)
    # rf_best_params = rf_search(x_train, labels)
    # feature_vectors, _ = create_descriptors(reviews, METHOD, NGRAM_TYPE, best_vocab_size) #only for naive bayes


    #Train full model. if you we have skipped the grid search by this point, hyperparameters should be specified as dict
    # e.g. {"C" = 3000}
    model=train_model(x_train, y_train, "bayes", hyperparameters=None)
    y_pred = model.predict(x_test)
    accuracy, precision, recall, f1_score = utils.evaluate_model(y_test, y_pred)
    print(accuracy, precision, recall, f1_score)

    #put the correct folder name
    with open("model_eval/bayes_tfidf_none.txt", "w") as f:
        f.write(f"accuracy: {accuracy:.4f}\nprecision: {precision:.4f}\nrecall{recall:.4f}\nf1_score: {f1_score:.4f} ")


if __name__ == "__main__":
    main()

