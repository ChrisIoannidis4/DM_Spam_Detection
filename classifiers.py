import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
import utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


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
    else:
        raise ValueError("Choose 'unigram' or 'bigram'.")
    
    if method == 'bow':
        model = CountVectorizer(ngram_range=ngram_range, max_features=vocab_size if vocab_size else None)
    elif method == 'tf-idf':
        model = TfidfVectorizer(ngram_range=ngram_range, max_features=vocab_size if vocab_size else None)
    else:
        raise ValueError("Available options: 'bow' or 'tf-idf'.")

    feature_vectors = model.fit_transform(df['review_text'])

    # print(f"rows x features: {feature_vectors.shape}")  
    # print(f"examples:\n {[model.get_feature_names_out()[i] for i in range(1, 200, 20)]}")
    return feature_vectors, model



def logreg_grid_search(feature_vectors, labels):
    log_reg = LogisticRegression(solver = 'liblinear', penalty='l1')
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 500, 1000]}
    grid_search = utils.perform_grid_search(log_reg, param_grid, feature_vectors, labels)
    utils.write_grid_search_results("grid_search_results/log_reg_results.txt", grid_search)
    return grid_search.best_params_

def tree_grid_search(feature_vectors, labels):
    decision_tree_clf= DecisionTreeClassifier(random_state=42)
    param_grid = {'ccp_alpha': [0.0001, 0.001, 0.01, 0.1, 1]}
    grid_search = utils.perform_grid_search(decision_tree_clf, param_grid, feature_vectors, labels)
    utils.write_grid_search_results("grid_search_results/tree_clf_results.txt", grid_search)
    return grid_search.best_params_


def rf_search(feature_vectors, labels):
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
    return best_model, best_params



def kfold_no_features(df, model, method, n_gram, vocab_sizes=[500, 1000, 2000, None]):
    
    best_f1_score = -1
    best_vocab_size = None
    all_results=[]
    for vocab_size in vocab_sizes:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        total_accuracy, total_precision, total_recall, total_f1_score = 0, 0, 0, 0

        _, feature_extractor = create_descriptors(df, method=method, ngram_type=n_gram, vocab_size=vocab_size)
        labels = df["deceptive_flag"].values.astype(int)

        for train_index, test_index in kf.split(df):
            
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


# to be done
def train_model(x_train, y_train, model_type, hyperparameters=None):  
    if hyperparameters is None:
        hyperparameters = {} 
    if model_type == 'logreg': 
        model = LogisticRegression(**hyperparameters)
    elif model_type == 'tree': 
        model = DecisionTreeClassifier(**hyperparameters)
    elif model_type == 'rf':  
        model = RandomForestClassifier(**hyperparameters)
    elif model_type == 'bayes': 
        model = MultinomialNB()  
    model.fit(x_train, y_train)
    return model



def main():
    
    METHOD="tf-idf"
    NGRAM_TYPE='unigram'
    VOCAB_SIZE = None

    reviews = pd.read_csv("reviews_data.csv")
    reviews['review_text'] = reviews['review_text'].apply(utils.preprocess_text)
    feature_vectors, _ = create_descriptors(reviews, method=METHOD, ngram_type=NGRAM_TYPE, vocab_size=VOCAB_SIZE)  
    labels = reviews["deceptive_flag"].values.astype(int)


    # best_vocab_size = naive_bayes_search(reviews, METHOD, NGRAM_TYPE, vocab_sizes= [500, 1000, 2000, None])
    logreg_best_params = logreg_grid_search(feature_vectors, labels)
    tree_best_params = tree_grid_search(feature_vectors, labels)
    _, best_params = rf_search(feature_vectors, labels)



    model=train_model(feature_vectors,labels, "logreg", hyperparameters=logreg_best_params)
    y_pred = model.predict(feature_vectors)
    accuracy, precision, recall, f1_score = utils.evaluate_model(labels, y_pred)
    print(accuracy, precision, recall, f1_score)
    # with open("model_eval/logreg.txt", "w") as f:
    #         f.write(f"Model Type: Logistic Regression\n")
    #         f.write(f"Best Hyperparameters: {logreg_best_params}\n")
    #         f.write(f"Accuracy: {accuracy:.4f}\n")
    #         f.write(f"Precision: {precision:.4f}\n")
    #         f.write(f"Recall: {recall:.4f}\n")
    #         f.write(f"F1 Score: {f1_score:.4f}\n")



if __name__ == "__main__":
    main()
