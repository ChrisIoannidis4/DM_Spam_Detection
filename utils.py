import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from classifiers import create_descriptors
from sklearn.model_selection import KFold, GridSearchCV

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    #convert to lowercase
    text = text.lower()
    #remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    #remove short words
    # text = re.sub(r'\b\w{1,2}\b', '', text)
    #remove stopwords 
    text = ' '.join([word for word in text.split() if word not in stop_words])
    #lemmatize the text
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


def evaluate_model(y_true, y_pred): 
    return accuracy_score(y_true, y_pred), \
        precision_score(y_true, y_pred), \
        recall_score(y_true, y_pred), \
        f1_score(y_true, y_pred)


def perform_grid_search(model, param_grid, feature_vectors, labels):
    '''
    used for logistic regression and tree search
    inputs:
    model: model object, param_grid:dict
    returns: grid search and its results
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=42) 
    
    scoring = {'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score)} 
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=scoring, refit='f1')
    grid_search.fit(feature_vectors, labels)
    return grid_search


def write_grid_search_results(file_path, grid_search):
    with open(file_path, "w") as f:
        f.write(f"best params: {grid_search.best_params_}\n")
        f.write(f"best model: {grid_search.best_estimator_}\n") 
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            mean_scores = grid_search.cv_results_[f'mean_test_{metric}']
            std_scores = grid_search.cv_results_[f'std_test_{metric}']
            for params, mean_score, std_score in zip(grid_search.cv_results_['params'], mean_scores, std_scores):
                f.write(f"params: {params}, mean {metric} score: {mean_score:.4f}, std_dev: {std_score:.4f}\n")


def write_bayes_results(best_vocab_size, all_scores): 
    with open("grid_search_results/bayes_results.txt", "w") as f:
        for score in all_scores:
            f.write(f"vocab size: {score['vocab_size']}\n")
            f.write(f"  acc: {score['accuracy']:.4f}\n")
            f.write(f"  prec: {score['precision']:.4f}\n")
            f.write(f"  recall: {score['recall']:.4f}\n")
            f.write(f"  f1_score: {score['f1_score']:.4f}\n\n")
        f.write(f"best size: {best_vocab_size}\n")
