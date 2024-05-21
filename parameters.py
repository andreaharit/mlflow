# Models tested
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from hyperopt import hp

parameters = { 
    "AdaBoost": {
        "algo" : AdaBoostClassifier(algorithm ='SAMME'),
        "space" : {
            "model__n_estimators": hp.choice("model__n_estimators", [50, 100, 200]),
            "model__learning_rate": hp.choice("model__learning_rate", [0.01, 0.1, 1.0]),
            }
        },           
    "GradientBoosting": {
        "algo": GradientBoostingClassifier(), 
        "space" : 
        {
            "model__n_estimators": hp.choice("model__n_estimators", [100, 200, 300]),
            "model__learning_rate": hp.choice("model__learning_rate", [0.01, 0.1, 0.2]),
            "model__max_depth": hp.choice("model__max_depth", [3, 5, 7])
        }
    }
}

"""  
    # Extra parameters for other models
    # I am not testing them because it takes too long and it's not the scope of the exercise     
    "DecisionTreeClassifier": {            
        "algo": DecisionTreeClassifier(),
        "space" : {
            'model__max_depth': hp.choice('model__max_depth', [None, 10, 20, 30]),
            'model__min_samples_split': hp.choice('model__min_samples_split', [2, 10, 20]),
            'model__min_samples_leaf': hp.choice('model__min_samples_leaf', [1, 5, 10]),
        }
    },    
    "KNeigborsClassifier": {
        "algo": KNeighborsClassifier(),
        "space" : {'model__n_neighbors': hp.choice('model__n_neighbors', [3, 5, 10]),
        'model__weights': hp.choice('model__weights', ['uniform', 'distance'])
        }
    },
    "RandomForestClassifier": {
        "algo": RandomForestClassifier(),
        "space" : {'model__n_estimators': hp.choice('model__n_estimators', [100, 200, 300]),
        'model__max_depth': hp.choice('model__max_depth',[None,10,20]),
        'model__min_samples_split': hp.choice('model__min_samples_split', [2, 5, 10])
        }
    },
    "LogisticRegression": {
        "algo": LogisticRegression(),
        "space" : {
            'model__C': hp.choice('model__C', [0.01, 0.1, 1, 10]),
            'model__penalty': hp.choice('model__penalty', ['l2'])
        }
    }
}
"""

