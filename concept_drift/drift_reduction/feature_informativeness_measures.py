from sklearn import feature_selection

from concept_drift.classifier.perform_classification import ClassifierFactory


def mutual_info(X, y):
    return feature_selection.mutual_info_classif(
        X,
        y,
        discrete_features=False,
        copy=True,
        n_neighbors=10
    )


def anova_f(X, y):
    result, _ = feature_selection.f_classif(X, y)
    return abs(result)


def random_forest(X, y):
    random_forest_clf = ClassifierFactory.make_classifier('random_forest')
    random_forest_clf.fit(X, y)
    return list(random_forest_clf.feature_importances_)


_NAME_TO_FUN = {
    'mutual_info': mutual_info,
    'anova_f': anova_f,
    'random_forest': random_forest
}


AVAILABLE_FUNCTIONS = _NAME_TO_FUN.keys()


def get_feature_informativeness_measure(fun_name):
    return _NAME_TO_FUN[fun_name]
