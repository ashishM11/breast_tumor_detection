from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "DT_GINI": DecisionTreeClassifier(criterion="gini"),
    "DT_ENTROPY": DecisionTreeClassifier(criterion="entropy"),
    "LOG_REG": LogisticRegression(n_jobs=-1, class_weight="balanced")
    "RF": RandomForestClassifier(n_jobs=-1)
}
