# Parametry vs hiperparametry

Parametry modelu uczą się z danych w trakcie treningu, np. wagi w modelu liniowym lub wektory nośne w SVM. Hiperparametry ustawia się przed treningiem i kontrolują proces uczenia, np. learning rate, siłę regularyzacji (C w SVM) lub `max_depth` w drzewie decyzyjnym. Hiperparametry zwykle stroi się walidacją krzyżową, np. za pomocą `GridSearchCV`.
