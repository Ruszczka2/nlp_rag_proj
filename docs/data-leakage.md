# Wyciek danych (data leakage)

Wyciek danych występuje, gdy informacje ze zbioru testowego lub z przyszłych danych wpływają na trening modelu, co daje zbyt optymistyczne wyniki ewaluacji. Klasyczny przykład: dopasowanie skalera lub wektoryzatora na całym zbiorze przed podziałem na train i test. Poprawne podejście: dopasowuj preprocessing i modele tylko na danych treningowych, a transformację i ewaluację wykonuj na danych odłożonych (held-out).
