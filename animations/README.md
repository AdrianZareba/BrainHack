# Animacje kolonii mrówek

Eksperyment Posnera z trafną wskazówką, pokazany krok po kroku dla jednej kolonii 64 mrówek.

- `ants_experiment_valid_T3.*`: target pojawia się w kroku T3. Mrówki idą za wskazówką, a większość jest przy lewym okienku.
- `ants_experiment_valid_T10.*`: target pojawia się w kroku T10. Lewy zasób się kończy, a wskazówka gaśnie, więc mrówki przechodzą do drugiego okienka.

Na filmie widać:
- procent mrówek przy każdym okienku,
- feromon jako grubość szlaku,
- zasób jako prostokąt nad okienkiem (przerywana ramka = pełny, zielone = co zostało),
- wskazówkę jako czarną ramkę,
- target jako żółtą gwiazdkę.

## Jak powstały

Skrypt `scripts/make_ants_video.py` używa reguły kolonii z `parameter_analysis/scripts/ant_model.py`, z domyślnymi parametrami z `posner_ants.ipynb`. Z repo root:

```bash
pip install -r animations/scripts/requirements.txt
ANTS_PANELS=valid ANTS_TARGET_ROUND=3 python animations/scripts/make_ants_video.py
ANTS_PANELS=valid ANTS_TARGET_ROUND=10 python animations/scripts/make_ants_video.py
```

Bez `ANTS_PANELS` skrypt pokazuje obok siebie trafną i nietrafną wskazówkę. Kolonia na filmie to ta, której przebieg jest najbliżej średniej (mean-field).
