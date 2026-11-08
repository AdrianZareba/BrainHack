# Jak działa model mrówkowy?

**Błysk przyciąga uwagę do jednej strony. Częste wybieranie tej strony zużywa jej zasób, więc po pewnym czasie uwaga przechodzi na drugą.** To proponowane wyjaśnienie przejścia od początkowej korzyści z podpowiedzi do hamowania powrotu (IOR).

Model z `posner_ants.ipynb` składa się z kolonii, która dzieli uwagę między dwie strony, oraz dekodera, który próbuje zlokalizować lampkę. Wyniki analizy parametrów są w [README.md](README.md).

## Od błysku do zmiany strony

Mrówki wybierają między dwoma szlakami. Odsetek mrówek na danym szlaku oznacza udział uwagi skierowanej w tę stronę. W każdej rundzie:

1. **Mrówki wybierają stronę.** Przyciąga je błysk i silniejszy ślad feromonowy. Część wybiera losowo.
2. **Wybrane miejsce daje sukces z prawdopodobieństwem zależnym od zasobu.** Sukcesy wzmacniają ślad, ale jednocześnie zużywają zasób.
3. **Błysk i stary ślad słabną.** W kolejnej rundzie wybór zależy od zmienionego stanu obu stron.

Początkowo wygrywa strona błysku. Ponieważ odwiedza ją więcej mrówek, szybciej traci zasób. Z czasem daje mniej sukcesów i słabiej wzmacnia swój ślad. Druga strona staje się atrakcyjniejsza.

**W tej architekturze zużycie zasobu jest konieczne dla IOR:** po jego wyłączeniu efekt znika. To wskazuje na przyczynę przełączenia w modelu, ale samo w sobie nie dowodzi, że tak samo działa ludzka uwaga.

## Jak uwaga wpływa na wynik?

Dekoder dostaje zaszumiony obraz lampki. Im więcej uwagi przypada na jej stronę, tym większy jest kontrast lampki i zwykle mniejszy błąd lokalizacji.

Porównujemy dwie sparowane próby: lampkę po stronie błysku i lampkę po stronie przeciwnej. Kolonia i szum są w obu próbach takie same. Gdy większość uwagi jest po stronie błysku, oczekujemy korzyści z trafnej podpowiedzi. Gdy udział spada poniżej połowy, oczekujemy IOR.

Ta zależność zakłada, że uwaga poprawia odczyt. Przy zbyt łatwym lub trudnym zadaniu różnica może być niewidoczna. Jeśli kontrast bez uwagi jest większy niż z uwagą, kierunek efektu się odwraca.

Odczyt nie wpływa na kolonię: sukces mrówki zależy od zasobu, a nie od poprawnej lokalizacji lampki.

## Co oznaczają mrówki, feromon i zasób?

| Element | Interpretacja |
|---|---|
| Podział mrówek między strony | Rywalizacja miejsc o ograniczoną pulę uwagi |
| Feromon | Pamięć wcześniejszych sukcesów, która podtrzymuje wybór miejsca |
| Zanik feromonu | Stopniowe zapominanie wcześniejszej przewagi |
| Błysk | Krótkotrwałe, odruchowe przyciągnięcie uwagi |
| Zużycie zasobu | Hipoteza adaptacji lub „zmęczenia” często wybieranego miejsca |
| Losowe wybory i liczba mrówek | Źródła zmienności między próbami |

To analogie, nie ustalone odpowiedniki biologiczne. Szczególnie zasób wymaga wyjaśnienia: w zadaniu Posnera po błysku nie ma zwykle niczego do „wyzbierania”. Adaptacja jest tu bardziej bezpośrednią interpretacją niż dosłowne żerowanie.

Średni przebieg można dobrze przybliżyć bez symulowania pojedynczych mrówek, zastępując losowania wartościami oczekiwanymi. Mediana różnicy względem 1800 kolonii wynosiła 0,004; w niektórych ustawieniach różnice dochodziły do 0,09. Sednem modelu jest więc dynamika rywalizacji, pamięci i zużycia. Mrówki dodają zmienność skończonej populacji.

## Czego model jeszcze nie wyjaśnia?

- **Czasu w milisekundach.** Model liczy rundy. Dopasowanie przecięcia w T7–T8 do przykładowych 250 ms dawałoby około 33 ms na rundę, ale jest to założenie kalibracyjne, nie wynik pomiaru.
- **Trwałości i odnowy uwagi.** Zasób tylko maleje. W końcu obie strony się wyczerpują i podział wraca w stronę 50/50. Model nie opisuje regeneracji ani kolejnych podpowiedzi bez resetu.
- **Czasów reakcji.** Wynikiem jest błąd lokalizacji, podczas gdy efekt Posnera zwykle ocenia się czasem reakcji. Nie można bezpośrednio porównywać wielkości tych efektów.
- **Realistycznej siły uwagi.** Skrajne kontrasty mają stosunek 9,4:1. Tak duży zakres wymaga uzasadnienia danymi. Także poprawa oszacowania tła w kolejnych rundach wpływa na kształt krzywych.
- **Ruchów oczu i przemieszczania obiektów.** Dwie stałe lokalizacje nie wystarczą do opisania tych sytuacji.
- **Jednoznacznych wartości parametrów.** Wiele ustawień może dawać podobny przebieg. Dodatkowo dekoder uczy się na udziałach uwagi od 0 do 1, choć w typowym teście są one bliskie 0,4–0,6.

Trzeba też oddzielać ograniczenia modelu od ograniczeń analizy. Jeśli IOR pojawia się po T10, obecne kryterium uznaje wynik za niespełniający warunku, mimo że przełączenie występuje później.

## Co warto sprawdzić dalej?

1. **Dodać odnawianie zasobu** i zbadać dłuższe przebiegi oraz kolejne błyski. To propozycja rozszerzenia; nie wiadomo jeszcze, czy poprawi zgodność z danymi.
2. **Dopasować czas i kontrast do konkretnego eksperymentu**, a następnie sprawdzić cały przebieg, nie tylko moment przecięcia.
3. **Uprościć odczyt.** Temperatura ma niewielki wpływ po ponownym treningu dekodera. Warto też sprawdzić stałą jakość tła w kolejnych rundach, korzystając z kontroli `equal_reads=True` w notebooku.
4. **Porównać alternatywne mechanizmy.** Obecny model wyprowadza IOR ze zużycia zasobu. Bezpośrednie porównanie z modelem jawnego hamowania pozwoli ocenić, czy daje to inne, sprawdzalne przewidywania.

## Literatura do weryfikacji

Punkty wyjścia dla interpretacji to: Posner i Cohen (1984) oraz Klein (2000) dla IOR; Reynolds i Heeger (2009) dla normalizacji uwagi; Tsodyks i Markram (1997) dla zużywania i odnowy zasobu synaptycznego; Carrasco, Ling i Read (2004) dla wpływu uwagi na postrzegany kontrast.

Odwołania przejęto z wcześniejszej wersji dokumentu i nie zweryfikowano ich w źródłach. Przed wykorzystaniem w pracy należy sprawdzić, które twierdzenia rzeczywiście wspierają.
