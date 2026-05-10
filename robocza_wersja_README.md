# Robocza wersja README

> To jest wersja robocza pliku README. Zawiera pierwotny opis roboczy oraz notatki do dalszej redakcji. Treść merytoryczna nie zostala zmieniona, dodano jedynie formatowanie dla lepszej czytelnosci.

# Co zrobić aby nasz lektor brzmiał dobrze i nadawał się do wrzucenia do filmu\serialu?

Dwa elementy są kluczowe.

## 1. Próbka głosu

Jeżeli dany model TTS obsluguje podanie próbki głosu to pierwszy krok to dobrze przygotowana próbka głosu.
Bez tego ani rusz, jak próbka będzie miała szumy czy trzaski, lub też ogólnie będzie słabej jakości to taki sam otrzymacie wynik.

## 2. Przygotowanie napisów

Przygotowanie napisów - tutaj akurat jest to podstawa dla każdego modelu TTS.
Generalnie jest to żmudna robota, teoretycznie można by napsiać prompta i wrzucić do jakiegoś AI aby to sam przerobił, tylko pytanie jaki był by rezultat, trzeba było by to potem zredagować i zobaczyć jak on to zrobił.

Jak przygotować odpowiednio napisy?

## Moja instrukcja

- usuwamy wszystkie tagi HTML, np.: `<i>`, `</i>`, `<b>`, `</b>`
- usuwamy wszystkie tagi ASS\SSA, np.: `{\an8}`, `{y:b}`, `{c:$1130bb}`
- usuwamy wszystkie tagi łamania linii: `\N`, `\n`, `\h`
- usuwamy nawiasy kwadratowe i ich zawartość, zazwyczaj są to opisy a nie kwestie dialogowe dla lektora, np.: `[muzyka]`, `[śmiech]`, `[telefon dzwoni]`
- usuwamy nawiasy zwykłe i ich zawartość, zazwyczaj są to opisy a nie kwestie dialogowe dla lektora, np.: `(jap.)`, `(niem.)` - tagi językowe
- usuwamy myślniki dialogowe na początku linii
- usuwamy znaki `&` lub zastępujemy je `i` w zależności od kontekstu
- usuwamy lub zastępujemy wielokropki `...` - jak wielokropek jest na samym końcu kwestii to usuwamy, jak jest w środku to zastępujemy `,`
- usuwamy wszelkie cudzysłowia i apostrofy, np.: `„”`, `“”`, `’`
- usuwamy wszelkie liczby i zastępujemy je wersjami fonetycznymi, np.: `1987` zamieniamy na: `tysiąc dziewięćset osiemdziesiąt siedem`, w zależności od kontekstu całego zdania formy końcowe mogą być różne
- usuwamy wszelkie znaki walut, procent oraz promili i zastępujemy je wersjami fonetycznymi, np.: `$` na `dolar`, w zależności od kontekstu całego zdania formy końcowe mogą być różne, może to być `dolar`, `dolarów`, `dolara`
- usuwamy etykiety mówców, np.: `JAN:`, `KOBIETA:`, `NARRATOR:`
- usuwamy artefakty złego kodowania np.: `�`, `Â`, `Ã`, `Å`, `Ĺ`, `â€™`
- usuwamy strzałki\symbole ekranowe np.: `->`, `←`, `↑`, `↓`
- usuwamy wszelkie nutki i znaczniki muzyki, np.: `♪`, `♫`, `♬`, np. `♪ tekst piosenki ♪`
- usuwamy wszelkie inne symbole np.: `★`, `•`, `◆`
- usuwamy wszelkie znaki języków obcych, np.: `é`, `ñ`, `ò`, `ì`, `À`, `Ä`, jeżeli są one częścią składową np. nazw własnych, imienia lub nazwiska, to trzeba to przemianować na wersję fonetyczną, używając tylko Polskich znaków aby lektor mógł to poprawnie przeczytać, np.: `Céline Dion` na `Selin Dion`; `À propos` na `a propos`
- wszelkie zagraniczne nazwy własne, imiona i nazwiska, które się inaczej piszę i inaczej wymawia sprowadzamy do wersji fonetycznej używając tylko polskich znaków tak aby lektor to poprawnie przeczytał, np.: `selfie` na `selfi`, `Roger` na `Rodżer`, `Skippy` na `Skipi`, `Viraj` na `Viraż`, `Woolbury` na `Łulbery`, `Grace` na `Grejs` itd....
- usuwamy wszelkie znaki interpunkcyjne z końca poszczególnych wypowiedzi, np. `Służby specjalne, wydział do spraw operacji finansowych, jesteś aresztowany.` na `Służby specjalne, wydział do spraw operacji finansowych, jesteś aresztowany` - dlaczego je usuwamy, bo każde zdanie to osobny plik lektora, musi on być w miarę precyzyjnie przycięty, tak aby plik zaczynał się kwestią lektora i nią kończył, bez niepotrzebnej ciszy na początku i końcu, takie prezycyjnie przycięte elemtny układanki, jak pliki będą miały za duzo `ciszy` to zacznie nam się lektor `rozjeżdżać`, są pewne wyjątki - o tym później

## Bardzo duży skrót

W bardzo dużym skrócie, w treści pliku z napisami, prócz oczywiście timestampów (znaczników czasowy napisów), mają być tylko:

- polskie słowa\litery
- wszystkie zapisy liczbowe mają być napisane w formie słownej
- wszelkie zagraniczne nazwy własne, imiona, nazwiska mają zostać przemianowane na wersje fonetyczne dla poprawnej wymowy przez lektora
- wszelkie symbole, znaki obce, tagi czy kodowania mają zniknąć z napisów badź zostać odpowiednio zastąpione

Generalnie nasz lektor ma mieć takie ustawienia, aby był `zimny`, czy też mało ekspresyjny, albo jeszcze inaczej - stabilny\bardziej przewidywalny czy też bardziej powtarzalny, chodzi o to, że tworzona ścieżka dźwiękowa lektora składa się z drobnych segmentów, każda linijka napisów to osobna wypowiedź, jak byśmy ustawili wyższą ekspresje lektora, to w każdym segmencie mogł by on brzmieć nieco inaczej, mieć inną barwę, w odsłuchu brzmiało by to tak, jakby każdą kwestię czytał inny lektor bo zmieniała by się barwa jego głosu - była by mniej przewidywalna\bardziej losowa

Co za tym idzie, jak nasz lektor ma być `zimny` to generalnie możemy podarować sobie takie znaki jak `?` czy `!` i zastąpić je albo `.` - dłuższa pauza, lub `,` - krótsza pauza.

Czyli właściwie to napisy sprowadzamy do samego tekstu z Polskich liter oraz dwóch znaków: `.` oraz `,` - to oczywiście duże uproszczenie, potem napiszę co zaobserwowałem w zachowaniu poszczególnych modeli TTS.

# Składnia

## 1. Przykład pierwszy

Przykładowy wycinek z napisów:

```text
- Więc to jednak dobry numer.
- Słuchaj, kolego...
```

Na poczatek usuwamy myślniki dialogowe i otrzymujemy:

```text
Więc to jednak dobry numer.
Słuchaj, kolego...
```

Teraz sprowadzamy to do jednej linii, mieliśmy myślniki dialogowe więc to zapewne kwestie dwóch różnych aktorów, więc oddzielmay te kwestie w zależności od modelu TTS (o tym później) albo `.` albo `,`, żeby mieć między nimi pauze:

```text
Więc to jednak dobry numer. Słuchaj, kolego...
```

Zachowałem `.` z porzedniej linii, ona da mi chwilową pauze, na koniec usuwamy `...` z końca linii, po pierwsze dlatego, że TTS-y nie poznają tego znaku, po drugie dlatego, że końcówka kwestii i tak jest przycinana, więc nie potrzeba nam na końcu kwestii generować dłuższego `ogona` do przycięcia, finalnie forma zostaje taka:

```text
Więc to jednak dobry numer. Słuchaj, kolego
```

## 2. Przykład drugi

Kolejną rzeczą jest to, że lektor bardzo nie lubi pojedynczych wyrazów, a już szczególnie jak są to bardzo krótkie wyrazy, gdy mamy coś takiego:

```srt
00:23:23,321 --> 00:23:24,656
Ty

00:23:24,756 --> 00:23:27,175
To Ty uradłeś mi samchód.
```

Widzimy sytuacje jak w jednej linii mamy tylko krótki wyraz `Ty` oraz że kolejna kwestia przypada praktycznie od razu po poprzedniej, w bardzo małym odstępnie czasowym. W takiej sytuacji musimy te dwie wypowiedzi połączyć. Do pierwszej wypowiedzi dopisujemy drugą, i potem tą drugą już pustą usuwamy z pliku z napisami. Dodatkowo aby zrobić krótką pauzę dodajemy `.` lub `,` między tymi dwoma połączonymi wypowiedziami, oraz usuwamy znak interpunkcyjny z końca wypowiedzi, zostaje:

```srt
00:23:23,321 --> 00:23:24,656
Ty. To Ty uradłeś mi samchód
```

## 3. Przykład trzeci

Lektorzy są wyczuleni na umieszczenie `.` oraz `,`, powinny się one znajdować bezpośrednio po slowie, którego dotyczą - po którym ma być pauza, a zaraz po nich ma być przerwa od następnego słowa:

Wersja poprawna:

```text
myślałem, że spędziemy razem wieczór
```

Wersje niepoprawne:

```text
myślałem , że spędziemy razem wieczór
myślałem ,że spędziemy razem wieczór
```

# Edge TTS

- znak `.` tworzy dużą pauzę, generalnie polecam zastąpić wszystkie `.` w napisach na `,`, jeżeli tego nie zrobisz to te długie pauzy będą sprawiały że lektor mocno będzie się `rozjeżdzał`
- znak `.` tworzy dużą pauzę, znak `..` jeszcze większą, znak `...` = `,`, czyli jako że jeszcze większe pauzy niż `.` są już kompletnie zbędne a `...` to `,` to lepiej po prostu nie stosować ani `..` ani `...`
- naki `!` oraz `?` powodują taką samą pauzę co `.`
- krótkie wyrazy są mu nie straszne, jak ma kwestię z jednym krótkim wyrazem, np.: `Ty` - to czyta ją poprawnie

# Chatterbox TTS

- znak `.` - średnia pauza (porównuje z edge TTS)
- znak `,` - krótka pauza
- znak `..` dla niego to jak `,` chociaż niekiedy widzę też delikatny efekt rozciągania mowy, natomiast znak `...` jest kompletnie ignorowany, jakby nic nie było, ogólnie lepiej nie stosować ani `..` ani `...`
- ma problem z słowami, które zawierają `si`, np.: `silos` czyta jako `śilos`, `seksi` czytata jako `sekśi`
- ma problem z pojedynczymi, krótkimi slowami, jak mu damy samo: `Ty` to będzie lunatykował i prócz tego `Ty` doda coś jeszcze od siebie, zauważyłem, że jak za `Ty` postawiłem `,` to wygenerował głos porawnie, przy `.` również lunatykował

# Omnivoice TTS

- między znakiem `,` a `.` jest tak mała różnica długości pauzy, moim zdaniem zarówno jeden jak i drugi znak tworzy krótkie pauzy, przy tym TTS polecał bym wszędzie sotoswać `.`, dodatkowo mam wrażenie, że długości tych pauz niekiedy bywają losowe
- ogólnie to ten model ma problemy z pauzami, mam wrażenie, że znak `?` robi lepszą pauzę od `.`, w sensie `?` to odpowiednik średniej pauzy - wymaga dokładniejszych testów
- krótkie wyrazy są mu nie straszne, jak ma kwestię z jednym krótkim wyrazem, np.: `Ty` - to czyta ją poprawnie, chociaż jak do `Ty` dodamy `.` lub `,` to już ma z Tym problem, dorabia ogon z lekkim zakłóceniem
