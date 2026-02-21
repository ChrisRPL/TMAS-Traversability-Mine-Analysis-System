  
**DOKUMENT KONCEPCJI TECHNICZNEJ**

**TMAS**

**Traversability & Mine Analysis System**

*System wizyjny AI do analizy przejezdności terenu i detekcji min/IED*

| ZASTOSOWANIE WOJSKOWE / SAPERSKIE |
| :---: |

| Wersja: 2.0 Data: Luty 2026 Status: Do dyskusji |
| :---: |

Platforma: NVIDIA Jetson Orin | Sensory: RGB \+ Thermal \+ opcjonalnie GPR

# **Spis treści**

# **1\. Streszczenie wykonawcze**

System TMAS (Traversability & Mine Analysis System) to zaawansowana platforma wizji komputerowej AI przeznaczona do wsparcia pojazdów wojskowych i saperskich w środowisku off-road. System łączy dwie krytyczne funkcje: analizę przejezdności terenu oraz detekcję min i improwizowanych urządzeń wybuchowych (IED).

| ZASTOSOWANIE KRYTYCZNE DLA BEZPIECZEŃSTWA System przeznaczony jest do operacji wojskowych i saperskich. Fałszywy negatyw (przeoczenie miny) może skutkować utratą życia lub sprzętu. System musi osiągnąć recall \> 99.5% dla detekcji zagrożeń wybuchowych przy zachowaniu akceptowalnego poziomu fałszywych alarmów. |
| :---- |

## **1.1 Trzy główne moduły systemu**

| MODUŁ PRZEJEZDNOŚCI Klasyfikacja terenu (14 klas) Estymacja kosztów przejazdu Analiza geometrii terenu Mapa BEV 20m × 20m | MODUŁ DETEKCJI MIN Miny przeciwpancerne (AT) Miny przeciwpiechotne (AP) IED przydrożne/zakopane Fuzja RGB \+ Thermal | MODUŁ PRZESZKÓD Osoby, pojazdy, zwierzęta Przeszkody statyczne Nagłe pojawienie obiektów Predykcja TTC (kolizji) |
| :---- | :---- | :---- |

## **1.2 Kluczowe parametry systemu**

| Parametr | Wartość docelowa |
| ----- | ----- |
| Latencja inferencji | \< 25 ms (multi-sensor fusion) |
| Częstotliwość odświeżania | ≥ 20 FPS |
| Zasięg detekcji min | 30 metrów (RGB), 15m (thermal) |
| Zasięg detekcji przeszkód | 50 metrów (RGB) |
| Recall detekcji min/IED | \> 99.5% (krytyczne) |
| Recall detekcji osób/pojazdów | \> 99% (krytyczne) |
| Latencja nagłej przeszkody | \< 50 ms (1-2 klatki) |
| Rozdzielczość mapy BEV | 5 cm/piksel (400×400 grid) |
| Platforma docelowa | NVIDIA Jetson AGX Orin 64GB |

# **2\. Opis problemu i kontekst operacyjny**

Pojazdy wojskowe i saperskie operujące w strefach konfliktów lub na terenach skażonych minami napotykają na podwójne wyzwanie: ocenę przejezdności terenu oraz identyfikację ukrytych zagrożeń wybuchowych. Tradycyjne metody rozminowania są czasochłonne i narażają personel na bezpośrednie niebezpieczeństwo.

## **2.1 Scenariusze operacyjne**

| Scenariusz | Opis i wymagania |
| ----- | ----- |
| Konwój wojskowy | Ochrona kolumny pojazdów na trasie przejazdu. Wymagana szybka analiza (\>15 FPS) na dystansie 30m przy prędkości do 30 km/h. |
| Rozminowanie humanitarne | Systematyczne oczyszczanie terenu. Priorytet: maksymalny recall, akceptowalna niższa prędkość. Dokumentacja każdej detekcji. |
| Patrol rozpoznawczy | Identyfikacja bezpiecznych korytarzy przejazdu. Wymagana analiza wielu wariantów trasy w czasie rzeczywistym. |
| Wsparcie pojazdu autonomicznego | Integracja z systemem nawigacji autonomicznej. Wymagana niska latencja i deterministyczne czasy odpowiedzi. |

## **2.2 Typy zagrożeń do detekcji**

| Typ zagrożenia | Rozmiar typowy | Głębokość zakopania | Priorytet |
| ----- | ----- | ----- | ----- |
| Mina przeciwpancerna (AT) | 20-40 cm średnicy | 0-15 cm | KRYTYCZNY |
| Mina przeciwpiechotna (AP) | 5-15 cm średnicy | 0-10 cm | KRYTYCZNY |
| IED (przydrożne) | Zmienny | Powierzchnia lub zakopane | KRYTYCZNY |
| IED z przewodem | Przewód widoczny | Powierzchnia | WYSOKI |
| Amunicja niewybuchowa (UXO) | Zmienny | Powierzchnia | WYSOKI |
| Anomalia terenu | Zmienny | N/A | ŚREDNI |

## **2.3 Wyzwania techniczne**

* Kamuflaż: miny są projektowane tak, aby zlewać się z otoczeniem

* Zmienność: różne typy min, IED improwizowane z dostępnych materiałów

* Warunki środowiskowe: kurz, błoto, roślinność maskująca zagrożenia

* Zakopanie: część min jest częściowo lub całkowicie zakopana

* Ograniczenia sensorów: RGB nie widzi przez ziemię, thermal ma ograniczony zasięg

# **3\. Architektura systemu TMAS**

System TMAS wykorzystuje architekturę multi-modalną z fuzją danych z wielu sensorów. Podejście to maksymalizuje prawdopodobieństwo detekcji poprzez komplementarne właściwości różnych typów sensorów.

## **3.1 Konfiguracja sensorów**

| Sensor | Zastosowanie | Zalety | Ograniczenia |
| ----- | ----- | ----- | ----- |
| Kamera RGB | Detekcja wizualna, klasyfikacja terenu | Wysoka rozdzielczość, kolor, tekstura | Nie widzi zakopanych obiektów |
| Kamera termalna (LWIR) | Detekcja anomalii termicznych | Działa w nocy, widzi różnice temp. | Niższa rozdzielczość, zależność od warunków |
| GPR (opcja) | Detekcja obiektów zakopanych | Widzi pod powierzchnią | Wolny, wymaga bliskiego kontaktu |
| LiDAR (opcja) | Precyzyjna geometria 3D | Dokładny depth, niezależny od światła | Wysoki koszt, problemy z kurzem |

## **3.2 Schemat architektury systemu**

| ARCHITEKTURA SYSTEMU TMAS ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │  KAMERA RGB  │  │   THERMAL    │  │  GPR (opcja) │ └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │                 │                 │        ▼                 ▼                 ▼ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │ RGB Backbone │  │Thermal Backb.│  │  GPR Signal  │ │ EfficientViT │  │  ResNet-18   │  │  Processing  │ └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        └─────────────────┼─────────────────┘                         ▼          ┌──────────────────────────────┐          │      MULTI-MODAL FUSION      │          │   (Cross-Attention Transformer)│          └─────────────┬────────────────┘        ┌───────────────┼───────────────┐        ▼               ▼               ▼ ┌──────────────┐┌──────────────┐┌──────────────┐ │   TERRAIN    ││    MINE      ││   OBSTACLE   │ │SEGMENTATION  ││  DETECTION   ││  DETECTION   │ └──────┬───────┘└──────┬───────┘└──────┬───────┘        └───────────────┼───────────────┘                        ▼          ┌──────────────────────────────┐          │     BEV TRANSFORM \+ FUSION   │          │   (Unified Threat \+ Cost Map)│          └──────────────────────────────┘ |
| :---: |

## **3.3 Moduł detekcji min i IED**

Moduł detekcji zagrożeń wybuchowych stanowi krytyczny element systemu. Wykorzystuje podejście multi-stage z kaskadą klasyfikatorów o rosnącej precyzji, co pozwala na osiągnięcie bardzo wysokiego recall przy akceptowalnym poziomie false positives.

### **Etap 1: Detekcja anomalii (High Recall)**

* Segmentacja semantyczna wyodrębnia regiony potencjalnie zawierające zagrożenia

* Analiza tekstury: wykrywanie regularnych kształtów nietypowych dla terenu naturalnego

* Analiza termalna: wykrywanie różnic temperatury (zakopane obiekty mają inną pojemność cieplną)

* Detekcja anomalii w depth map: wykrywanie nienaturalnych wypukłości/wgłębień

### **Etap 2: Klasyfikacja zagrożeń (High Precision)**

* Detektor obiektów (YOLOv8/RT-DETR) trenowany na zdjęciach min i IED

* Klasyfikator typu zagrożenia: AT mine / AP mine / IED / UXO / false positive

* Estymacja pewności z Evidential Deep Learning

* Fuzja wielosensorowa: RGB \+ Thermal zwiększa pewność detekcji

### **Etap 3: Weryfikacja i lokalizacja**

* Temporal consistency: śledzenie detektcji przez wiele klatek

* Triangulacja pozycji z wielu ujęć dla precyzyjnej lokalizacji GPS

* Confidence aggregation: wzrost pewności przy powtarzających się detekcjach

| ZASADA BEZPIECZEŃSTWA System zawsze faworyzuje fałszywe alarmy nad przeoczenia. Każda potencjalna anomalia jest raportowana operatorowi. Próg detekcji jest ustawiony bardzo nisko (high sensitivity), a ostateczna decyzja należy do człowieka. |
| :---- |

## **3.4 Moduł detekcji przeszkód**

Oprócz detekcji min i analizy terenu, system TMAS zawiera dedykowany moduł do wykrywania przeszkód na trasie przejazdu. Moduł ten obsługuje zarówno przeszkody statyczne, jak i dynamiczne (poruszające się), w tym obiekty pojawiające się nagle w polu widzenia.

### **Typy wykrywanych przeszkód**

| Typ przeszkody | Kategoria | Priorytet | Przykłady |
| ----- | ----- | ----- | ----- |
| Osoby / piesi | Dynamiczna | KRYTYCZNY | Żołnierze, cywile, saperzy |
| Pojazdy | Dynamiczna | KRYTYCZNY | Samochody, ciężarówki, sprzęt wojskowy |
| Zwierzęta | Dynamiczna | WYSOKI | Duże zwierzęta (dziki, sarny, psy) |
| Przewrócone drzewa | Statyczna | WYSOKI | Blokada drogi, konary |
| Głazy / skały | Statyczna | WYSOKI | Kamienie \> 30cm średnicy |
| Wraki pojazdów | Statyczna | ŚREDNI | Porzucone/zniszczone pojazdy |
| Gruz / debris | Statyczna | ŚREDNI | Fragmenty budynków, metalu |
| Barykady | Statyczna | ŚREDNI | Improwizowane blokady |
| Dziury / kratery | Statyczna | WYSOKI | Wyrwy po wybuchach, erozja |

### **Architektura detekcji przeszkód**

* Detektor obiektów: RT-DETR / YOLOv8 trenowany na 20+ klasach przeszkód wojskowych i cywilnych

* Segmentacja instancji: rozdzielenie nakładających się obiektów dla precyzyjnej lokalizacji

* Estymacja głębi: odległość do przeszkody z monocular depth lub stereo

* Tracking obiektów: ByteTrack dla śledzenia przeszkód dynamicznych między klatkami

* Predykcja trajektorii: dla obiektów ruchomych \- przewidywanie kolizji

### **Detekcja nagłych przeszkód (Sudden Obstacle Detection)**

System implementuje specjalny mechanizm do wykrywania obiektów pojawiających się nagle w polu widzenia:

* Frame differencing: porównanie kolejnych klatek dla wykrycia nagłych zmian

* Motion saliency: wyodrębnienie regionów o wysokiej dynamice ruchu

* Edge-triggered alerts: natychmiastowy alert przy wykryciu nowego obiektu w strefie krytycznej

* Time-to-collision (TTC): estymacja czasu do potencjalnej kolizji

* Emergency brake recommendation: przy TTC \< 2s system rekomenduje hamowanie awaryjne

| Strefy detekcji przeszkód • Strefa krytyczna (0-10m): natychmiastowy alert, możliwe hamowanie awaryjne• Strefa ostrzegawcza (10-20m): ostrzeżenie, przygotowanie do reakcji• Strefa obserwacji (20-50m): monitorowanie, planowanie omijania |
| :---- |

### **Parametry wydajnościowe modułu przeszkód**

| Parametr | Wartość docelowa |
| ----- | ----- |
| Recall dla osób/pojazdów | \> 99% (krytyczne) |
| Recall dla przeszkód statycznych | \> 95% |
| Latencja detekcji nagłej przeszkody | \< 50 ms (1-2 klatki) |
| Dokładność estymacji odległości | ± 0.5m do 20m |
| Dokładność predykcji TTC | ± 0.3s |
| Minimalny wykrywalny obiekt | 30cm × 30cm @ 20m |

# **4\. Klasyfikacja zagrożeń i terenu**

## **4.1 Hierarchia klas zagrożeń i przeszkód**

| Klasa | Priorytet | Koszt BEV | Akcja | Uwagi |
| ----- | ----- | ----- | ----- | ----- |
| Mina AT (potwierdzona) | KRYTYCZNY | ∞ (blokada) | STOP \+ ALARM | Okrągły kształt 20-40cm |
| Mina AP (potwierdzona) | KRYTYCZNY | ∞ (blokada) | STOP \+ ALARM | Mały obiekt 5-15cm |
| IED (potwierdzone) | KRYTYCZNY | ∞ (blokada) | STOP \+ ALARM | Nieregularny, przewody |
| Podejrzana anomalia | WYSOKI | 0.95 | OSTRZEŻENIE | Wymaga weryfikacji |
| Osoba / pieszy | KRYTYCZNY | ∞ (blokada) | STOP | Detekcja \+ tracking |
| Pojazd w ruchu | KRYTYCZNY | ∞ (blokada) | STOP/OMIJAJ | Predykcja trajektorii |
| Duże zwierzę | WYSOKI | 0.95 | HAMUJ | Nieprzewidywalne |
| Przewrócone drzewo | WYSOKI | ∞ (blokada) | SZUKAJ OBJAZDU | Blokada całkowita |
| Głaz / skała \> 30cm | WYSOKI | 0.9 | OMIJAJ | Ryzyko uszkodzenia |
| Wrak pojazdu | ŚREDNI | 0.85 | OMIJAJ | Możliwe pułapki |
| Krater / dziura | WYSOKI | 0.9 | OMIJAJ | Ryzyko ugrzęźnięcia |
| Gruz / debris | ŚREDNI | 0.7 | OSTROŻNIE | Możliwe ukryte zagrożenia |

## **4.2 Klasy terenu i przejezdność**

| Typ terenu | Koszt bazowy | Kategoria | Uwagi taktyczne |
| ----- | ----- | ----- | ----- |
| Droga utwardzona | 0.0 | Łatwy | Preferowana, ale wyższe ryzyko IED przydrożne |
| Droga żwirowa | 0.1 | Łatwy | Dobra widoczność, łatwiejsza detekcja |
| Sucha trawa (niska) | 0.15 | Łatwy | Dobra widoczność min powierzchniowych |
| Ubita ziemia | 0.2 | Łatwy | Możliwe zakopane miny AT |
| Piasek | 0.4 | Średni | Łatwe ukrycie min, trudna detekcja |
| Wysoka trawa/zioła | 0.5 | Średni | Ograniczona widoczność, ryzyko AP |
| Gęste zarośla | 0.7 | Trudny | Bardzo ograniczona widoczność |
| Teren podmokły | 0.6 | Trudny | Miny mogą być przesunięte |
| Gruzy/ruiny | 0.8 | Trudny | Wysokie ryzyko IED i pułapek |

## **4.3 Formuła końcowego kosztu przejezdności**

| Wzór obliczania kosztu Cost\_final \= max(Cost\_terrain \+ Cost\_geometry, Cost\_threat)gdzie:• Cost\_terrain \= bazowy koszt typu terenu \[0-1\]• Cost\_geometry \= modyfikator za slope/roughness \[0-0.4\]• Cost\_threat \= koszt wykrytego zagrożenia \[0-∞\]Jeśli Cost\_threat \= ∞, komórka jest zablokowana niezależnie od terenu. |
| :---- |

# **5\. Szczegóły modułu detekcji min**

## **5.1 Sygnatury wizualne min**

System jest trenowany na rozpoznawanie charakterystycznych cech wizualnych różnych typów min. Poniżej przedstawiono główne kategorie i ich sygnatury:

| Typ miny | Charakterystyczne cechy wizualne |
| ----- | ----- |
| Miny AT (np. TM-62) | Okrągły lub kwadratowy kształt, średnica 20-40cm, często metalowy lub plastikowy, zapalnik centralny, może być częściowo widoczny brzeg |
| Miny AP naciskowe | Małe okrągłe obiekty 5-15cm, często zielone/brązowe, mogą mieć widoczny mechanizm nacisku |
| Miny AP odłamkowe | Charakterystyczny kształt (np. MON-50 \- zakrzywiona), mogą być zamontowane na kijkach lub drzewach |
| IED przydrożne | Nieregularne kształty, widoczne przewody, nietypowe obiekty przy drodze (worki, pojemniki, gruz) |
| IED zakopane | Świeżo przekopana ziemia, anomalie w teksturze powierzchni, wystający element spustowy |

## **5.2 Sygnatury termalne**

* Pojemność cieplna: zakopane obiekty metalowe/plastikowe mają inną pojemność cieplną niż ziemia

* Efekt świtu/zmierzchu: największy kontrast termiczny występuje przy zmianach temperatury otoczenia

* Anomalie nocne: zakopane obiekty wolniej oddają ciepło niż otaczająca ziemia

* Wzorce termalne IED: komponenty elektroniczne mogą generować ciepło

## **5.3 Architektura detektora**

| Komponent | Specyfikacja |
| ----- | ----- |
| Backbone RGB | EfficientViT-L2 (współdzielony z segmentacją) |
| Backbone Thermal | ResNet-18 (lekki, szybki) |
| Detektor obiektów | RT-DETR-L (real-time DETR) |
| Fuzja modalnośći | Cross-Attention Transformer |
| Liczba klas detekcji | 8 (typy min \+ IED \+ anomalie) |
| Estymacja niepewności | Evidential Deep Learning \+ MC Dropout |
| Temporal tracking | ByteTrack z confidence accumulation |

# **6\. Stack technologiczny**

## **6.1 Platforma sprzętowa**

| Komponent | Specyfikacja |
| ----- | ----- |
| Jednostka obliczeniowa | NVIDIA Jetson AGX Orin 64GB |
| GPU | 2048 rdzeni CUDA, 64 rdzeni Tensor |
| CPU | 12-core ARM Cortex-A78AE |
| Pamięć | 64GB LPDDR5 (204 GB/s) |
| Pobór mocy | 15-60W (konfigurowalny) |
| Kamera RGB | FLIR Blackfly S (global shutter, 12MP) |
| Kamera termalna | FLIR Boson 640 (LWIR, 640×512) |
| GPR (opcja) | GSSI StructureScan Mini XT |
| GPS/INS | Emlid Reach RS2+ (RTK, cm accuracy) |
| Obudowa | MIL-STD-810G (wstrząsy, temp., IP67) |

## **6.2 Stack oprogramowania**

| Warstwa | Technologia |
| ----- | ----- |
| System operacyjny | JetPack 6.0 (Ubuntu 22.04 hardened) |
| CUDA / cuDNN / TensorRT | CUDA 12.2 / cuDNN 8.9 / TensorRT 8.6 |
| Framework ML | PyTorch 2.2 → ONNX → TensorRT |
| Middleware | ROS 2 Iron (LTS) |
| Synchronizacja sensorów | Hardware trigger \+ PTP |
| Logging/Recording | MCAP format (deterministyczny) |
| Interfejs operatora | Qt6 \+ RViz2 |

## **6.3 Wymagania wojskowe**

* MIL-STD-810G: odporność na wstrząsy, wibracje, temperaturę (-40°C do \+55°C)

* MIL-STD-461G: kompatybilność elektromagnetyczna (EMC/EMI)

* IP67: pyło- i wodoszczelność

* Zasilanie: 9-36V DC (kompatybilność z pojazdami wojskowymi)

* MTBF: \> 5000 godzin

# **7\. Strategia treningu i dane**

## **7.1 Źródła danych treningowych**

| Dataset / Źródło | Rozmiar | Typ danych | Zastosowanie |
| ----- | ----- | ----- | ----- |
| GICHD Mine Database | 10k+ images | RGB anotowane | Trening detekcji min |
| Synthetic Mine Data | 100k+ images | Renderowane 3D | Augmentacja, edge cases |
| RELLIS-3D | 13k frames | RGB \+ LiDAR | Segmentacja terenu |
| TartanDrive | 200k frames | RGB \+ stereo | Depth estimation |
| Thermal Mine Dataset | 5k images | LWIR anotowane | Trening thermal branch |
| Dane własne (poligon) | 50+ km tras | Multi-sensor | Fine-tuning, walidacja |

## **7.2 Generowanie danych syntetycznych**

Ze względu na ograniczoną dostępność rzeczywistych danych z minami, znaczącą część treningu opieramy na danych syntetycznych:

* Modele 3D min: dokładne odwzorowania geometryczne popularnych typów min (TM-62, PMN, MON-50, itp.)

* Proceduralne generowanie terenu: różne typy podłoża, oświetlenie, warunki pogodowe

* Domain Randomization: losowa zmiana tekstur, kolorów, pozycji dla lepszej generalizacji

* Symulacja thermal: modelowanie właściwości termicznych min w różnych warunkach

* Symulacja zakopania: częściowo widoczne miny, świeża ziemia, maskowanie

## **7.3 Strategia walidacji**

| WALIDACJA KRYTYCZNA Model musi przejść rygorystyczną walidację na rzeczywistych danych z poligonu przed dopuszczeniem do użytku operacyjnego. Wymagane testy:• Test na poligonie z atrapami min (min. 500 scenariuszy)• Test w różnych warunkach pogodowych i oświetleniowych• Test z różnymi typami maskowania (trawa, piasek, błoto)• Ocena przez ekspertów saperskich |
| :---- |

# **8\. Interfejs wyjściowy i wizualizacja**

## **8.1 Format danych wyjściowych**

| Wyjście | Format / Opis |
| ----- | ----- |
| Mapa kosztów BEV | Grid 400×400, float32, 5cm/piksel, zasięg 20m×20m |
| Mapa zagrożeń | Grid 400×400, uint8, klasy zagrożeń \+ confidence |
| Lista detekcji min/IED | Array: {type, position\_GPS, confidence, bbox, tracking\_id} |
| Lista przeszkód | Array: {class, bbox, distance, velocity, TTC, tracking\_id} |
| Mapa semantyczna | Grid 400×400, uint8, 14 klas terenu |
| Strumień alertów | Real-time: {severity, type, position, TTC, timestamp} |
| Predykcje trajektorii | Array: {object\_id, predicted\_path\[\], collision\_risk} |
| Log detekcji | MCAP: pełna historia do analizy post-mission |

## **8.2 Interfejs operatora (HMI)**

| INTERFEJS OPERATORA TMAS v2.0 ┌────────────────────────────────────────────────────────────┐ │  WIDOK KAMERY RGB    │    MAPA BEV (20m × 20m)           │ │  ┌──────────────┐    │    ░░░░░░▓▓▓▓▓▓░░░░░░            │ │  │   \[□\]DRZEWO  │    │    ░░░\[■\]████▓▓░░░░  ← DRZEWO    │ │  │              │    │    ░░░▓██    ██▓░░░              │ │  │   \[\!\] MINA   │    │    ░░░▓█\[X\]██▓▓░░░  ← MINA AT   │ │  │    AT-95%    │    │    ░░░░▓▓████▓▓░░░░              │ │  │  \[○\]OSOBA→   │    │    ░░\[○\]→░░░░░░░░░  ← OSOBA     │ │  │   TTC: 3.2s  │    │    ░░░░░░░░░░░░░░░░              │ │  └──────────────┘    │         \[▲\]                      │ │                      │        POJAZD                    │ ├──────────────────────┴───────────────────────────────────┤ │  ⚠️  ALERT: WYKRYTO MINĘ PRZECIWPANCERNĄ                 │ │  Typ: TM-62 (95%) | Dystans: 12.5m | ZATRZYMAJ POJAZD\!   │ ├────────────────────────────────────────────────────────────┤ │  ⚡ PRZESZKODY: Drzewo@8m (blokada) | Osoba@15m (TTC:3.2s)│ ├────────────────────────────────────────────────────────────┤ │  Status: AKTYWNY | FPS: 22 | Lat: 24ms | Obiektów: 3     │ └────────────────────────────────────────────────────────────┘ |
| :---: |

# **9\. Bezpieczeństwo i weryfikacja**

## **9.1 Wymagania bezpieczeństwa (Safety Requirements)**

| ID | Wymaganie | Weryfikacja |
| ----- | ----- | ----- |
| SR-1 | Recall detekcji min AT \> 99.5% | Test na 1000+ scenariuszy poligonowych |
| SR-2 | Recall detekcji min AP \> 99.0% | Test na 1000+ scenariuszy poligonowych |
| SR-3 | Recall detekcji IED \> 98.5% | Test na 500+ scenariuszy z atrapami IED |
| SR-4 | Recall detekcji osób \> 99.5% | Test na scenariuszach z pieszymi |
| SR-5 | Recall detekcji pojazdów \> 99.0% | Test z różnymi typami pojazdów |
| SR-6 | Latencja detekcji nagłej przeszkody \< 50ms | Test z nagłym pojawieniem obiektu |
| SR-7 | Czas reakcji na zagrożenie \< 100ms | Testy latencji end-to-end |
| SR-8 | System degradacji graceful przy awarii sensora | Testy fault injection |
| SR-9 | Logowanie 100% detekcji do post-analysis | Audit trail verification |
| SR-10 | Alert dźwiękowy przy detekcji zagrożenia | Test funkcjonalny |
| SR-11 | Brak autonomicznych decyzji \- człowiek w pętli | Design review |
| SR-12 | Dokładność TTC (Time-to-Collision) ± 0.3s | Test z obiektami ruchomymi |

## **9.2 Scenariusze awaryjne**

| Awaria | Reakcja systemu | Akcja operatora |
| ----- | ----- | ----- |
| Utrata kamery RGB | Przełączenie na tryb thermal-only, alert | Ograniczyć prędkość, rozważyć wycofanie |
| Utrata kamery thermal | Kontynuacja z RGB-only, obniżony confidence | Zwiększona ostrożność |
| Utrata GPS | Lokalizacja względna, oznaczenie na mapie | Ręczne notowanie pozycji |
| Przegrzanie GPU | Redukcja FPS, priorytet dla detekcji min | Zatrzymanie, ochłodzenie |
| Błąd software | Automatyczny restart, alert | Zatrzymanie do weryfikacji |

## **9.3 Metryki jakości**

* True Positive Rate (Recall) dla min: \> 99.5% \- absolutnie krytyczne

* True Positive Rate (Recall) dla osób/pojazdów: \> 99% \- krytyczne

* Recall dla przeszkód statycznych: \> 95%

* False Positive Rate (miny): \< 10% \- akceptowalne, fałszywe alarmy są bezpieczne

* False Positive Rate (przeszkody): \< 15%

* Precision (ogólna): \> 50% \- przy wysokim recall, niższa precision jest akceptowalna

* mIoU segmentacji terenu: \> 75%

* Depth RMSE: \< 1.5m dla obiektów w zasięgu 15m

* Dokładność predykcji TTC: ± 0.3s dla obiektów ruchomych

* Uncertainty calibration (ECE): \< 0.05

# **10\. Harmonogram implementacji**

| Faza | Czas | Deliverables |
| ----- | ----- | ----- |
| 1\. R\&D | 8 tyg. | Prototyp modelu detekcji min, baseline na danych syntetycznych |
| 2\. Integracja sensorów | 4 tyg. | Kalibracja RGB-Thermal, synchronizacja hardware |
| 3\. Optymalizacja | 4 tyg. | TensorRT INT8, multi-stream inference \< 25ms |
| 4\. Dane poligonowe | 6 tyg. | Zbieranie danych na poligonie z atrapami min |
| 5\. Fine-tuning | 4 tyg. | Trening na danych rzeczywistych, domain adaptation |
| 6\. Integracja pojazdu | 4 tyg. | Montaż na pojeździe, ROS 2, HMI |
| 7\. Walidacja | 6 tyg. | Testy akceptacyjne, certyfikacja bezpieczeństwa |
| 8\. Pilotaż | 4 tyg. | Testy operacyjne z udziałem saperów |

| Całkowity czas realizacji 40 tygodni (około 10 miesięcy) od rozpoczęcia do gotowego systemu po walidacji. Faza pilotażowa wymaga współpracy z jednostką wojskową/saperską. |
| :---- |

## **10.1 Kamienie milowe**

| Tydzień | Milestone |
| ----- | ----- |
| T+8 | ✓ Recall \> 95% na danych syntetycznych |
| T+12 | ✓ Działająca fuzja RGB \+ Thermal |
| T+16 | ✓ Inferencja \< 25ms na Jetson AGX Orin |
| T+22 | ✓ Dataset poligonowy: min. 500 scenariuszy z minami |
| T+26 | ✓ Recall \> 99% na danych rzeczywistych |
| T+30 | ✓ System zintegrowany na pojeździe |
| T+36 | ✓ Przejście testów akceptacyjnych |
| T+40 | ✓ Pozytywna ocena z pilotażu operacyjnego |

# **11\. Ryzyka i mitygacje**

| Ryzyko | Prawdop. | Wpływ | Mitygacja |
| ----- | ----- | ----- | ----- |
| Niewystarczający recall detekcji | Średnie | KRYTYCZNY | Architektura kaskadowa, ensemble models, conservative thresholds |
| Brak dostępu do danych o minach | Wysokie | Wysoki | Dane syntetyczne, współpraca z GICHD, poligon wojskowy |
| Słaba generalizacja na nowe typy min | Średnie | Wysoki | Domain randomization, continual learning, modular architecture |
| Problemy z synchronizacją sensorów | Niskie | Średni | Hardware trigger, PTP, fallback na software sync |
| Przegrzanie w warunkach polowych | Średnie | Średni | Active cooling, thermal throttling, shade mounting |
| Opóźnienia w certyfikacji wojskowej | Wysokie | Średni | Wczesne zaangażowanie MON, parallel certification track |

# **12\. Podsumowanie**

System TMAS stanowi kompleksowe rozwiązanie łączące zaawansowaną analizę przejezdności terenu, detekcję min i IED, oraz wykrywanie przeszkód na trasie przejazdu. Kluczowe innowacje projektu obejmują:

* Multi-modal sensor fusion (RGB \+ Thermal \+ opcjonalnie GPR) dla maksymalnego recall

* Trójmodułowa architektura: przejezdność \+ miny/IED \+ przeszkody statyczne/dynamiczne

* Kaskadowa detekcja z priorytetem na eliminację fałszywych negatywów

* Detekcja nagłych przeszkód z predykcją Time-to-Collision (TTC)

* Evidential Deep Learning dla kalibrowanej estymacji niepewności

* Tracking obiektów ruchomych z predykcją trajektorii

* Safety-critical design z redundancją i graceful degradation

| KLUCZOWE ZASTRZEŻENIE System TMAS jest narzędziem wspomagającym, nie zastępującym wykwalifikowanego sapera. Ostateczna decyzja o bezpieczeństwie terenu zawsze należy do człowieka. System może nie wykryć wszystkich zagrożeń \- żadna technologia nie gwarantuje 100% skuteczności w detekcji min. |
| :---- |

| Następne kroki 1\) Zatwierdzenie koncepcji przez interesariuszy wojskowych2) Pozyskanie dostępu do poligonu i danych treningowych3) Zamówienie platformy sprzętowej (Jetson AGX Orin \+ sensory)4) Rozpoczęcie fazy R\&D (tydzień 1-8)5) Nawiązanie współpracy z jednostką saperską dla pilotażu |
| :---- |

| Dokument przygotowany do dyskusji Wersja 2.0 | Luty 2026 |
| :---: |

