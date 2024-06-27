# Repozytorium badań nad transkrypcją audio

To repozytorium zawiera dane i skrypty używane do badań nad różnymi metodami wykrywania częstotliwości. Badane metody obejmują:

- Szybka transformacja Fouriera (FFT)
- Krótko-okresowa transformacja Fouriera (STFT)
- Transformacja Constant-Q (CQT)
- Transformacja Variable-Q (VQT)
- Konwolucyjne sieci neuronowe (CNN) z użyciem FFT

## Struktura Repozytorium

- samples/  
  Zawiera próbki testowe używane do badań i oceny różnych metod wykrywania częstotliwości.

- scripts/  
  Zawiera skrypty Python wykorzystujące biblioteki takie jak `librosa`, `numpy`, `scipy`, `tensorflow.keras`, `opencv-python`, `matplotlib` i inne. Te skrypty były używane do przeprowadzania eksperymentów dla każdej z metod.

## Wymagania

Aby uruchomić skrypty znajdujące się w tym repozytorium, należy zainstalować następujące biblioteki:

- `librosa`
- `numpy`
- `scipy`
- `tensorflow.keras`
- `opencv-python`
- `matplotlib`

Można je zainstalować za pomocą `pip`:

```bash
pip install librosa numpy scipy tensorflow opencv-python matplotlib
