# Contactless Fingerprint Quality Assessment Pipeline

A Python-based fingerprint quality assessment system that evaluates the quality of contactless fingerprint images before biometric processing. The project computes multiple image quality metrics and provides a final quality score along with user guidance.

---

## Features

- Blur Detection using Laplacian Variance
- Brightness Assessment
- Glare Detection
- ROI (Region of Interest) Completeness
- Ridge Clarity Analysis using Gabor Filter
- Composite Quality Score (0–100)
- Pass / Fail Decision
- User Guidance for Improving Capture
- Interactive Streamlit Web Application
- Unit Tests using PyTest

---

## Project Structure

```
fingerprint-quality-assessment-final/
│
├── quality_assessment.py      # Core quality assessment pipeline
├── quality_app.py             # Streamlit application
├── test_quality.py            # Unit tests
├── requirements.txt           # Project dependencies
├── README.md                  # Documentation
├── test_images/               # Sample fingerprint images
└── venv/
```

---

## Quality Metrics

The system evaluates fingerprint quality using the following metrics:

| Metric | Technique |
|---------|-----------|
| Blur | Laplacian Variance |
| Brightness | Mean Grayscale Intensity |
| Glare | High Intensity Pixel Ratio |
| ROI Completeness | Otsu Thresholding |
| Ridge Clarity | Gabor Filter Response |

---

## Composite Quality Score

Each metric contributes to the final score using weighted normalization.

| Metric | Weight |
|---------|-------|
| Blur | 25% |
| Brightness | 15% |
| Glare | 15% |
| ROI Completeness | 20% |
| Ridge Clarity | 25% |

Final Score = Weighted Sum × 100

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/fingerprint-quality-assessment.git

cd fingerprint-quality-assessment
```

Create a virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit App

```bash
streamlit run quality_app.py
```

The application will open in your browser at

```
http://localhost:8501
```

---

## Running Unit Tests

```bash
pytest test_quality.py -v
```

Example Output

```
==========================
13 passed in 0.56s
==========================
```

---

## Sample Output

```
Composite Score : 87.5

Passed : True

Blur Score : 165.43

Brightness : 132.14

Glare : False

ROI Complete : True

Ridges Clear : True

Guidance :

Good capture — ready for processing.
```

---

## Technologies Used

- Python 3.9+
- OpenCV
- NumPy
- Streamlit
- PyTest

---

## Future Improvements

- NFIQ 2.0 Integration
- Deep Learning-based Quality Prediction
- Live Camera Capture
- Mobile Deployment
- Batch Processing
- Quality Report Export

---

## Author

**Prashanth**

Assignment 4

Contactless Fingerprint Quality Assessment Pipeline

---

## License

This project is developed for educational and assignment purposes.