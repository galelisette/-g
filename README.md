# EСG-Ваsеd Нumаn Diseаsе Rесоgnitiоn Systеm

## Overview

This project presents a web-based application for automated disease detection using electrocardiogram (ECG) analysis. The system employs machine learning techniques to identify five distinct pathological conditions (or their absence) from uploaded digitized ECG data in CSV format. The application utilizes a gradient boosting model trained on publicly available datasets to provide probabilistic disease assessments.

## Dataset and Validation

The training dataset was sourced from publicly available repositories, ensuring reproducibility and scientific validity. For demonstration purposes, three representative ECG samples are provided (samples.csv, samples2.csv, samples3.csv), derived from the PhysioNet database (physionet.org), a well-established repository for physiological signal data.

## System Architecture and Methodology

### Data Input Requirements

The application accepts ECG recordings in CSV format with the following specifications:

- **Temporal Resolution**: Minimum sampling frequency of 250 Hz
- **Amplitude Resolution**: Maximum quantization error of 10 μV
- **Signal Amplitude**: Minimum cardiac signal amplitude of 1 mV
- **Recording Duration**: Minimum 10-minute recording period (>600 cardiac cycles)
- **Data Structure**: Time-series format with time values (seconds) as independent variables and electrical potential (millivolts) as dependent variables

### Signal Processing Pipeline

1. **ECG Signal Acquisition**: The system processes high-resolution digitized ECG recordings meeting the aforementioned technical specifications.

2. **Feature Extraction and Codogram Generation**: The application generates a diagnostic codogram—a computational representation encoding amplitude increments, temporal intervals, and cardiac cycle phase characteristics. This codogram serves as a unique physiological signature containing pathology-specific information.

3. **Machine Learning Classification**: A gradient boosting algorithm analyzes the extracted features to compute probabilistic estimates for each target condition.

## Diagnostic Capabilities

The system provides probabilistic assessments for the following medical conditions:

- **Coronary Heart Disease** (Ischemic heart disease)
- **Vegetovascular Dystonia** (Autonomic dysfunction)
- **Peptic Ulcer Disease** (Gastric ulceration)
- **Cholelithiasis** (Gallstone disease)
- **Nodular Thyroid Goiter**
- **Healthy Control** (No pathology detected)

### Clinical Decision Threshold

The system outputs probability scores for each condition. **Probabilities exceeding 50% warrant clinical evaluation by appropriate medical specialists.**

## Scalability and Future Development

Current research indicates the feasibility of extending this methodology to detect approximately 40 additional pathological conditions. The system architecture is designed for scalable expansion, representing a prototype for comprehensive ECG-based diagnostic screening.

## Important Disclaimers

- This application is developed for **research and educational purposes only**
- The system represents a prototype commissioned by a medical facility (confidentiality restrictions apply)
- **This tool is not intended for clinical diagnosis or medical decision-making**
- All diagnostic interpretations should be validated by qualified healthcare professionals
- Users should consult appropriate medical specialists for any health concerns

## Technical Implementation

The web application is currently deployed on a private server infrastructure, which may result in variable response times. The system accepts both provided sample files and user-generated ECG data formatted according to the specified requirements.

## Data Sources and Acknowledgments

Training data sourced from publicly available physiological databases. Sample ECG recordings provided courtesy of PhysioNet (physionet.org), a research resource for complex physiologic signals maintained by the MIT Laboratory for Computational Physiology.

## Application Output Example

![ECG Analysis Results](https://raw.githubusercontent.com/galelisette/-g/5e857bff959031cfa12a7b9aa193a577e15c5bc7/imagine.png)

*Figure 1: Representative output showing probabilistic disease assessment results from the ECG analysis system*

---

*This project demonstrates the application of machine learning techniques in biomedical signal analysis and represents ongoing research in automated diagnostic screening technologies.*
