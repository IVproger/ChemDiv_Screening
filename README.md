
# Molecule Screening Against Target Proteins

This project involves screening a molecule dataset against four proteins:
- **1O5M (FPT)**
- **6VKV (GAG)**
- **7L5E (XPO1)**
- **8QYR (MYH7)**

## Team Members:
- **Ivan Golov** – Lead Developer, ML engineer
- **Ruslan Lukin** –  Lead and domain consultant

## Project Overview:
The goal is to computationally screen the given molecular dataset for potential interactions with the target proteins. This process includes:
1. Preparing protein structures.
2. Running molecular docking simulations.
3. Analyzing binding affinities.
4. Filtering potential drug candidates.

## Project Repository Structure

This document provides an overview of the structure of the repository and describes the purpose of each directory and file.

### Root Directory
- `.dvc/`: Directory for DVC (Data Version Control) configuration and cache files.
  - `config`: DVC configuration file.
- `configs/`: Directory for configuration files.
  - `README.md`: Documentation for configuration setup.
- `data/`: Directory for all data files.
  - `BindingDB/`: Raw BindingDB data.
  - `proteins/`: Protein data files.
  - `screening_data/`: Data used for screening experiments.
- `datastore/`: Managed datastore files. _Should be created manually._
- `models/`: Contains trained models and related files.
- `notebooks/`: Jupyter notebooks for data analysis and visualization.
- `src/`: Source code for the project, including scripts for docking and analysis.
- `requirements.txt`: List of Python dependencies.
- `pyproject.toml`: Configuration file for Poetry and project metadata.

## How to Use:

1. Clone the repository:
```bash
git clone git@github.com:IVproger/ChemDiv_Screening.git
cd ChemDiv_Screening
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:

- **For Linux/MacOS:**
```bash
source .venv/bin/activate
```

- **For Windows:**
```bash
.\.venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -Ur requirements.txt
```

## Acknowledgments:
We would like to thank ChemDiv for providing access to their molecular dataset and protein structures. Special thanks to the contributors of open-source libraries used in this project.

## License:
This project is licensed under the MIT License. See the `LICENSE` file for details.
