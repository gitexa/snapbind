# Protein Binding Site Analysis Pipeline Documentation

This documentation covers three Jupyter notebooks that form a complete pipeline for protein binding site discovery, data augmentation, and dataset merging for machine learning applications.

## Overview

The pipeline consists of three main components:
1. **discovery.ipynb** - Extracts binding sites from BindingDB and PDB structures
2. **augment.ipynb** - Augments the dataset with mutated protein sequences
3. **merge.ipynb** - Creates negative examples and merges positive/negative datasets

---

## 1. discovery.ipynb

### Purpose
Automatically extracts protein binding sites from BindingDB data by:
- Loading BindingDB TSV files
- Downloading PDB structures
- Identifying binding sites with ligand-protein interactions
- Creating binding arrays showing which residues participate in binding
- Exporting comprehensive results with full sequence information

### Key Features
- **Robust Data Loading**: Handles large BindingDB files with multiple encoding strategies
- **Threaded Processing**: Uses multiple threads for efficient PDB structure processing
- **Binding Site Detection**: Identifies meaningful ligands and contact residues within 5Å cutoff
- **Sequence Mapping**: Maps binding interactions to complete protein sequences
- **Comprehensive Export**: Saves results with binding arrays, sequences, and structural information

### Requirements
```python
pip install biopython pandas numpy matplotlib
```

### Main Components

#### BindingDBLoader Class
```python
class BindingDBLoader:
    @staticmethod
    def load_bindingdb_data(filepath, max_rows=None, sample_fraction=None)
```
- Loads BindingDB TSV files (fetch from https://www.bindingdb.org/) with error handling
- Supports sampling for large datasets
- Multiple encoding strategies for robust loading

#### ThreadedBindingSiteExtractor Class
```python
class ThreadedBindingSiteExtractor:
    def __init__(self, output_dir="pdb_structures", contact_cutoff=5.0, max_workers=4)
```
- Downloads PDB structures automatically
- Extracts complete protein sequences
- Identifies ligands (excludes water, ions, buffers)
- Finds contact residues within specified cutoff
- Creates binding arrays mapping interactions to sequence positions

### Usage Example
```python
# Configuration
BINDINGDB_FILE = "BindingDB_All.tsv"
MAX_STRUCTURES = 100000
SAMPLE_FRACTION = 0.04  # 4% of dataset
OUTPUT_FILE = "binding_sites_with_sequences.csv"

# Run extraction
binding_sites, results_df, summary = extract_binding_sites_from_bindingdb(
    BINDINGDB_FILE, 
    max_structures=MAX_STRUCTURES,
    sample_fraction=SAMPLE_FRACTION
)
```

### Output Format
The resulting CSV contains:
- **Basic Info**: PDB ID, ligand name, contact residues count
- **Structural Data**: Ligand center coordinates, binding site volume
- **Sequence Data**: Complete protein sequences for each chain
- **Binding Arrays**: JSON arrays showing binding positions (1=binding, 0=non-binding)
- **Contact Details**: Residue numbers, distances, and contact information

### Performance Notes
- Start with `SAMPLE_FRACTION=0.1` for testing
- Increase `MAX_WORKERS` based on CPU cores
- Typical processing: ~10-50 structures per minute depending on complexity

---

## 2. augment.ipynb

### Purpose
Augments the protein dataset by creating mutated versions of non-binding protein sequences to balance the dataset for machine learning applications.

### Key Features
- **Binding Detection**: Identifies entries with no binding interactions
- **Point Mutations**: Introduces random amino acid substitutions
- **Controlled Augmentation**: Creates specified number of variants per non-binding entry
- **Validation**: Verifies augmentation results and class balance

### Main Components

#### ProteinDataAugmenter Class
```python
class ProteinDataAugmenter:
    def __init__(self, csv_path: str)
```

#### Key Methods
- `identify_no_binding_entries()`: Finds proteins with all-zero binding arrays
- `mutate_sequence()`: Introduces random point mutations
- `augment_dataset()`: Creates augmented versions with unique identifiers
- `validate_augmentation()`: Checks results and provides statistics

### Usage Example
```python
# Configuration
input_file = "merged_protein_dataset_ext.csv"
output_file = "augmented_protein_dataset.csv"
augmentation_factor = 20  # Create 20 variants per non-binding entry

# Initialize and run
augmenter = ProteinDataAugmenter(input_file)
augmenter.load_data()
augmented_dataset = augmenter.augment_dataset(augmentation_factor=augmentation_factor)
augmenter.save_augmented_dataset(augmented_dataset, output_file)
```

### Mutation Strategy
- Random amino acid substitutions (1-10 mutations per sequence)
- Preserves sequence length and structure
- Creates variants with modified binding properties
- Maintains original binding arrays (all zeros for non-binding sequences)

### Output
- Original dataset + augmented variants
- Augmented entries have modified PDB IDs: `{original_pdb}_aug_{id}`
- Validation statistics showing class balance improvement

---

## 3. merge.ipynb

### Purpose
Creates a balanced dataset by generating negative examples (non-binding proteins) and merging them with positive examples (binding proteins) for supervised learning.

### Key Features
- **Negative Example Generation**: Processes specific protein types that don't bind small molecules
- **Exact Same Processing**: Uses identical structure extraction as discovery.ipynb
- **All-Zero Binding Arrays**: Creates negative examples with no binding sites
- **Dataset Balancing**: Merges positive and negative examples with class labels

### Target Protein Types for Negative Examples
- **Fabs (Antibodies)**: Bind proteins, not small molecules
- **Structural Proteins**: Elastin, Fibrin, Keratin
- **Nucleases**: Bind DNA, not small molecules

### Main Components

#### NegativeExampleExtractor Class
```python
class NegativeExampleExtractor:
    def __init__(self, output_dir="negative_pdb_structures", max_workers=4)
```
- Uses identical PDB processing as discovery.ipynb
- Creates all-zero binding arrays for negative examples
- Maintains same data structure for consistency

### Usage Example
```python
# Configuration
NEGATIVE_CSV = "negative.csv"  # List of non-binding PDB IDs
POSITIVE_CSV = "binding_sites_with_sequences_30000.csv"
OUTPUT_CSV = "merged_protein_dataset.csv"
MAX_STRUCTURES = 30000

# Process negative examples
extractor = NegativeExampleExtractor(max_workers=10)
negative_examples, failed_pdbs = extractor.process_threaded(negative_pdb_ids)

# Merge datasets
merge_with_positive_dataset(negative_df, POSITIVE_CSV, OUTPUT_CSV)
```

### Output Format
- **Balanced Dataset**: Positive and negative examples combined
- **Class Labels**: `is_binding` column (1=binding, 0=non-binding)
- **Consistent Structure**: Same columns as discovery.ipynb output
- **Shuffled Data**: Randomly ordered for training

---

## Pipeline Workflow

### Complete Pipeline Execution
```python
# Step 1: Discovery (discovery.ipynb)
binding_sites = extract_binding_sites_from_bindingdb("BindingDB_All.tsv")
# Output: binding_sites_with_sequences.csv

# Step 2: Augmentation (augment.ipynb) 
augmenter = ProteinDataAugmenter("binding_sites_with_sequences.csv")
augmented_dataset = augmenter.augment_dataset(augmentation_factor=20)
# Output: augmented_protein_dataset.csv

# Step 3: Merging (merge.ipynb)
extractor = NegativeExampleExtractor()
merged_dataset = merge_with_positive_dataset(negative_df, positive_csv)
# Output: merged_protein_dataset.csv
```

### Data Flow
```
BindingDB TSV → discovery.ipynb → Positive Examples (CSV)
                                        ↓
PDB IDs (negative.csv) → merge.ipynb → Negative Examples (CSV)
                                        ↓
                               Merged Dataset (CSV)
                                        ↓
                               augment.ipynb → Final Augmented Dataset
```

## Configuration Guidelines

### For Small-Scale Testing
```python
SAMPLE_FRACTION = 0.01    # 1% of BindingDB
MAX_STRUCTURES = 1000     # Limit PDB processing
MAX_WORKERS = 4           # Conservative threading
```

### For Production Use
```python
SAMPLE_FRACTION = 1.0     # Full BindingDB dataset
MAX_STRUCTURES = 100000   # Process many structures
MAX_WORKERS = 12          # Utilize available cores
```

## Common Issues and Solutions

### Memory Management
- Use `sample_fraction` for large BindingDB files
- Process in batches if memory is limited
- Monitor disk space for PDB downloads

### Threading Optimization
- Start with `max_workers=4` and increase based on performance
- Balance between CPU cores and network bandwidth
- Consider I/O limitations for PDB downloads

### Data Quality
- Check binding array consistency across pipeline steps
- Validate sequence lengths and binding position mappings
- Monitor success/failure rates in processing logs

## Output Data Structure

### Key Columns in Final Dataset
- `pdb_id`: Protein structure identifier
- `is_binding`: Class label (1=binding, 0=non-binding)
- `chain_X_sequence`: Complete protein sequence for chain X
- `chain_X_binding_array`: JSON array of binding positions
- `ligand_name`: Bound ligand identifier (or 'NONE' for negatives)
- `num_contact_residues`: Number of binding residues

This pipeline provides a complete solution for creating machine learning datasets from protein structural data, suitable for binding site prediction and related bioinformatics applications.
