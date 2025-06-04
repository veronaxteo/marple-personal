# Configuration System

This directory contains the refactored configuration system for the RSM simulation. 

## Files

- `schema.py` - Main configuration classes and data structures
- `default.yaml` - Default configuration file
- `visual.yaml` - Configuration optimized for visual evidence simulations
- `audio.yaml` - Configuration optimized for audio evidence simulations
- `config.py` - Legacy config classes (to be phased out)
- `params.py` - Deprecated legacy configuration (commented out)

## Configuration Structure

The main configuration is handled by `SimulationConfig` which contains:

### Trial Configuration (`TrialConfig`)
```yaml
trial:
  name: snack1                    # Trial name
  data_dir: "trials"              # Base directory for trial files
  suspect_subdir: "suspect/json"  # Subdirectory for suspect files
  detective_subdir: "detective/json" # Subdirectory for detective files
  file_extension: ".json"         # File extension for trial files
```

### Sampling Configuration (`SamplingConfig`)
```yaml
sampling:
  num_suspect_paths: 50           # Number of suspect paths to sample
  num_detective_paths: 1000       # Number of detective paths to sample
  max_steps: 25                   # Maximum steps for path generation
  max_steps_middle: 15            # Maximum steps for middle sequences
  naive_temp: 0.05                # Temperature for naive agent
  sophisticated_temp: 0.05        # Temperature for sophisticated agent
  cost_weight: 0.5                # Cost weight parameter
  noisy_planting_sigma: 0.0       # Noise for planting behavior
```

### Evidence Configuration (`EvidenceConfig`)
```yaml
evidence:
  type: visual                    # Evidence type: visual, audio, or multimodal
  
  # Visual evidence parameters
  visual_smoothing_sigma: 0.0
  sophisticated_detective_sigma: 1.0
  sophisticated_suspect_sigma: 1.0
  
  # Audio evidence parameters
  audio_similarity_sigma: 0.1
  audio_gt_step_size: 2
  
  # Environment parameters
  door_close_prob: 0.0
  crumb_sigma: 0.0
```

## Usage

### Basic Usage
```python
from src.configs import SimulationConfig

# Load from file
cfg = SimulationConfig.load("configs/visual.yaml")

# Load with CLI overrides
cfg = SimulationConfig.from_cli(
    cfg_file="configs/default.yaml",
    overrides=["sampling.num_detective_paths=500", "evidence.type=audio"]
)
```

### Command Line Usage
```bash
# Use default configuration
python src/run.py

# Use specific config file
python src/run.py --cfg src/configs/visual.yaml

# Override specific parameters
python src/run.py --cfg src/configs/audio.yaml --extra sampling.max_steps=30 evidence.audio_similarity_sigma=0.2
```

## Migration Notes

- The old `SimConfig` class is aliased to `SimulationConfig` for backward compatibility
- Legacy configuration files in `params.py` have been deprecated
- Trial file paths are now properly configured through the `TrialConfig` structure
- Evidence-specific parameters are organized under the `EvidenceConfig` structure

## Adding New Configurations

To add a new configuration variant:

1. Copy an existing YAML file (e.g., `visual.yaml`)
2. Modify the parameters as needed
3. Update the `log_dir` to avoid conflicts
4. Use with `--cfg path/to/new/config.yaml` 