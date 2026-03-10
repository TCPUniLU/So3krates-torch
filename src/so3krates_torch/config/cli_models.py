from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: str
    data_path: str
    output_file: str = "results.h5"
    ensemble_size: int = 1
    device: str = "cuda"
    batch_size: int = 5
    model_type: str = "so3lr"
    r_max_lr: Optional[float] = None
    multispecies: bool = False
    multihead_model: bool = False
    compute_dipole: bool = False
    compute_stress: bool = False
    compute_hirshfeld: bool = False
    compute_partial_charges: bool = False
    dispersion_energy_cutoff_lr_damping: float = 2.0
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"
    stress_key: str = "REF_stress"
    virials_key: str = "REF_virials"
    dipole_key: str = "REF_dipoles"
    charges_key: str = "REF_charges"
    total_charge_key: str = "charge"
    total_spin_key: str = "total_spin"
    hirshfeld_key: str = "REF_hirsh_ratios"
    head_key: str = "head"
    head: str = "head"
    dtype: Literal["float32", "float64"] = "float32"
    return_att: bool = False


class PreprocessArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    input: str
    output: str
    mode: Literal["raw", "preprocessed"]
    r_max: Optional[float] = None
    r_max_lr: Optional[float] = None
    energy_key: Optional[str] = None
    forces_key: Optional[str] = None
    stress_key: Optional[str] = None
    dipole_key: Optional[str] = None
    charges_key: Optional[str] = None
    total_charge_key: Optional[str] = None
    total_spin_key: Optional[str] = None
    description: Optional[str] = None
    dtype: Literal["float32", "float64"] = "float64"
    validate_output: bool = Field(default=False, alias="validate")
    batch_size: int = 100_000

    @model_validator(mode="after")
    def validate_preprocessed_requires_r_max(self):
        if self.mode == "preprocessed" and self.r_max is None:
            raise ValueError("--r-max is required when mode is 'preprocessed'")
        return self


class MetricArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: str
    data: str
    output_args: List[str] = ["energy", "forces"]
    batch_size: int = 16
    device: str = "cpu"
    save: str = "./"
    return_predictions: bool = False
    log_file: str = "test_ensemble.log"
    results_file: str = "ensemble_test_results.npz"
    r_max_lr: Optional[float] = None
    dispersion_energy_cutoff_lr_damping: float = 2.0
    energy_key: str = "REF_energy"
    forces_key: str = "REF_forces"
    stress_key: str = "REF_stress"
    virials_key: str = "REF_virials"
    dipole_key: str = "REF_dipoles"
    charges_key: str = "REF_charges"
    total_charge_key: str = "charge"
    total_spin_key: str = "total_spin"
    hirshfeld_key: str = "REF_hirsh_ratios"
    multihead_model: bool = False
    multihead_return_mean: bool = False
    head_key: str = "head"
    head_name: str = "head"


class MergeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: List[str]
    output: str
    description: Optional[str] = None
    batch_size: int = 100_000

    @model_validator(mode="after")
    def validate_min_inputs(self):
        if len(self.inputs) < 2:
            raise ValueError("At least 2 input files are required for merging")
        return self


class Jax2TorchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path_to_params: str
    path_to_hyperparams: str
    save_settings_path: Optional[str] = None
    save_state_dict_path: Optional[str] = None
    save_model_path: Optional[str] = None
    so3lr: bool = True
    use_defined_shifts: bool = False
    trainable_rbf: bool = False
    dtype: str = "float32"
    device: str = "cpu"

    @model_validator(mode="after")
    def validate_at_least_one_save_path(self):
        if (
            self.save_settings_path is None
            and self.save_state_dict_path is None
            and self.save_model_path is None
        ):
            raise ValueError(
                "At least one of --save_settings_path, "
                "--save_state_dict_path, or --save_model_path "
                "must be provided"
            )
        return self


class Torch2JaxArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path_to_state_dict: str
    path_to_hyperparams: str
    save_settings_path: Optional[str] = None
    save_params_path: Optional[str] = None
    so3lr: bool = True
    use_defined_shifts: bool = False
    trainable_rbf: bool = False
    dtype: str = "float32"

    @model_validator(mode="after")
    def validate_at_least_one_save_path(self):
        if self.save_settings_path is None and self.save_params_path is None:
            raise ValueError(
                "At least one of --save_settings_path or "
                "--save_params_path must be provided"
            )
        return self


class CreateLammpsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: str
    elements: List[str]
    head: Optional[str] = None
    dtype: Literal["float32", "float64"] = "float64"
