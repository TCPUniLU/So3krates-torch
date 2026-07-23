"""Example: LoRA-finetune a bundled pretrained SO3LR checkpoint.

Loads one of the four bundled pretrained SO3LR checkpoints
(``v1``/``v2-s``/``v2-m``/``v2-l``) via ``load_pretrained_so3lr`` --
the same state_dict + settings.yaml loading path used by
``SO3LRCalculator`` -- converts it to LoRA format via
``setup_finetuning(..., finetune_choice="lora")``, and runs one real
forward+backward pass on a small, self-contained rattled NaCl
rock-salt supercell. It prints the loss and confirms that gradients
flow into the (and only the) LoRA parameters, doubling as a smoke test
you can run yourself.

Examples::

    python examples/finetuning/finetune_pretrained_so3lr.py --model v1
    python examples/finetuning/finetune_pretrained_so3lr.py \\
        --model v2-s --use_pme

Note on ``--r_max_lr``/``--dispersion_energy_cutoff_lr_damping``: every
bundled checkpoint has ``dispersion_energy_bool=True`` by default, and
``v1``'s checkpoint saves ``r_max_lr=None``/``dispersion_energy_
cutoff_lr_damping=None`` (v2's checkpoints save a very large
``r_max_lr=1000.0``, impractical for a real neighbor list). This
script always overrides both to sane finite defaults via
``architecture_overrides`` -- the same two values ``SO3LRCalculator``
itself defaults to (``r_max_lr=12.0``,
``dispersion_energy_cutoff_lr_damping=2.0``) -- whether or not
``--use_pme`` is set, since leaving ``dispersion_energy_cutoff_lr_
damping`` unset while ``r_max_lr`` is finite raises a ``ValueError``
(see ``blocks/physical_potentials.py``'s ``DispersionInteraction``).
"""

import argparse

import torch
from ase.build import bulk

from so3krates_torch.calculator.so3 import load_pretrained_so3lr
from so3krates_torch.data.atomic_data import AtomicData
from so3krates_torch.data.utils import KeySpecification, config_from_atoms
from so3krates_torch.tools import torch_geometric, utils
from so3krates_torch.tools.finetune import setup_finetuning


def build_nacl_structure(seed: int = 7):
    """Small, self-contained rattled 2x2x2 NaCl rock-salt supercell (16
    atoms) -- the same construction used throughout this repo's PME
    test suite (see
    ``tests/test_v2_parity.py::_build_rattled_nacl_structure``)."""
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.6402).repeat(
        (2, 2, 2)
    )
    atoms.rattle(stdev=0.05, seed=seed)
    return atoms


def build_batch(atoms, r_max: float, r_max_lr, theory_level: int = 0):
    """Convert an ``ase.Atoms`` structure into a single-graph SO3LR
    batch. Setting ``atoms.info["theory_level"]`` is required for the
    v2 checkpoints (``num_theory_levels=16``); it is a harmless no-op
    for v1 (``num_theory_levels=1``)."""
    atoms.info["theory_level"] = theory_level
    config = config_from_atoms(atoms, key_specification=KeySpecification())
    z_table = utils.AtomicNumberTable(list(range(1, 119)))
    data = AtomicData.from_config(
        config, z_table=z_table, cutoff=r_max, cutoff_lr=r_max_lr
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[data], batch_size=1, shuffle=False, drop_last=False
    )
    return next(iter(loader))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["v1", "v2-s", "v2-m", "v2-l"],
        help="Which bundled pretrained SO3LR checkpoint to finetune.",
    )
    parser.add_argument(
        "--use_pme",
        action="store_true",
        help="Enable PME (k-space) electrostatics instead of the "
        "real-space-only default.",
    )
    parser.add_argument("--pme_smearing", type=float, default=0.5)
    parser.add_argument("--pme_mesh_spacing", type=float, default=0.25)
    parser.add_argument(
        "--r_max_lr",
        type=float,
        default=12.0,
        help="Long-range (electrostatics/dispersion) cutoff.",
    )
    parser.add_argument(
        "--dispersion_energy_cutoff_lr_damping",
        type=float,
        default=2.0,
    )
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    architecture_overrides = {
        "r_max_lr": args.r_max_lr,
        "dispersion_energy_cutoff_lr_damping": (
            args.dispersion_energy_cutoff_lr_damping
        ),
    }
    if args.use_pme:
        architecture_overrides.update(
            use_pme=True,
            pme_smearing=args.pme_smearing,
            pme_mesh_spacing=args.pme_mesh_spacing,
        )

    print(f"Loading pretrained SO3LR checkpoint {args.model!r}...")
    model = load_pretrained_so3lr(
        args.model, device=args.device, **architecture_overrides
    )

    model = setup_finetuning(
        model,
        finetune_choice="lora",
        device_name=args.device,
        lora_rank=args.lora_rank,
        seed=args.seed,
    )

    atoms = build_nacl_structure()
    batch = build_batch(atoms, r_max=model.r_max, r_max_lr=model.r_max_lr).to(
        args.device
    )
    batch.positions.requires_grad_(True)

    out = model(batch.to_dict(), training=True, compute_force=True)
    loss = out["energy"].pow(2).sum() + out["forces"].pow(2).sum()
    loss.backward()

    lora_params = [
        (name, param)
        for name, param in model.named_parameters()
        if "lora_" in name
    ]
    assert lora_params, "No LoRA parameters found after setup_finetuning"
    non_lora_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" not in name
    ]

    def _has_grad(param):
        return param.grad is not None and torch.any(param.grad != 0)

    # LoRA's "B" matrices are zero-initialized (so the adapter is a
    # no-op at the start of finetuning), which has the well-known
    # side effect that, on this very first backward pass, d(loss)/dA
    # is analytically zero (it is proportional to B) while d(loss)/dB
    # is not (it is proportional to A, which is not zero). Both A and
    # B are trainable and will receive gradients from the second
    # optimizer step onward, once B stops being exactly zero -- so
    # reporting "B" separately from "A" here avoids the false alarm
    # of e.g. "only half of the LoRA parameters got gradients".
    a_params = [(n, p) for n, p in lora_params if "lora_A_" in n]
    b_params = [(n, p) for n, p in lora_params if "lora_B_" in n]
    n_a_grad = sum(1 for _, p in a_params if _has_grad(p))
    n_b_grad = sum(1 for _, p in b_params if _has_grad(p))
    n_with_grad = n_a_grad + n_b_grad

    print(f"model={args.model} use_pme={args.use_pme}")
    print(f"loss={loss.item():.6f}")
    print(
        f"LoRA parameters with nonzero gradients: "
        f"{n_with_grad}/{len(lora_params)} "
        f"(A: {n_a_grad}/{len(a_params)}, B: {n_b_grad}/{len(b_params)} "
        f"-- A is expected to be 0 on this first step, see comment "
        f"above)"
    )
    print(
        f"Non-LoRA trainable parameters: {len(non_lora_trainable)} "
        f"(setup_finetuning's freeze_shifts/freeze_scales default to "
        f"False, so atomic_energy_output_block.energy_shifts/"
        f"energy_scales stay trainable alongside LoRA by default)"
    )


if __name__ == "__main__":
    main()
