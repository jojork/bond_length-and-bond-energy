# bond_length-and-bond-energy
this is the script for finding the bond length and bond energy of 2 elements in a unit cell
from ase import Atoms
from ase.optimize import LBFGS
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW, FermiDirac
from ase.visualize import view
from ase.io import write

# Define range of lattice parameters
llist = np.linspace(1.5, 2.5, 10)
elist = []

for l in llist:
    b = l * 1.732  # approximate √3 * a

    # Build atomic structure (C and Fe pair)
    atoms = Atoms('CFe', [(0, 0, 10), (l / 2, b / 2, 10)])
    atoms.set_pbc((True, True, True))
    atoms.set_cell((l, b, 20.0))

    # Define calculator
    calc = GPAW(mode='lcao',
                basis='dzp',
                xc='PBE',
                kpts=(10, 10, 1),
                occupations=FermiDirac(0.05),
                convergence={'energy': 1e-5},
                txt=f'chain_lin_{l:.2f}.txt')

    atoms.calc = calc

    # Get potential energy (optional optimization commented)
    e = atoms.get_potential_energy()
    elist.append(e)
    print(f"Step: l={l:.2f} Å → E={e:.6f} eV")

    # === Save structure files ===
    xyz_filename = f'structure_{l:.2f}.xyz'
    txt_filename = f'structure_{l:.2f}.txt'
    calc.write(f'structure_{l:.2f}.gpw', mode='all')


    # Save .xyz (for ASE or visualization)
    write(xyz_filename, atoms)

    # Save .txt (custom readable format)
    with open(txt_filename, 'w') as f:
        f.write(f"Lattice parameters (Å): a={l:.6f}, b={b:.6f}, c=20.000000\n")
        f.write("Atomic positions (Å):\n")
        for i, atom in enumerate(atoms):
            f.write(f"{i+1:2d}  {atom.symbol:2s}  {atom.position[0]:10.6f}  {atom.position[1]:10.6f}  {atom.position[2]:10.6f}\n")

# === Analyze and plot results ===
elist = np.array(elist)
min_length = llist[elist.argmin()] / 2
print("\nMinimum energy bond length (Å):", min_length)

plt.figure(figsize=(6, 4))
plt.plot(llist / 2, elist, '-o', lw=1.8)
plt.xlabel('Bond Length (Å)')
plt.ylabel('Total Energy (eV)')
plt.title('C–Fe Bond Energy Curve')
plt.grid(True)
plt.tight_layout()
plt.savefig('bond_energy_curve.png', dpi=300)
plt.show()

# View final structure
view(atoms)
