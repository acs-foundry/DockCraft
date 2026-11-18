//! Layer 0 (`struct-core`): 3D biomolecular topology, PDB/mmCIF/MOL2/SDF/PDBQT parsers,
//! pH 7.4 protonation, Gasteiger-Marsili PEOE partial charges, AutoDock4 typing, and 3D SMILES conformers.

#![forbid(unsafe_code)]

pub mod chemistry;
pub mod error;
pub mod formats;
pub mod types;

pub use chemistry::{
    MAX_SMILES_3D_ATOMS, apply_torsion_angles, assign_ad4_types_and_flags,
    assign_gasteiger_charges, build_rotatable_bonds, cross3, embed_3d_coordinates,
    ideal_bond_length, norm3, parse_smiles_to_3d, perceive_all, prepare_receptor,
    randomize_torsions,
};
pub use error::StructError;
pub use formats::{
    MAX_LIGAND_ATOMS, MAX_RECEPTOR_ATOMS, parse_mmcif, parse_mol2, parse_pdb, parse_pdbqt,
    parse_sdf_3d_with_limit, parse_sdf_mol_block, write_mmcif, write_mol2, write_pdb,
    write_pdbqt_ligand, write_pdbqt_receptor, write_sdf_3d,
};
pub use types::{
    Ad4Type, Atom3D, Bond3D, BondOrder, Molecule3D, RotatableBond, element_atomic_number,
    element_covalent_radius, element_weight, normalize_element,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pdb_mmcif_mol2_sdf_and_pdbqt_with_charges_and_ph74() -> Result<(), StructError> {
        let pdb = "\
HEADER    MINI_POCKET
ATOM      1  CA  ALA A  10       1.000   2.000   3.000  1.00 12.00           C
ATOM      2  NZ  LYS A  11       4.000   2.000   3.000  1.00 15.00           N
HETATM    3  O   HOH A 201      10.000  10.000  10.000  1.00 30.00           O
END
";
        let prot = parse_pdb(pdb, true)?;
        assert_eq!(prot.atoms.iter().filter(|a| a.element != "H").count(), 2);
        assert_eq!(prot.atoms[1].formal_charge, 1);

        let cif = write_mmcif(&prot);
        let prot2 = parse_mmcif(&cif, true)?;
        assert_eq!(prot2.heavy_atom_count(), 2);

        let benzoic = parse_smiles_to_3d("c1ccccc1C(=O)O", Some("BenzoicAcid"))?;
        assert_eq!(benzoic.heavy_atom_count(), 9);
        assert!(benzoic.atoms.iter().any(|a| a.formal_charge == -1));
        assert!(benzoic.atoms.iter().any(|a| a.aromatic));
        assert_eq!(benzoic.rotatable_bonds.len(), 1);
        // C7H5O2- (pH 7.4 carboxylate): MW includes implicit aromatic ring C-H hydrogens (~121.12 Da)
        assert!((benzoic.molecular_weight() - 121.12).abs() < 0.2);

        let benzene = parse_smiles_to_3d("c1ccccc1", Some("Benzene"))?;
        assert!((benzene.molecular_weight() - 78.11).abs() < 0.1);
        assert_eq!(benzene.formula(), "C6H6");
        // Carbonyl carbon attached to oxygen must NOT be flagged hydrophobic
        let c_carbonyl = benzoic
            .atoms
            .iter()
            .find(|a| a.element == "C" && !a.aromatic)
            .ok_or(StructError::EmptyInput)?;
        assert!(!c_carbonyl.hydrophobic);

        let sdf = write_sdf_3d(&[&benzoic]);
        let sdf_back = parse_sdf_3d_with_limit(&sdf, 10)
            .into_iter()
            .next()
            .ok_or(StructError::EmptyInput)??;
        assert_eq!(sdf_back.heavy_atom_count(), 9);
        assert!((sdf_back.rmsd_heavy(&benzoic).unwrap_or(99.0)) < 1e-3);

        let mol2 = write_mol2(&benzoic);
        let mol2_back = parse_mol2(&mol2)?;
        assert_eq!(mol2_back.heavy_atom_count(), 9);

        let pdbqt = write_pdbqt_ligand(&benzoic);
        assert!(pdbqt.contains("ROOT") && pdbqt.contains("TORSDOF 1"));
        let pdbqt_back = parse_pdbqt(&pdbqt)?;
        assert_eq!(pdbqt_back.heavy_atom_count(), 9);
        Ok(())
    }
}
