"""
DrugMirror – Molecule Visualisation
=====================================
Generates a 2D depiction of a molecule from a SMILES string and returns it
as a base64-encoded PNG.  This module is **fully functional** — it has no
dependency on the ML model files.
"""

import base64
import io

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


def get_molecule_image(smiles: str, width: int = 400, height: int = 300) -> str | None:
    """Generate a 2D molecule depiction and encode it as a base64 PNG string.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule to draw.
    width : int
        Image width in pixels (default 400).
    height : int
        Image height in pixels (default 300).

    Returns
    -------
    str | None
        A base64-encoded PNG string (no ``data:image/png;base64,`` prefix) if
        the SMILES is valid, or ``None`` if parsing fails.

    Examples
    --------
    >>> img = get_molecule_image("CCO")
    >>> assert img is not None
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        # Use the high-quality Cairo-based renderer
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        # Convert SVG → PIL Image → PNG bytes → base64
        # Fallback to the simple PIL-based Draw.MolToImage if cairosvg is absent
        try:
            import cairosvg
            png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        except ImportError:
            # cairosvg not installed → fall back to RDKit's built-in PIL renderer
            pil_img = Draw.MolToImage(mol, size=(width, height))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

        return base64.b64encode(png_bytes).decode("utf-8")

    except Exception as exc:
        # Log the error but never crash the API
        print(f"[molecule] Failed to render SMILES '{smiles}': {exc}")
        return None
