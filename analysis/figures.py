"""Build Figure 1 (recorded example and audit), the schematic overview, and statistical figures."""
from .overview_figure import main as render_overview
from .empirical_figures import main as render_empirical
from .hero_figure import main as render_hero
from .framework_figure import main as render_framework
from .gallery_figure import main as render_gallery
from .result_figures import main as render_results
from .appendix_tables import main as render_appendix_tables
from .ablation_figure import main as render_ablations
from .scale import main as render_scale


def main():
    render_overview()
    render_empirical()
    render_hero()
    render_framework()
    render_gallery()
    render_results()
    render_appendix_tables()
    render_ablations()
    render_scale()
    crop_all()


def crop_all():
    """Trim blank page margins so every figure fills the width it is placed at."""
    import subprocess
    from src.common import ROOT
    for pdf in sorted((ROOT / "paper/figures").glob("*.pdf")):
        subprocess.run(["pdfcrop", "--margins", "2", str(pdf), str(pdf)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
