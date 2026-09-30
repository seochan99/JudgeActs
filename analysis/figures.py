"""Build Figure 1 (recorded example and audit), the schematic overview, and statistical figures."""
from .overview_figure import main as render_overview
from .empirical_figures import main as render_empirical
from .hero_figure import main as render_hero
from .framework_figure import main as render_framework
from .gallery_figure import main as render_gallery
from .result_figures import main as render_results
from .appendix_tables import main as render_appendix_tables


def main():
    render_overview()
    render_empirical()
    render_hero()
    render_framework()
    render_gallery()
    render_results()
    render_appendix_tables()


if __name__ == "__main__":
    main()
