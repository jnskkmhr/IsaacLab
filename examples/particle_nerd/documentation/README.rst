Particle NeRD report exports
===========================

The canonical content is
``examples/particle_nerd/documentation/methodology.rst``. The report exporter reads
that file directly, renames the abstract to an executive summary, and adds
numbered sections and the A4 layout used by
``docs/_build/cable_admm_proxy_scaling_report.pdf``. It does not maintain a second
copy of the methodology prose.

From the repository root, regenerate figures and the report with:

.. code-block:: bash

   uv run --no-project --with matplotlib python examples/particle_nerd/documentation/gen_pipeline.py
   uv run --no-project --with matplotlib python examples/particle_nerd/documentation/gen_contact_geometry.py
   uv run --no-project --with matplotlib python examples/particle_nerd/documentation/gen_network_layers.py
   uv run --no-project --with matplotlib python examples/particle_nerd/documentation/gen_particle_selection.py
   uv run --no-project --with matplotlib python examples/particle_nerd/documentation/gen_dataset.py
   uv run --no-project --with docutils --with beautifulsoup4 --with playwright --with pymupdf \
     python examples/particle_nerd/documentation/build_report.py --date 2026-11-02

The report date is explicit for this revision; omit ``--date`` to use the local
build date. The exporter uses Google Chrome or Chromium when found on PATH;
``--browser /path/to/browser`` selects another installed executable. It requires
network access to load pinned MathJax 3.2.2 while rendering equations. Nimbus Sans
matches the reference typography when installed, with Arial as a fallback.

Outputs are ``docs/_build/particle_nerd_methodology.pdf`` and the corresponding
``.html``. The PDF contains the green first-page banner, running headers, and page
numbers. The HTML provides the report body and styling; the PDF adds page
furniture after browser pagination and replaces the figure previews with their
vector PDF originals. Regenerate figures before building the report. Output PDFs are build artifacts and are not
committed. Figure vector PDFs are in ``docs/_build/particle_nerd_figures/``;
small JPG previews live in ``examples/particle_nerd/documentation/figures/``.

``dataset_counts.json`` records the archived data URL, SHA-256, and aggregate
counts used by the dataset figure. The pipeline, collider-geometry, network-layer, and particle-selection figures are authored schematics, not measured data.
