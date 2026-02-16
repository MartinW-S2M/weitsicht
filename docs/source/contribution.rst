.. toctree::
   :maxdepth: 1
   :hidden:


Contribution Guide
==================


Every contribution is welcome!

- Report bugs or request features via GitHub Issues.
- Documentation (suggestions and improvement on examples and user guides).
- Submit code changes (new functionality, refactors, performance fixes, tests).
- Help triage issues by confirming bugs or proposing minimal reproductions.


Getting started
---------------
1. Fork the repository on GitHub and clone your fork:

   .. code-block:: bash

      git clone < link to your fork of repository >

2. Create a branch for your work:


   .. code-block:: bash

      git checkout -b "feature_short-description"

If you have already created an Issue like for bugs or feature request, you best name your branch starting with #ISSUE_NUMBER  (e.g. #3_adapt_docu)
Then you could close the Issue with git commit -m "fix #ISSUE_NUMBER" or "close #ISSUE_NUMBER"

3. Keep your fork in sync with ``dev`` to reduce merge conflicts.

Development environment
-----------------------
- Python 3.10 or newer is required.
- Create and activate a virtual environment (``python -m venv .venv`` or your preferred tool).
- Install ``weitsicht`` in editable mode with developer extras:

  .. code-block:: bash

     pip install -e .[develop]

  This pulls in ``pre-commit``, ``ruff``, ``pyright``, ``pytest``, ``coverage`` and the doc tooling defined in ``pyproject.toml``.

Alternatively do not install ``weitsicht`` but just the needed packages.
Since pip does not support installing dependencies without building and installing the package, we need to install all dependencies separately.

.. code-block:: bash

      # Packages needed for weitsicht to run
      pip install numpy, pyproj, shapely, rasterio, rtree, trimesh, tomli

      # Test, coverage
      pip install pytest, coverage,
      # Format, Linting, Type checking
      pip install pre-commit, pyright,ruff
      # Docu
      pip install  pydata_sphinx_theme, sphinx-design, sphinx_github_changelog>=1.2.1

For all the checking and testing there are config files in place. So you should run the tasks without any arguments.

Testing
-------
All tests must pass otherwise commits and pull requests will not be considered. If you need help on failing tests I am happy to assist.
Also within the foler **examples** all scirpts are tested. Therefore for running test example folder is needed and data from within.

- Run the test suite locally before opening a PR:

  .. code-block:: bash

      pytest


- The default options from ``pyproject.toml`` include ``--doctest-modules`` and enable the ``slow`` marker for long-running tests. To skip slow tests:

  .. code-block:: bash

      pytest -m "not slow"

Quality and consistency
-----------------------
To ensure that code style and quality is at some standard, it's mandatory to run checks via ``ruff`` and ``pyright`` for commits and pull requests.

- Formatting (ruff):

  .. code-block:: bash

      ruff format

- Linting (ruff):

  .. code-block:: bash

      ruff check
      # Some errors can be fixed automatically like unused imports or sorting of imports
      ruff check --fix


- Static typing (pyright). ``pyright`` is currently configured with ``typeCheckingMode = "standard"``:

  .. code-block:: bash

      pyright

Pre-commit
-----------------------
Pre-commit.ci is used to check all commits. You can check commits locally by using pre-commit.
To run pre-commits locally, install pre-commit once per clone and let them run before each commit.
pre-commit is configured to use the current local python installation or activated environment.
Otherwise there could be dependency issues for pyright.

.. code-block:: bash

   pre-commit install


Then it will run on each commit. It will perform ``ruff``and ``pyright``.
So if you run all the checking beforehand it should pass without problems.
You can check that via:

.. code-block:: bash

      pre-commit run --all-files


Coverage
--------
- Coverage is configured in ``pyproject.toml`` to run pytest. Typical workflow:

  .. code-block:: bash

      coverage run
      coverage html  # optional, view in htmlcov/index.html

Documentation
-------------
- Docs live in ``docs/source`` and use Sphinx. To build locally:

  .. code-block:: bash

      pip install -e .[docs] # if not already installed all developing dependencies
      cd docs
      make html

- Open ``docs/build/html/index.html`` in a browser to preview the site.

Pull requests
-------------

- Keep changes focused and include a brief summary of what/why.
- Add or update tests and docs alongside code changes.

Ensure the checklist below passes before opening a PR:

- ``pytest`` (plus coverage if relevant)
- ``ruff format``
- ``ruff check``
- ``pyright``
- Docs build cleanly when documentation is touched.
- ``pre-commit run --all-files``


Every pull request will be checked automatically using pre-commit.ci

Support
-------
If you are unsure how to start or need feedback, open a GitHub Issue with the label ``question`` or ``help wanted`` and describe your idea or blocker. We are happy to help.
